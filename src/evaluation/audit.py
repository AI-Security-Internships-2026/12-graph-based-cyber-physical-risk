"""
Issue #2 — reusable split audit.

`audit_split()` describes what a train/test protocol actually asks a
model to generalize to: identity/topology overlap, attack-taxonomy
coverage, temporal leakage, distribution shift, and a set of descriptive
warnings. It does NOT judge whether a split is "bad" — a high host-overlap
number is a fact about the split, not automatically proof of a shortcut
(see "Do Not Do In This Issue": don't interpret overlap alone as
causation). It reports; a human (or a later issue) interprets.

Works for two shapes of split, selected by which optional columns are
passed:
  - flow-level (SCADANet): pass src_col/dst_col/label_col/time_col,
    train_df/test_df are row-per-flow.
  - window-level (BATADAL): pass window_start_col/window_end_col/
    label_col, train_df/test_df are row-per-window. No host identity
    exists at this granularity (BATADAL's topology is a fixed sensor
    correlation graph, not per-window) — audit_split reports that
    explicitly (node_overlap=1.0, "topology fixed across all windows")
    rather than silently omitting the section.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _overlap_stats(train_vals: set, test_vals: set) -> Dict:
    if not test_vals:
        return {"train_count": len(train_vals), "test_count": 0,
                "overlap_count": 0, "pct_test_seen_in_train": None,
                "pct_test_unseen": None}
    overlap = train_vals & test_vals
    return {
        "train_count": len(train_vals),
        "test_count": len(test_vals),
        "overlap_count": len(overlap),
        "pct_test_seen_in_train": len(overlap) / len(test_vals),
        "pct_test_unseen": 1 - len(overlap) / len(test_vals),
    }


def _identity_topology_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                src_col: Optional[str], dst_col: Optional[str]) -> Dict:
    if src_col is None or dst_col is None:
        return {
            "applicable": False,
            "reason": "no per-record host identity in this split granularity "
                      "(e.g. BATADAL's topology is a single fixed sensor "
                      "correlation graph shared by every window, not "
                      "per-window host identity)",
            "node_overlap": 1.0,
        }

    train_hosts = set(train_df[src_col].astype(str)) | set(train_df[dst_col].astype(str))
    test_hosts = set(test_df[src_col].astype(str)) | set(test_df[dst_col].astype(str))
    host_stats = _overlap_stats(train_hosts, test_hosts)

    train_pairs = set(zip(train_df[src_col].astype(str), train_df[dst_col].astype(str)))
    test_pairs = set(zip(test_df[src_col].astype(str), test_df[dst_col].astype(str)))
    pair_stats = _overlap_stats(train_pairs, test_pairs)

    return {
        "applicable": True,
        "host_overlap": host_stats,
        "pair_overlap": pair_stats,
        "pct_test_nodes_unseen": host_stats["pct_test_unseen"],
        "pct_test_edges_unseen": pair_stats["pct_test_unseen"],
    }


def _attack_coverage(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      label_col: Optional[str], src_col: Optional[str]) -> Dict:
    if label_col is None:
        return {"applicable": False, "reason": "no label column provided"}

    all_labels = sorted(set(train_df[label_col].unique()) | set(test_df[label_col].unique()))
    per_label = {}
    n_evaluable = 0
    for label in all_labels:
        tr_rows = train_df[train_df[label_col] == label]
        te_rows = test_df[test_df[label_col] == label]
        evaluable = len(te_rows) > 0
        n_evaluable += int(evaluable)
        entry = {
            "train_count": int(len(tr_rows)),
            "test_count": int(len(te_rows)),
            "evaluable": evaluable,
        }
        if src_col is not None:
            entry["unique_src_ips_overall"] = int(
                pd.concat([tr_rows[src_col], te_rows[src_col]]).astype(str).nunique()
            )
            entry["single_fixed_source_ip"] = entry["unique_src_ips_overall"] == 1
        per_label[str(label)] = entry

    return {
        "applicable": True,
        "per_label": per_label,
        "n_labels_total": len(all_labels),
        "n_labels_evaluable_in_test": n_evaluable,
        "pct_taxonomy_covered": n_evaluable / len(all_labels) if all_labels else None,
    }


def _temporal_leakage_flow(train_df: pd.DataFrame, test_df: pd.DataFrame,
                            time_col: Optional[str]) -> Dict:
    if time_col is None:
        return {"applicable": False, "reason": "no time column provided (non-temporal split)"}

    tr_min, tr_max = train_df[time_col].min(), train_df[time_col].max()
    te_min, te_max = test_df[time_col].min(), test_df[time_col].max()
    raw_overlap = not (tr_max <= te_min or te_max <= tr_min)
    return {
        "applicable": True,
        "train_time_range": [float(tr_min), float(tr_max)],
        "test_time_range": [float(te_min), float(te_max)],
        "train_duration": float(tr_max - tr_min),
        "test_duration": float(te_max - te_min),
        "raw_record_time_overlap": bool(raw_overlap),
    }


def _temporal_leakage_windows(train_df: pd.DataFrame, test_df: pd.DataFrame,
                               window_start_col: Optional[str],
                               window_end_col: Optional[str]) -> Dict:
    if window_start_col is None or window_end_col is None:
        return {"applicable": False, "reason": "no window bounds provided"}

    train_ranges = list(zip(train_df[window_start_col], train_df[window_end_col]))
    test_ranges = list(zip(test_df[window_start_col], test_df[window_end_col]))

    overlapping_pairs = 0
    for ts, te in test_ranges:
        for trs, tre in train_ranges:
            if ts < tre and trs < te:  # half-open interval overlap
                overlapping_pairs += 1
                break  # count each test window once, not per train window it touches

    return {
        "applicable": True,
        "n_test_windows": len(test_ranges),
        "n_test_windows_sharing_raw_rows_with_train": overlapping_pairs,
        "pct_test_windows_overlapping_train": (
            overlapping_pairs / len(test_ranges) if test_ranges else None
        ),
    }


def _distribution_shift(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         label_col: Optional[str], positive_label_check=None) -> Dict:
    """positive_label_check: callable(label_value) -> bool, identifies the
    "attack" class when label_col isn't already binary 0/1. For SCADANet's
    Attack_Type strings, pass a lambda checking 'normal' not in the string."""
    if label_col is None:
        return {"applicable": False, "reason": "no label column provided"}

    if positive_label_check is None:
        is_attack_train = train_df[label_col].astype(bool)
        is_attack_test = test_df[label_col].astype(bool)
    else:
        is_attack_train = train_df[label_col].apply(positive_label_check)
        is_attack_test = test_df[label_col].apply(positive_label_check)

    train_rate = float(is_attack_train.mean())
    test_rate = float(is_attack_test.mean())
    rel_change = (test_rate - train_rate) / train_rate if train_rate else None

    per_label_rates = None
    if positive_label_check is not None:
        tr_rates = train_df[label_col].value_counts(normalize=True).to_dict()
        te_rates = test_df[label_col].value_counts(normalize=True).to_dict()
        all_labels = set(tr_rates) | set(te_rates)
        per_label_rates = {
            str(l): {"train_rate": tr_rates.get(l, 0.0), "test_rate": te_rates.get(l, 0.0),
                     "relative_change": (
                         (te_rates.get(l, 0.0) - tr_rates.get(l, 0.0)) / tr_rates[l]
                         if tr_rates.get(l) else None
                     )}
            for l in all_labels
        }

    return {
        "applicable": True,
        "train_base_rate": train_rate,
        "test_base_rate": test_rate,
        "relative_base_rate_change": rel_change,
        "per_label_base_rates": per_label_rates,
    }


def _generate_warnings(identity: Dict, coverage: Dict, temporal_flow: Dict,
                        temporal_windows: Dict, protocol_name: str) -> List[str]:
    warnings = []

    if coverage.get("applicable"):
        n_total = coverage["n_labels_total"]
        n_eval = coverage["n_labels_evaluable_in_test"]
        if n_total > 1 and n_eval < n_total:
            warnings.append(
                f"WARNING: {n_total - n_eval}/{n_total} attack classes "
                f"absent from test set under '{protocol_name}'."
            )
        for label, entry in coverage["per_label"].items():
            if entry.get("single_fixed_source_ip"):
                warnings.append(
                    f"WARNING: attack subtype '{label}' is associated with "
                    f"one fixed source IP (structurally near-impossible to "
                    f"miss once that IP's degree pattern is learned)."
                )

    if identity.get("applicable"):
        pct_seen = identity["host_overlap"]["pct_test_seen_in_train"]
        if pct_seen is not None and pct_seen >= 0.9:
            warnings.append(
                f"WARNING: {pct_seen*100:.0f}% of test hosts were seen "
                f"in training under '{protocol_name}'."
            )

    if temporal_flow.get("applicable") and temporal_flow.get("raw_record_time_overlap"):
        warnings.append(
            f"WARNING: train/test raw-record time ranges overlap under '{protocol_name}'."
        )

    if temporal_windows.get("applicable"):
        pct = temporal_windows["pct_test_windows_overlapping_train"]
        if pct:
            warnings.append(
                f"WARNING: {pct*100:.0f}% of test sliding windows share "
                f"raw observations with a train window under '{protocol_name}'."
            )

    return warnings


def audit_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol_name: str,
    src_col: Optional[str] = None,
    dst_col: Optional[str] = None,
    label_col: Optional[str] = None,
    time_col: Optional[str] = None,
    window_start_col: Optional[str] = None,
    window_end_col: Optional[str] = None,
    positive_label_check=None,
) -> Dict:
    """Run the full Issue #2 audit on one train/test split and return a
    single JSON-serializable dict. Every section is present even when not
    applicable (marked applicable=False with a reason) — this issue's own
    philosophy carried over from Issue #1: never silently omit a required
    field, always say why it's null."""
    identity = _identity_topology_overlap(train_df, test_df, src_col, dst_col)
    coverage = _attack_coverage(train_df, test_df, label_col, src_col)
    temporal_flow = _temporal_leakage_flow(train_df, test_df, time_col)
    temporal_windows = _temporal_leakage_windows(train_df, test_df, window_start_col, window_end_col)
    distribution = _distribution_shift(train_df, test_df, label_col, positive_label_check)
    warnings = _generate_warnings(identity, coverage, temporal_flow, temporal_windows, protocol_name)

    return {
        "protocol_name": protocol_name,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "identity_topology_overlap": identity,
        "attack_coverage": coverage,
        "temporal_leakage_flow": temporal_flow,
        "temporal_leakage_windows": temporal_windows,
        "distribution_shift": distribution,
        "warnings": warnings,
    }
