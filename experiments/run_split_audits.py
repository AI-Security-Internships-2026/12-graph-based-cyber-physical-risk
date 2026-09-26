"""
Issue #2 — run the split audit on all 5 protocols named in the issue and
write the required artifacts:

    experiments/results/q1/split_audits/scadanet_track_a_audit.json
    experiments/results/q1/split_audits/scadanet_track_b_audit.json
    experiments/results/q1/split_audits/scadanet_temporal_audit.json
    experiments/results/q1/split_audits/batadal_static_audit.json
    experiments/results/q1/split_audits/batadal_temporal_audit.json
    experiments/results/q1/split_audits/split_audit_summary.csv

Usage:
    python -m experiments.run_split_audits \
        --scadanet-path /path/to/scadanet.csv \
        --batadal-path BATADAL_dataset04.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.batadal import load_batadal_df, build_windowed_graphs
from src.data.scadanet import load_scadanet_df, build_graph_tensors, LABEL_COL, SRC_IP_COL, DST_IP_COL
from src.evaluation.audit import audit_split
from src.evaluation.splits import (
    scadanet_track_a_split, scadanet_track_b_split, batadal_static_cv_folds,
)
from src.data.scadanet import add_temporal_windows, temporal_split_positions

OUT_DIR = Path("experiments/results/q1/split_audits")
IS_NOT_NORMAL = lambda label: "normal" not in str(label).strip().lower()


def audit_scadanet(csv_path: str) -> dict:
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    results = {}

    # --- Track A (IP-held-out) ---
    train_idx, test_idx, _ = scadanet_track_a_split(gt.query_edges, gt.all_ips, seed=42)
    train_df, test_df = df.iloc[train_idx.numpy()], df.iloc[test_idx.numpy()]
    results["track_a"] = audit_split(
        train_df, test_df, protocol_name="scadanet_track_a",
        src_col=SRC_IP_COL, dst_col=DST_IP_COL, label_col=LABEL_COL,
        positive_label_check=IS_NOT_NORMAL,
    )

    # --- Track B (single-fixed-IP subtypes vs multi-IP subtypes) ---
    train_idx, test_idx, _ = scadanet_track_b_split(df, LABEL_COL, seed=42)
    train_df, test_df = df.iloc[train_idx.numpy()], df.iloc[test_idx.numpy()]
    results["track_b"] = audit_split(
        train_df, test_df, protocol_name="scadanet_track_b",
        src_col=SRC_IP_COL, dst_col=DST_IP_COL, label_col=LABEL_COL,
        positive_label_check=IS_NOT_NORMAL,
    )

    # --- Temporal (equal-count windows, chronological) ---
    df_t, _ = add_temporal_windows(df, gt.ip_index, n_windows=20)
    train_pos, test_pos, _ = temporal_split_positions(df_t)
    train_df, test_df = df_t.iloc[train_pos], df_t.iloc[test_pos]
    results["temporal"] = audit_split(
        train_df, test_df, protocol_name="scadanet_temporal",
        src_col=SRC_IP_COL, dst_col=DST_IP_COL, label_col=LABEL_COL,
        time_col="Time", positive_label_check=IS_NOT_NORMAL,
    )

    return results


def _batadal_window_df(window_labels, extra) -> pd.DataFrame:
    starts = extra["starts"]
    window_size = extra["window_size"]
    return pd.DataFrame({
        "window_start": starts,
        "window_end": [s + window_size for s in starts],
        "label": window_labels,
    })


def audit_batadal(csv_path: str) -> dict:
    df = load_batadal_df(csv_path)
    window_graphs, window_labels, extra = build_windowed_graphs(df)
    window_df = _batadal_window_df(window_labels, extra)
    results = {}

    # --- Static 5-fold CV: audit fold 0 as representative ---
    folds = batadal_static_cv_folds(window_labels, n_folds=5, seed=42)
    train_idx0, test_idx0 = folds[0]
    train_df = window_df.iloc[train_idx0]
    test_df = window_df.iloc[test_idx0]
    static_audit = audit_split(
        train_df, test_df, protocol_name="batadal_static_cv",
        window_start_col="window_start", window_end_col="window_end",
        label_col="label",
    )
    static_audit["notes"] = "audited fold 0/5 as representative; other 4 folds not individually audited"
    results["static_cv"] = static_audit

    # --- Temporal 70/30 chronological ---
    split_idx = int(len(window_graphs) * 0.7)
    train_df = window_df.iloc[:split_idx]
    test_df = window_df.iloc[split_idx:]
    results["temporal"] = audit_split(
        train_df, test_df, protocol_name="batadal_temporal",
        window_start_col="window_start", window_end_col="window_end",
        label_col="label",
    )

    return results


def build_summary_csv(scadanet_results: dict, batadal_results: dict, out_path: Path):
    rows = []
    combined = {
        "scadanet_track_a": scadanet_results["track_a"],
        "scadanet_track_b": scadanet_results["track_b"],
        "scadanet_temporal": scadanet_results["temporal"],
        "batadal_static_cv": batadal_results["static_cv"],
        "batadal_temporal": batadal_results["temporal"],
    }
    for name, audit in combined.items():
        identity = audit["identity_topology_overlap"]
        coverage = audit["attack_coverage"]
        temporal_w = audit["temporal_leakage_windows"]
        temporal_f = audit["temporal_leakage_flow"]

        rows.append({
            "protocol": name,
            "train_size": audit["train_size"],
            "test_size": audit["test_size"],
            "host_overlap_pct": (
                identity["host_overlap"]["pct_test_seen_in_train"]
                if identity.get("applicable") else None
            ),
            "pair_overlap_pct": (
                identity["pair_overlap"]["pct_test_seen_in_train"]
                if identity.get("applicable") else None
            ),
            "attack_types_covered": (
                f"{coverage['n_labels_evaluable_in_test']}/{coverage['n_labels_total']}"
                if coverage.get("applicable") else "N/A"
            ),
            "pct_test_nodes_unseen": (
                identity.get("pct_test_nodes_unseen") if identity.get("applicable") else None
            ),
            "pct_test_edges_unseen": (
                identity.get("pct_test_edges_unseen") if identity.get("applicable") else None
            ),
            "temporal_overlap": (
                temporal_w.get("pct_test_windows_overlapping_train")
                if temporal_w.get("applicable")
                else (temporal_f.get("raw_record_time_overlap") if temporal_f.get("applicable") else "N/A")
            ),
            "n_warnings": len(audit["warnings"]),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scadanet-path", default=None)
    parser.add_argument("--batadal-path", default="BATADAL_dataset04.csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scadanet_results = audit_scadanet(args.scadanet_path)
    for key, fname in [("track_a", "scadanet_track_a_audit.json"),
                        ("track_b", "scadanet_track_b_audit.json"),
                        ("temporal", "scadanet_temporal_audit.json")]:
        path = OUT_DIR / fname
        with open(path, "w") as f:
            json.dump(scadanet_results[key], f, indent=2, default=str)
        print(f"wrote {path}")
        for w in scadanet_results[key]["warnings"]:
            print(f"  {w}")

    batadal_results = audit_batadal(args.batadal_path)
    for key, fname in [("static_cv", "batadal_static_audit.json"),
                        ("temporal", "batadal_temporal_audit.json")]:
        path = OUT_DIR / fname
        with open(path, "w") as f:
            json.dump(batadal_results[key], f, indent=2, default=str)
        print(f"wrote {path}")
        for w in batadal_results[key]["warnings"]:
            print(f"  {w}")

    build_summary_csv(scadanet_results, batadal_results, OUT_DIR / "split_audit_summary.csv")


if __name__ == "__main__":
    main()
