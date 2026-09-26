"""
Issue #5 — strict leakage-free temporal evaluation.

This module does NOT retrain or restructure the models (Issues #1/#3's
EdgeClassifierWithAttr / WindowGraphClassifier are reused unchanged). It
fixes the two concrete leakage mechanisms Issue #5 identifies in the
*evaluation* code path:

  SCADANet (Part A): `src.experiments.runner.run_scadanet_temporal` and
  `src.experiments.baseline_runner.run_experiment_b2` both evaluate every
  test flow against `snapshots[-1]["edge_index"]` — the FINAL, fully
  streamed topology — regardless of which window that flow falls in.
  A flow in the first test window can therefore see IP-pair topology
  that was only ever observed in the last test window. This module's
  `scadanet_window_eval(mode="causal")` instead walks test windows in
  chronological order and, for window t, uses
  `snapshots[t - 1]["edge_index"]` — the topology cumulative through the
  window immediately BEFORE t, i.e. exactly what an online system would
  have observed at prediction time. `mode="fixed"` (T3) freezes the
  topology at the train/test boundary (`snapshots[split_window - 1]`)
  for every test window instead, as a second, stricter-still comparison
  point the issue asks for. The model itself is trained once and frozen
  before either eval loop runs (see "Important design choice" in the
  issue: frozen model + evolving observed topology, continual learning
  left for future work).

  BATADAL (Part B): `src.data.batadal.temporal_split_windows` slices an
  already-built list of overlapping sliding windows (WINDOW_SIZE=24h,
  STRIDE=6h) by position — adjacent windows share up to 18h of raw rows,
  so a window straddling the 70/30 boundary can appear effectively on
  both sides. `batadal_purged_split()` instead works in raw-ROW space:
  a window is TRAIN only if its full [start, start+window_size) row
  range ends at or before the boundary row, TEST only if it starts at or
  after the boundary row, and PURGED (dropped from both) if it straddles
  the boundary. This is the "simplest correct method" the issue
  explicitly says is acceptable — it guarantees zero shared raw rows
  between train and test by construction, not by a post-hoc check.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.metrics import compute_metrics, per_attack_recall

# Track these attack subtypes when present in a test window.
TRACKED_SCADANET_ATTACKS = [
    "vuln_scan", "modbus_fdi", "modbus_abuse", "insider_threat", "apt_exfil",
]


# ---------------------------------------------------------------------
# Part A — SCADANet strict temporal evaluation
# ---------------------------------------------------------------------

def scadanet_window_eval(
    model: torch.nn.Module,
    x: torch.Tensor,
    query_edges_t: torch.Tensor,
    edge_attr_t: torch.Tensor,
    y_t: torch.Tensor,
    df_t: pd.DataFrame,
    snapshots: List[Dict],
    split_window: int,
    n_windows: int,
    label_col: str,
    mode: str = "causal",
) -> Tuple[List[Dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk test windows [split_window, n_windows) in chronological order,
    evaluating the (already-trained, frozen) model once per window against
    only the topology that would have been observable at that point.

    mode="causal"  (T2): topology = snapshots[w - 1] — cumulative through
                          the window immediately before w.
    mode="fixed"   (T3): topology = snapshots[split_window - 1] for every
                          test window (training-time topology only).

    Returns (per_window_records, concatenated pred, true, probs) so the
    caller can both save the per-window CSV (diagnostics) and compute one
    overall aggregate metric block for Table T1.
    """
    if mode not in ("causal", "fixed"):
        raise ValueError(f"mode must be 'causal' or 'fixed', got {mode!r}")

    model.eval()
    fixed_topology = snapshots[split_window - 1]["edge_index"]
    records: List[Dict] = []
    all_pred, all_true, all_probs = [], [], []

    with torch.no_grad():
        for w in range(split_window, n_windows):
            w_mask = (df_t["window"] == w).to_numpy()
            w_pos = np.nonzero(w_mask)[0]
            if len(w_pos) == 0:
    # Record empty windows so the chronological audit remains complete.
                records.append({"window": w, "mode": mode, "n_flows": 0,
                                 "reason": "no flows fell in this window"})
                continue

            idx = torch.as_tensor(w_pos, dtype=torch.long)
            topo = fixed_topology if mode == "fixed" else snapshots[w - 1]["edge_index"]

            logits = model(x, topo, query_edges_t[:, idx], edge_attr_t[idx])
            probs = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            true = y_t[idx]

            m = compute_metrics(pred, true, probs)
            w_df = df_t.iloc[w_pos]
            attack_rows = w_df[w_df["is_attack"] == 1] if "is_attack" in w_df else w_df.iloc[0:0]
            attack_type_counts = attack_rows[label_col].value_counts().to_dict() if len(attack_rows) else {}
            per_attack = per_attack_recall(pred, true, w_df[label_col].to_numpy())
            tracked = {k: per_attack.get(k) for k in TRACKED_SCADANET_ATTACKS}

            records.append({
                "window": w, "mode": mode,
                "n_flows": int(len(idx)),
                "attack_rate": float(true.float().mean().item()),
                "attack_type_counts": attack_type_counts,
                "f1": m["f1"], "auprc": m["auprc"], "recall": m["recall"],
                "precision": m["precision"], "fpr": m["fpr"],
                "per_attack_recall": per_attack,
                "tracked_attack_recall": tracked,
            })
            all_pred.append(pred)
            all_true.append(true)
            all_probs.append(probs)

    agg_pred = torch.cat(all_pred) if all_pred else torch.empty(0, dtype=torch.long)
    agg_true = torch.cat(all_true) if all_true else torch.empty(0, dtype=torch.long)
    agg_probs = torch.cat(all_probs) if all_probs else torch.empty(0)
    return records, agg_pred, agg_true, agg_probs


def scadanet_future_leak_eval(
    model: torch.nn.Module,
    x: torch.Tensor,
    query_edges_t: torch.Tensor,
    edge_attr_t: torch.Tensor,
    y_t: torch.Tensor,
    test_idx: torch.Tensor,
    snapshots: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T1 — the PREVIOUS protocol, kept only as a historical-reference row
    in Table T1: every test flow (regardless of window) scored against
    `snapshots[-1]`, the fully-streamed final topology. This is exactly
    what `run_scadanet_temporal`/`run_experiment_b2` already compute; this
    wrapper exists so Table T1 can call one consistent function for all
    three rows instead of re-deriving T1's number from a different code
    path than T2/T3 use."""
    model.eval()
    eval_topology = snapshots[-1]["edge_index"]
    with torch.no_grad():
        logits = model(x, eval_topology, query_edges_t[:, test_idx], edge_attr_t[test_idx])
        probs = F.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
    return pred, y_t[test_idx], probs


def scadanet_graph_growth(
    df_t: pd.DataFrame, snapshots: List[Dict], src_col: str, dst_col: str,
) -> pd.DataFrame:
    """Figure T2 data — per-window new nodes / new IP pairs / cumulative
    totals. Node growth is computed here directly from df_t (snapshots
    only track cumulative edges); edge growth reuses snapshots'
    `n_edges_seen` rather than recomputing it, so the two are guaranteed
    consistent with what the eval loop above actually used as topology."""
    seen_nodes = set()
    prev_edges = 0
    rows = []
    for w, snap in enumerate(snapshots):
        w_df = df_t[df_t["window"] == w]
        window_nodes = set(w_df[src_col].astype(str)) | set(w_df[dst_col].astype(str))
        new_nodes = len(window_nodes - seen_nodes)
        seen_nodes |= window_nodes
        cum_edges = snap["n_edges_seen"]
        rows.append({
            "window": w,
            "new_nodes": new_nodes,
            "cumulative_nodes": len(seen_nodes),
            "new_ip_pairs": cum_edges - prev_edges,
            "cumulative_edges": cum_edges,
        })
        prev_edges = cum_edges
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Part B — BATADAL purged/embargoed temporal split
# ---------------------------------------------------------------------

def batadal_purged_split(
    n_rows: int, starts: List[int], window_size: int, split_frac: float,
) -> Tuple[List[int], List[int], List[int], Dict]:
    """Assign each window (by its position in `starts`) to train, test, or
    purged, using ROW ranges rather than window position — see module
    docstring for why. A window with raw range [s, s+window_size) is:
      - TRAIN  if s + window_size <= train_end_row (entirely before the boundary)
      - TEST   if s >= train_end_row (entirely at/after the boundary)
      - PURGED otherwise (straddles the boundary; dropped from both sides)

    Returns (train_window_idx, test_window_idx, purged_window_idx, meta),
    where the three index lists are positions into whatever list of
    per-window objects (window_graphs, starts, ...) they were derived
    from — same convention as `starts` itself.
    """
    train_end_row = int(n_rows * split_frac)
    train_idx, test_idx, purged_idx = [], [], []
    for i, s in enumerate(starts):
        e = s + window_size
        if e <= train_end_row:
            train_idx.append(i)
        elif s >= train_end_row:
            test_idx.append(i)
        else:
            purged_idx.append(i)

    meta = {
        "protocol": f"purged_{round(split_frac * 100)}_{round((1 - split_frac) * 100)}",
        "split_frac": split_frac,
        "n_rows": n_rows,
        "train_end_row": train_end_row,
        "window_size": window_size,
        "n_train_windows": len(train_idx),
        "n_test_windows": len(test_idx),
        "n_purged_windows": len(purged_idx),
    }
    return train_idx, test_idx, purged_idx, meta


def batadal_window_bounds_df(starts: List[int], window_size: int, idx: List[int]) -> pd.DataFrame:
    """Builds the tiny (window_start, window_end) frame `audit_split`
    (Issue #2) expects for its `window_start_col`/`window_end_col` path,
    for a chosen subset of windows (train, test, or purged). Reusing
    Issue #2's own overlap check here — rather than writing a second,
    parallel overlap-counting function — is how this issue verifies its
    own purge is correct: after purging, running `audit_split` on the
    resulting train/test frames should report zero overlapping windows,
    where before purging it reports exactly the historical T4 number."""
    return pd.DataFrame({
        "window_start": [starts[i] for i in idx],
        "window_end": [starts[i] + window_size for i in idx],
    })
