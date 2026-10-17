"""Strict leakage-free temporal evaluation. This module does not retrain or restructure the
models.

SCADANet: `run_scadanet_temporal` and `run_experiment_b2` evaluate every test flow
against the final, fully streamed topology, so a flow in the first test window can see
IP-pair edges first observed in a later window. `scadanet_window_eval(mode="causal")`
instead walks test windows in order and, for window t, uses `snapshots[t - 1]`, i.e.
only what an online system would have observed. `mode="fixed"` freezes the topology at
the train/test boundary (`snapshots[split_window - 1]`) for every test window. The model
is trained once and frozen before either loop runs.

BATADAL: `temporal_split_windows` slices overlapping windows (24h, stride 6h) by
position, so a window straddling the 70/30 boundary can appear on both sides.
`batadal_purged_split()` works in raw-row space: a window is TRAIN only if its full row
range ends at or before the boundary row, TEST only if it starts at or after it, and
PURGED (dropped) if it straddles the boundary. This guarantees zero shared raw rows
between train and test by construction.
"""
from src.utils.reproducibility import replayable_split
from typing import Dict, List, Optional, Tuple

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.metrics import compute_metrics, per_attack_recall

# Reported when present in the data; a subtype absent from a given window's positives is
# simply absent from that window's per_attack_recall dict (per_attack_recall's own "no
# test rows" case), not an error.
TRACKED_SCADANET_ATTACKS = [
    "vuln_scan", "modbus_fdi", "modbus_abuse", "insider_threat", "apt_exfil",
]


# ---------------------------------------------------------------------
# SCADANet strict temporal evaluation
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
    inference_ms=0.0
    all_pred, all_true, all_probs = [], [], []

    with torch.no_grad():
        for w in range(split_window, n_windows):
            w_mask = (df_t["window"] == w).to_numpy()
            w_pos = np.nonzero(w_mask)[0]
            if len(w_pos) == 0:
                # Empty test window (possible with a small n_windows on a
                # small dataset) — record it rather than silently skipping,
                # per this repo's "applicable=False with a reason" norm.
                records.append({"window": w, "mode": mode, "n_flows": 0,
                                 "reason": "no flows fell in this window"})
                continue

            idx = torch.as_tensor(w_pos, dtype=torch.long)
            topo = fixed_topology if mode == "fixed" else snapshots[w - 1]["edge_index"]

            if x.is_cuda:torch.cuda.synchronize(x.device)
            infer_start=time.perf_counter()
            logits = model(x, topo, query_edges_t[:, idx], edge_attr_t[idx])
            if x.is_cuda:torch.cuda.synchronize(x.device)
            inference_ms += (time.perf_counter()-infer_start)*1000
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
    model._last_inference_ms=inference_ms
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
# BATADAL purged/embargoed temporal split
# ---------------------------------------------------------------------

@replayable_split
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
    """Builds the tiny (window_start, window_end) frame `audit_split` expects for its
    `window_start_col`/`window_end_col` path, for a chosen subset of windows (train,
    test, or purged).
    """
    return pd.DataFrame({
        "window_start": [starts[i] for i in idx],
        "window_end": [starts[i] + window_size for i in idx],
    })


def scadanet_topology_audit(df_t, snapshots, split_window, n_windows, ip_index, mode="causal"):
    """Independently verify each supplied topology against earlier observed rows."""
    records=[]
    for w in range(split_window,n_windows):
        cutoff=w if mode=="causal" else split_window
        earlier=df_t[df_t["window"] < cutoff]
        allowed={(ip_index[str(a)],ip_index[str(b)]) for a,b in zip(earlier["Source"],earlier["Destination"])}
        snapshot_window=cutoff-1
        used=set(map(tuple,snapshots[snapshot_window]["edge_index"].detach().cpu().T.tolist()))
        unexpected=used-allowed
        rec={"test_window":w,"mode":mode,"topology_through_window":snapshot_window,
             "n_used_edges":len(used),"n_unobserved_edges":len(unexpected),
             "uses_only_prior_windows":not unexpected and snapshot_window<w,
             "exact_observed_topology":used==allowed,
             "time_semantics":"ordered equal-count windows; equal timestamps may span windows"}
        if not rec["uses_only_prior_windows"] or not rec["exact_observed_topology"]:
            raise AssertionError(f"Noncausal or inconsistent topology: {rec}")
        records.append(rec)
    return records
