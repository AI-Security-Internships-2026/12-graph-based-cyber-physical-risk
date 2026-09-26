"""
Issue #6 — host-relative feature normalization + shortcut mitigation,
orchestration.

    python -m src.experiments.host_norm_runner --protocol track_b --seed 42
    python -m src.experiments.host_norm_runner --protocol temporal --seed 42
    python -m src.experiments.host_norm_runner --protocol both --seed 42

Runs H1-H4 (section 3) on both required protocols (section 4): SCADANet
Track B (Issue #1) and the strict SCADANet temporal protocol (Issue #5,
`mode="causal"` / T2 — the "strict" one, since Issue #6's premise is that
host behavior may drift, which only the causal/evolving-topology
protocol actually tests, not T1's future-leaking one or T3's frozen
one). H1-H4 differ ONLY in (a) which edge_attr was used to train and (b)
which threshold is applied at test time:

    H1 = model_original,       threshold 0.5
    H2 = model_original,       threshold = validation-selected
    H3 = model_host_normalized, threshold 0.5
    H4 = model_host_normalized, threshold = validation-selected

so exactly two models are trained per protocol (not four), and each is
scored twice.

Optional H5 (section 3: "global normalization matched control") is NOT
implemented here — section 3 marks it explicitly optional ("useful if
current preprocessing differs significantly"), and the four REQUIRED
conditions (H1-H4) are what's built and run. If you need H5, it would
be a third model trained on an edge_attr where `numeric_cols` use a
plain (non-host) z-score re-fit the same way `fit_host_stats`'s
`"__global__"` entry already computes — that dict is already returned
by `build_host_normalized_edge_attr`, so H5 mainly needs a small
"use only __global__, never a per-host entry" variant of
`host_normalized_matrix` plus the same train/test wiring H1-H4 already
have; it was left out rather than half-built, to keep this file's
tested surface matching what it actually runs.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.scadanet import (
    load_scadanet_df, build_graph_tensors, add_temporal_windows,
    temporal_split_positions, LABEL_COL, SRC_IP_COL, DST_IP_COL,
)
from src.evaluation.host_normalization import (
    HOST_NORM_NUMERIC_COLS, MIN_HOST_TRAIN_SAMPLES, KNOWN_PROBLEMATIC_HOSTS,
    build_host_normalized_edge_attr, false_positives_by_host, top_k_fp_share,
)
from src.evaluation.metrics import compute_metrics, per_attack_recall
from src.evaluation.splits import scadanet_track_b_split, carve_validation
from src.evaluation.temporal_protocol import scadanet_window_eval
from src.experiments.baseline_runner import _threshold_search
from src.experiments.temporal_runner import _train_frozen_scadanet_model
from src.models.baselines import predict_labels
from src.models.gnn import EdgeClassifierWithAttr, compute_subtype_balanced_weights
from src.utils.io import get_git_commit, get_library_versions
from src.utils.seed import set_seed

RESULTS_DIR = Path("experiments/results/q1/host_normalization")
CONFIG_PATH = RESULTS_DIR / "host_normalization_config.json"


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _train_trackb_style_model(
    x, edge_index, query_edges, edge_attr, y, fit_idx, per_flow_weight,
    seed: int, epochs: int, lr: float, hidden_dim: int = 16, log_every: int = 0,
) -> torch.nn.Module:
    """Same training loop as Issue #1's `run_scadanet_track_b`
    (per-sample subtype-balanced weighting, fixed full topology,
    EdgeClassifierWithAttr) — generalized to accept any edge_attr tensor
    so the ORIGINAL and HOST-NORMALIZED conditions differ only in which
    edge_attr is passed in, nothing else about training changes."""
    set_seed(seed)
    fit_y = y[fit_idx]
    model = EdgeClassifierWithAttr(in_dim=x.shape[1], edge_attr_dim=edge_attr.shape[1],
                                    hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index, query_edges[:, fit_idx], edge_attr[fit_idx])
        per_sample_loss = F.cross_entropy(out, fit_y, reduction="none")
        loss = (per_sample_loss * per_flow_weight).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            with torch.no_grad():
                acc = (out.argmax(dim=1) == fit_y).float().mean().item()
            print(f"[host_norm_runner][track_b] epoch {epoch + 1:4d}/{epochs} "
                  f"loss={loss.item():.4f} train_acc={acc:.4f}")
    model.eval()
    return model


def _predict_probs(model, x, edge_index, query_edges, edge_attr, idx) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, query_edges[:, idx], edge_attr[idx])
        probs = F.softmax(logits, dim=1)[:, 1]
    return probs


def _condition_result(
    name: str, leakage_condition_note: str, pred, true, probs, threshold: float,
    df: pd.DataFrame, idx, label_col: str, src_col: str, dst_col: str,
) -> Dict:
    m = compute_metrics(pred, true, probs)
    attack_labels = df.iloc[idx.numpy() if torch.is_tensor(idx) else np.asarray(idx)][label_col].to_numpy()
    par = per_attack_recall(pred, true, attack_labels)
    fp_by_src, fp_by_dst = false_positives_by_host(df, idx, src_col, dst_col, pred, true)
    src_share = top_k_fp_share(fp_by_src)
    tracked_fp = {h: int(fp_by_src.get(h, 0)) for h in KNOWN_PROBLEMATIC_HOSTS}
    return {
        "condition": name, "note": leakage_condition_note, "threshold": float(threshold),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "auprc": m["auprc"], "fpr": m["fpr"], "total_fp": int(fp_by_src.sum()),
        **src_share, "tracked_host_fp": tracked_fp, "per_attack_recall": par,
        "_fp_by_src": fp_by_src, "_fp_by_dst": fp_by_dst,
    }


def _write_csvs(rows: List[Dict], protocol: str, seed: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_") and k not in
         ("per_attack_recall", "tracked_host_fp")}
        for r in rows
    ])
    summary_csv = RESULTS_DIR / f"{protocol}_host_normalization_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[host_norm_runner] wrote {summary_csv}")

    fp_rows = []
    for r in rows:
        for host, count in r["_fp_by_src"].items():
            fp_rows.append({"protocol": protocol, "seed": seed, "condition": r["condition"],
                             "host": host, "role": "source", "fp_count": int(count)})
        for host, count in r["_fp_by_dst"].items():
            fp_rows.append({"protocol": protocol, "seed": seed, "condition": r["condition"],
                             "host": host, "role": "destination", "fp_count": int(count)})
    fp_csv = RESULTS_DIR / f"{protocol}_false_positives_by_host.csv"
    pd.DataFrame(fp_rows).to_csv(fp_csv, index=False)
    print(f"[host_norm_runner] wrote {fp_csv}")

    par_rows = []
    for r in rows:
        for attack, recall in r["per_attack_recall"].items():
            par_rows.append({"protocol": protocol, "seed": seed, "condition": r["condition"],
                              "attack_type": attack, "recall": recall})
    par_csv = RESULTS_DIR / f"{protocol}_per_attack_recall.csv"
    pd.DataFrame(par_rows).to_csv(par_csv, index=False)
    print(f"[host_norm_runner] wrote {par_csv}")


def _update_config(protocol: str, entry: Dict) -> None:
    """host_normalization_config.json is ONE shared file across both
    protocols (Issue #6 section 6 names it in the singular) — each
    runner call updates only its own `entry["protocol"]` key, preserving
    whatever the other protocol already wrote."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except Exception:
            config = {}
    config[protocol] = entry
    config["git_commit"] = get_git_commit()
    config["library_versions"] = get_library_versions()
    CONFIG_PATH.write_text(json.dumps(config, indent=2, default=str))
    print(f"[host_norm_runner] wrote {CONFIG_PATH}")


# ---------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------

def run_trackb_host_normalization(
    seed: int = 42, csv_path: str = None, epochs: int = 400, lr: float = 0.003,
    hidden_dim: int = 16, val_fraction: float = 0.15, log_every: int = 0,
) -> List[Dict]:
    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)

    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    subtype_weights = compute_subtype_balanced_weights(df, LABEL_COL, fit_idx)
    fit_labels = df.iloc[fit_idx.numpy()][LABEL_COL].values
    per_flow_weight = torch.tensor([subtype_weights[l] for l in fit_labels], dtype=torch.float32)

    # ---- H1/H2: original (globally normalized) features ----
    model_orig = _train_trackb_style_model(
        gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y, fit_idx,
        per_flow_weight, seed, epochs, lr, hidden_dim, log_every)
    val_probs_orig = _predict_probs(model_orig, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, val_idx)
    test_probs_orig = _predict_probs(model_orig, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, test_idx)
    t_orig, val_f1_orig = _threshold_search(val_probs_orig.numpy(), gt.y[val_idx].numpy())

    h1 = _condition_result(
        "H1_original_default_threshold", "reference",
        torch.as_tensor(predict_labels(test_probs_orig.numpy(), 0.5)), gt.y[test_idx],
        test_probs_orig, 0.5, df, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)
    h2 = _condition_result(
        "H2_original_calibrated_threshold", "existing mitigation (threshold only)",
        torch.as_tensor(predict_labels(test_probs_orig.numpy(), t_orig)), gt.y[test_idx],
        test_probs_orig, t_orig, df, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)

    # ---- H3/H4: host-normalized features ----
    # Fit host statistics on training rows excluding validation.
    host_edge_attr, host_stats, flag_rate_map, host_feature_names = build_host_normalized_edge_attr(
        df, gt.edge_attr, gt.feature_names, fit_idx, SRC_IP_COL)
    model_host = _train_trackb_style_model(
        gt.x, gt.edge_index, gt.query_edges, host_edge_attr, gt.y, fit_idx,
        per_flow_weight, seed, epochs, lr, hidden_dim, log_every)
    val_probs_host = _predict_probs(model_host, gt.x, gt.edge_index, gt.query_edges, host_edge_attr, val_idx)
    test_probs_host = _predict_probs(model_host, gt.x, gt.edge_index, gt.query_edges, host_edge_attr, test_idx)
    t_host, val_f1_host = _threshold_search(val_probs_host.numpy(), gt.y[val_idx].numpy())

    h3 = _condition_result(
        "H3_host_normalized_default_threshold", "feature-level fix alone",
        torch.as_tensor(predict_labels(test_probs_host.numpy(), 0.5)), gt.y[test_idx],
        test_probs_host, 0.5, df, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)
    h4 = _condition_result(
        "H4_host_normalized_calibrated_threshold", "combined mitigation",
        torch.as_tensor(predict_labels(test_probs_host.numpy(), t_host)), gt.y[test_idx],
        test_probs_host, t_host, df, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)

    rows = [h1, h2, h3, h4]
    for r in rows:
        r["protocol"] = "track_b"
        r["seed"] = seed
    _write_csvs(rows, "trackb", seed)
    _update_config("track_b", {
        "seed": seed, "method": "zscore",
        "numeric_cols_host_normalized": HOST_NORM_NUMERIC_COLS,
        "min_host_train_samples": MIN_HOST_TRAIN_SAMPLES,
        "unseen_or_small_host_fallback": "global training-period mean/std",
        "flag_rate_feature_appended": host_feature_names[len(gt.feature_names):] or None,
        "calibrated_threshold_original": float(t_orig),
        "calibrated_threshold_host_normalized": float(t_host),
        "val_f1_original": float(val_f1_orig), "val_f1_host_normalized": float(val_f1_host),
        "n_hosts_seen_in_train": len(host_stats) - 1,  # minus "__global__"
    })
    return rows


# ---------------------------------------------------------------------
# Strict SCADANet temporal protocol (Issue #5's mode="causal" / T2)
# ---------------------------------------------------------------------

def run_temporal_host_normalization(
    seed: int = 42, csv_path: str = None, n_windows: int = 20, epochs: int = 300,
    lr: float = 0.003, hidden_dim: int = 16, val_fraction: float = 0.15, log_every: int = 0,
) -> List[Dict]:
    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    df_t, snapshots = add_temporal_windows(df, gt.ip_index, n_windows=n_windows)
    orig_idx = torch.as_tensor(df_t["orig_idx"].to_numpy().copy(), dtype=torch.long)

    query_edges_t = gt.query_edges[:, orig_idx]
    edge_attr_t = gt.edge_attr[orig_idx]
    y_t = gt.y[orig_idx]
    train_pos, test_pos, split_window = temporal_split_positions(df_t)
    train_idx = torch.as_tensor(train_pos, dtype=torch.long)
    test_idx = torch.as_tensor(test_pos, dtype=torch.long)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    train_topology = snapshots[split_window - 1]["edge_index"]
    x = torch.ones((len(gt.all_ips), 4), dtype=torch.float32)

    # ---- H1/H2: original features, strict causal (T2) test protocol ----
    model_orig = _train_frozen_scadanet_model(
        x, train_topology, query_edges_t, edge_attr_t, y_t, fit_idx,
        epochs=epochs, lr=lr, log_every=log_every)
    val_probs_orig = _predict_probs(model_orig, x, train_topology, query_edges_t, edge_attr_t, val_idx)
    t_orig, val_f1_orig = _threshold_search(val_probs_orig.numpy(), y_t[val_idx].numpy())
    _, test_pred_default_orig, test_true, test_probs_orig = scadanet_window_eval(
        model_orig, x, query_edges_t, edge_attr_t, y_t, df_t, snapshots,
        split_window, n_windows, LABEL_COL, mode="causal")

    h1 = _condition_result(
        "H1_original_default_threshold", "reference",
        test_pred_default_orig, test_true, test_probs_orig, 0.5,
        df_t, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)
    h2 = _condition_result(
        "H2_original_calibrated_threshold", "existing mitigation (threshold only)",
        torch.as_tensor(predict_labels(test_probs_orig.numpy(), t_orig)), test_true,
        test_probs_orig, t_orig, df_t, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)

    # ---- H3/H4: host-normalized features, same strict causal protocol ----
    # Fit host statistics on pre-boundary training rows, excluding validation.
    host_edge_attr_t, host_stats, flag_rate_map, host_feature_names = build_host_normalized_edge_attr(
        df_t, edge_attr_t, gt.feature_names, fit_idx, SRC_IP_COL)
    model_host = _train_frozen_scadanet_model(
        x, train_topology, query_edges_t, host_edge_attr_t, y_t, fit_idx,
        epochs=epochs, lr=lr, log_every=log_every)
    val_probs_host = _predict_probs(model_host, x, train_topology, query_edges_t, host_edge_attr_t, val_idx)
    t_host, val_f1_host = _threshold_search(val_probs_host.numpy(), y_t[val_idx].numpy())
    _, test_pred_default_host, test_true_host, test_probs_host = scadanet_window_eval(
        model_host, x, query_edges_t, host_edge_attr_t, y_t, df_t, snapshots,
        split_window, n_windows, LABEL_COL, mode="causal")

    h3 = _condition_result(
        "H3_host_normalized_default_threshold", "feature-level fix alone",
        test_pred_default_host, test_true_host, test_probs_host, 0.5,
        df_t, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)
    h4 = _condition_result(
        "H4_host_normalized_calibrated_threshold", "combined mitigation",
        torch.as_tensor(predict_labels(test_probs_host.numpy(), t_host)), test_true_host,
        test_probs_host, t_host, df_t, test_idx, LABEL_COL, SRC_IP_COL, DST_IP_COL)

    rows = [h1, h2, h3, h4]
    for r in rows:
        r["protocol"] = "temporal_causal"
        r["seed"] = seed
    _write_csvs(rows, "temporal", seed)
    _update_config("temporal_causal", {
        "seed": seed, "method": "zscore", "test_mode": "causal (Issue #5 T2)",
        "numeric_cols_host_normalized": HOST_NORM_NUMERIC_COLS,
        "min_host_train_samples": MIN_HOST_TRAIN_SAMPLES,
        "unseen_or_small_host_fallback": "global training-period mean/std",
        "flag_rate_feature_appended": host_feature_names[len(gt.feature_names):] or None,
        "calibrated_threshold_original": float(t_orig),
        "calibrated_threshold_host_normalized": float(t_host),
        "val_f1_original": float(val_f1_orig), "val_f1_host_normalized": float(val_f1_host),
        "n_hosts_seen_in_train": len(host_stats) - 1,
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=["track_b", "temporal", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()

    if args.protocol in ("track_b", "both"):
        run_trackb_host_normalization(
            seed=args.seed, csv_path=args.csv_path,
            epochs=args.epochs or 400, log_every=args.log_every)
    if args.protocol in ("temporal", "both"):
        run_temporal_host_normalization(
            seed=args.seed, csv_path=args.csv_path,
            epochs=args.epochs or 300, log_every=args.log_every)


if __name__ == "__main__":
    main()
