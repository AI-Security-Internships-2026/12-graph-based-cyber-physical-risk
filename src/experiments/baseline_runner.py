"""
Issue #3 — fair non-GNN and GNN baselines under identical splits.

Runs Experiments B1 (SCADANet Track B, all 6 model families), B2
(strict SCADANet temporal, principal models), B3 (BATADAL temporal,
principal models). Reuses Issue #1's data/split code unchanged
(src/data/*, src/evaluation/splits.py's existing functions) and adds
only what Issue #3 needs on top: a validation carve-out
(splits.carve_validation), a small validation-only hyperparameter/
threshold search, the non-GNN baselines (src/models/baselines.py), and
the GCN/GAT edge+window classifiers (src/models/gnn.py).

Every *_family() function below trains ONE model family on already-
built tensors/features and returns one result dict with the schema
src.utils.io.REQUIRED_FIELDS plus Issue #3's extra fields (auroc,
per_attack_recall, n_trainable_params, inference_latency_ms,
best_hyperparams). experiments/results/q1/baselines/*.csv is one row
per (protocol, model, seed) — the "save per-seed results rather than
only the final mean" requirement (item 7).
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.batadal import load_batadal_df, build_windowed_graphs
from src.data.scadanet import (
    load_scadanet_df, build_graph_tensors, add_temporal_windows,
    temporal_split_positions, LABEL_COL,
)
from src.evaluation.metrics import (
    compute_metrics, per_attack_recall, count_trainable_params,
    measure_inference_latency,
)
from src.evaluation.splits import (
    scadanet_track_b_split, carve_validation, batadal_static_cv_folds,
)
from src.models.baselines import (
    LogisticRegressionBaseline, MLPBaseline, TreeBaseline,
    scadanet_content_features, flatten_window_graph_features,
    predict_labels, HParamGrid,
)
from src.models.gnn import (
    GNNEdgeClassifierWithAttr, GNNWindowClassifier, EndpointOnlyEdgeClassifier,
    compute_subtype_balanced_weights, build_class_weights,
)
from src.utils.io import Timer, write_result
from src.utils.seed import set_seed

GNN_CONV_TYPES = ["graphsage", "gcn", "gat"]
NON_GNN_FAMILIES = ["logistic_regression", "mlp", "tree"]
ALL_FAMILIES = NON_GNN_FAMILIES + GNN_CONV_TYPES

DEFAULT_GRID = HParamGrid()


def _resolve_device(device: Optional[str] = None) -> torch.device:
    """Auto-detects CUDA (e.g. Colab's T4) unless overridden. Neither this
    file nor src/experiments/runner.py (Issue #1) moved anything off CPU
    before this fix — confirmed by grepping both files for 'cuda'/'.to('
    and finding nothing, and by Issue #1's own notebook run (single
    config, 400 epochs) taking 1128.6s despite a GPU runtime being
    attached. B1's default grid trains 24 full GNN configs (3 conv types
    x 8 hyperparameter combos), so that per-config cost multiplies into
    hours; moving tensors/model to GPU when available is the single
    biggest lever here."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Threshold selection uses validation data only.
# ---------------------------------------------------------------------

def _threshold_search(val_probs: np.ndarray, val_y: np.ndarray,
                       grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Return (best_threshold, best_val_f1). Falls back to 0.5 when
    validation has only one class present (F1 undefined either way)."""
    if len(np.unique(val_y)) < 2:
        return 0.5, 0.0
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (val_probs >= t).astype(int)
        tp = int(((pred == 1) & (val_y == 1)).sum())
        fp = int(((pred == 1) & (val_y == 0)).sum())
        fn = int(((pred == 0) & (val_y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------
# GNN edge-classifier family (GraphSAGE / GCN / GAT) — SCADANet
# ---------------------------------------------------------------------

def _train_gnn_edge(conv_type, x, edge_index, query_edges, edge_attr, y,
                     fit_idx, sample_weight_fit, hidden_dim, lr, dropout,
                     epochs, seed, device, val_idx=None, log_every=50, log_prefix=""):
    set_seed(seed)
    if conv_type == "endpoint_only":
    # Endpoint-only ablation uses its own classifier without graph convolution.
        model = EndpointOnlyEdgeClassifier(
            num_nodes=x.shape[0], edge_attr_dim=edge_attr.shape[1],
            hidden_dim=hidden_dim, edge_proj_dim=hidden_dim, dropout=dropout,
        ).to(device)
    else:
        model = GNNEdgeClassifierWithAttr(
            conv_type, in_dim=x.shape[1], edge_attr_dim=edge_attr.shape[1],
            hidden_dim=hidden_dim, edge_proj_dim=hidden_dim, dropout=dropout,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    fit_y = y[fit_idx].to(device)
    w = torch.as_tensor(sample_weight_fit, dtype=torch.float32, device=device)
    fit_query_edges = query_edges[:, fit_idx].to(device)
    fit_edge_attr = edge_attr[fit_idx].to(device)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(x, edge_index, fit_query_edges, fit_edge_attr)
        per_sample_loss = F.cross_entropy(out, fit_y, reduction="none")
        loss = (per_sample_loss * w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            with torch.no_grad():
                train_pred = out.argmax(dim=1)
                train_metrics = compute_metrics(train_pred.cpu(), fit_y.cpu())
            msg = (f"{log_prefix}epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  "
                   f"TRAIN acc={train_metrics['accuracy']:.4f} f1={train_metrics['f1']:.4f}")
            if val_idx is not None:
                val_probs = _gnn_edge_probs(model, x, edge_index, query_edges, edge_attr, val_idx, device)
                val_pred = predict_labels(val_probs, 0.5)
                val_metrics = compute_metrics(torch.as_tensor(val_pred), y[val_idx])
                msg += f"  |  VAL acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}"
                model.train()
            print(msg)
    return model


def _gnn_edge_probs(model, x, edge_index, query_edges, edge_attr, idx, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, query_edges[:, idx].to(device), edge_attr[idx].to(device))
        probs = F.softmax(logits, dim=1)[:, 1]
    return probs.cpu().numpy()


def gnn_edge_family(
    conv_type: str, x, edge_index, query_edges, edge_attr, y,
    df: pd.DataFrame, fit_idx: torch.Tensor, val_idx: torch.Tensor,
    test_idx: torch.Tensor, seed: int, epochs: int = 300,
    grid: HParamGrid = DEFAULT_GRID, subtype_balanced: bool = True,
    device: Optional[torch.device] = None, log_every: int = 50,
    return_model: bool = False, strict_temporal_test: Optional[Dict] = None,
):
    """One GraphSAGE/GCN/GAT edge classifier, small validation-only
    search over (hidden_dim, lr, dropout), validation-only threshold.
    `subtype_balanced=True` matches Track B's per-subtype weighting
    (Issue #1's compute_subtype_balanced_weights); False uses plain
    binary class weights (temporal protocol). `log_every`: print
    train/val progress every N epochs (matches Issue #1's runner.py
    logging style); 0 disables per-epoch logging entirely.

    `return_model`: if True, returns (result_dict, best_model, device,
    x_dev, edge_index_dev) instead of just result_dict — used by Issue
    #4's feature-permutation counterfactuals (Table C2), which need to
    re-run inference with perturbed edge_attr on an already-trained
    Condition C model rather than retraining. Default False preserves
    Issue #3's original single-dict return for every existing caller.

    `strict_temporal_test`: only for a chronologically-ordered `df`
    (i.e. `df_t` from `add_temporal_windows`) — pass
    {"snapshots": ..., "split_window": ..., "n_windows": ...} to score
    the FINAL test set via Issue #5's `scadanet_window_eval(mode=
    "causal")` (walking test windows with topology from before each
    window) instead of this function's normal single-fixed-topology
    test eval. Training and validation-threshold search still use
    whatever `edge_index` the caller passed in for every hyperparameter
    combo — for a strict-temporal caller that must be the TRAIN-BOUNDARY
    topology (`snapshots[split_window - 1]["edge_index"]`), never the
    full/final topology, since validation flows are all pre-boundary and
    scoring them with anything else would score training-only data
    against of a future-leaking topology. Added on a later
    full-requirements re-audit: `run_experiment_b2` was originally
    always scoring test with the FINAL streamed topology regardless of
    window — the exact leakage Issue #5 was built to fix, silently
    un-fixed here despite this function's own "temporal_strict" label.
    See README_ISSUE3.md's "Fix applied on review"."""
    device = device or _resolve_device()
    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)

    if subtype_balanced:
        subtype_weights = compute_subtype_balanced_weights(df, LABEL_COL, fit_idx)
        fit_labels = df.iloc[fit_idx.numpy()][LABEL_COL].values
        sample_weight_fit = np.array([subtype_weights[l] for l in fit_labels], dtype=np.float32)
    else:
        fit_y = y[fit_idx]
        cw = build_class_weights(fit_y)
        sample_weight_fit = cw[fit_y].numpy()

    combos = list(itertools.product(grid.hidden_dim, grid.lr, grid.dropout))
    best = None
    searched = []
    with Timer() as train_timer:
        for combo_i, (hidden_dim, lr, dropout) in enumerate(combos, start=1):
            prefix = f"[{conv_type}] combo {combo_i}/{len(combos)} (hidden={hidden_dim}, lr={lr}, dropout={dropout})  "
            if log_every:
                print(f"{prefix}starting...")
            model = _train_gnn_edge(conv_type, x_dev, edge_index_dev, query_edges, edge_attr, y,
                                     fit_idx, sample_weight_fit, hidden_dim, lr, dropout,
                                     epochs, seed, device, val_idx=val_idx if log_every else None,
                                     log_every=log_every, log_prefix=prefix)
            val_probs = _gnn_edge_probs(model, x_dev, edge_index_dev, query_edges, edge_attr, val_idx, device)
            val_y = y[val_idx].numpy()
            t, val_f1 = _threshold_search(val_probs, val_y)
            searched.append({"hidden_dim": hidden_dim, "lr": lr, "dropout": dropout,
                              "threshold": t, "val_f1": val_f1})
            if log_every:
                print(f"{prefix}done. best threshold={t:.2f}, val_f1={val_f1:.4f}")
            if best is None or val_f1 > best["val_f1"]:
                best = {"model": model, "hidden_dim": hidden_dim, "lr": lr,
                        "dropout": dropout, "threshold": t, "val_f1": val_f1}

    if strict_temporal_test is not None:
        from src.evaluation.temporal_protocol import scadanet_window_eval
        # scadanet_window_eval does no device placement of its own (it
        # assumes everything passed in already matches `model`'s device,
        # which is true for temporal_runner.py's own CPU-only callers but
        # NOT here, since this function explicitly supports GPU via
        # _resolve_device()) — move query_edges/edge_attr/y and every
        # snapshot's edge_index to `device` before calling it, or this
        # silently crashes with a device-mismatch the moment `device` is
        # CUDA. Only "edge_index" is reconstructed per snapshot because
        # that's the only key scadanet_window_eval actually reads.
        snapshots_dev = [{"edge_index": snap["edge_index"].to(device)}
                          for snap in strict_temporal_test["snapshots"]]
        query_edges_dev = query_edges.to(device)
        edge_attr_dev = edge_attr.to(device)
        y_dev = y.to(device)
        _, _, test_true_t, test_probs_t = scadanet_window_eval(
            best["model"], x_dev, query_edges_dev, edge_attr_dev, y_dev, df,
            snapshots_dev, strict_temporal_test["split_window"],
            strict_temporal_test["n_windows"], LABEL_COL, mode="causal")
        test_probs = test_probs_t.cpu().numpy()
        test_pred = predict_labels(test_probs, best["threshold"])
        y_for_metrics = test_true_t.cpu()
        attack_labels = df.iloc[test_idx.numpy()][LABEL_COL].values  # same row order — see temporal_runner.py's own Table T2 alignment note
        eval_note = "strict_causal_temporal (Issue #5 T2, via scadanet_window_eval)"
    else:
        test_probs = _gnn_edge_probs(best["model"], x_dev, edge_index_dev, query_edges, edge_attr, test_idx, device)
        test_pred = predict_labels(test_probs, best["threshold"])
        y_for_metrics = y[test_idx]
        attack_labels = df.iloc[test_idx.numpy()][LABEL_COL].values
        eval_note = "fixed_topology"

    metrics = compute_metrics(
        torch.as_tensor(test_pred), y_for_metrics, torch.as_tensor(test_probs)
    )
    par = per_attack_recall(torch.as_tensor(test_pred), y_for_metrics, attack_labels)

    latency = measure_inference_latency(
        lambda: _gnn_edge_probs(best["model"], x_dev, edge_index_dev, query_edges, edge_attr, test_idx, device),
        n_repeats=10, warmup=2,
    )

    result = {
        "dataset": "scadanet", "model": f"{conv_type}_edge_attr", "model_family": conv_type,
        "seed": seed, "train_size": int(len(fit_idx)), "validation_size": int(len(val_idx)),
        "test_size": int(len(test_idx)), "feature_set": "enriched_edge_attr",
        "graph_mode": "static", "threshold": best["threshold"], "runtime": train_timer.elapsed,
        "n_trainable_params": count_trainable_params(best["model"]),
        "inference_latency_ms": latency,
        "per_attack_recall": par,
        "best_hyperparams": {"hidden_dim": best["hidden_dim"], "lr": best["lr"],
                              "dropout": best["dropout"]},
        "hyperparam_search_log": searched,
        "notes": {"loss_weighting": "per_sample_subtype_balanced" if subtype_balanced
                  else "binary_class_weighted",
                  "node_feature_dim": int(x.shape[1]), "edge_attr_dim": int(edge_attr.shape[1]),
                  "epochs_per_combo": epochs, "test_eval_mode": eval_note},
        **metrics,
    }
    if return_model:
        return result, best["model"], device, x_dev, edge_index_dev
    return result


# ---------------------------------------------------------------------
# Non-GNN families (Logistic Regression / MLP / tree) — SCADANet content
# features, per Issue #3 section 3's fairness rule
# ---------------------------------------------------------------------

def _fit_predict_nongnn(family: str, X_fit, y_fit, X_val, y_val, X_test,
                         sample_weight_fit, seed, grid: HParamGrid, log_every: int = 0,
                         mlp_epochs: int = 200):
    searched = []
    best = None

    if family == "logistic_regression":
        combos = [{"C": c} for c in grid.lr_C]
    elif family == "mlp":
        combos = [{"hidden_dim": h, "lr": lr, "dropout": d}
                   for h, lr, d in itertools.product(grid.hidden_dim, grid.lr, grid.dropout)]
    elif family == "tree":
        combos = [{"max_depth": d, "n_estimators": n}
                   for d, n in itertools.product(grid.tree_max_depth, grid.tree_n_estimators)]
    else:
        raise ValueError(family)

    for combo_i, combo in enumerate(combos, start=1):
        prefix = f"[{family}] combo {combo_i}/{len(combos)} ({combo})  "
        if log_every:
            print(f"{prefix}starting...")

        if family == "logistic_regression":
            model = LogisticRegressionBaseline(C=combo["C"], seed=seed)
            model.fit(X_fit, y_fit, sample_weight=sample_weight_fit)
        elif family == "mlp":
            model = MLPBaseline(in_dim=X_fit.shape[1], hidden_dim=combo["hidden_dim"],
                                 dropout=combo["dropout"], lr=combo["lr"], seed=seed,
                                 epochs=mlp_epochs)
            model.fit(X_fit, y_fit, sample_weight=sample_weight_fit,
                      log_every=log_every, log_prefix=prefix)
        else:
            model = TreeBaseline(max_depth=combo["max_depth"],
                                  n_estimators=combo["n_estimators"], seed=seed)
            model.fit(X_fit, y_fit, sample_weight=sample_weight_fit)
        val_probs = model.predict_proba(X_val)
        t, val_f1 = _threshold_search(val_probs, y_val)
        searched.append({**combo, "threshold": t, "val_f1": val_f1})
        if log_every:
            print(f"{prefix}done. best threshold={t:.2f}, val_f1={val_f1:.4f}")
        if best is None or val_f1 > best["val_f1"]:
            best = {"model": model, "combo": combo, "threshold": t, "val_f1": val_f1}

    return best, searched


def nongnn_family(
    family: str, edge_attr: torch.Tensor, y: torch.Tensor, df: pd.DataFrame,
    fit_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor,
    seed: int, grid: HParamGrid = DEFAULT_GRID, subtype_balanced: bool = True,
    log_every: int = 0, mlp_epochs: int = 200, return_model: bool = False,
) -> Dict:
    """Logistic Regression / MLP / tree, all consuming
    scadanet_content_features(edge_attr, idx) — the exact same tensor the
    GraphSAGE edge_proj consumes (Issue #3 section 3 fairness rule).

    `return_model`: if True, returns (result_dict, best_model, X_test,
    test_probs) instead of just result_dict — same additive pattern as
    `gnn_edge_family`'s own `return_model` (added for Issue #4's
    counterfactuals, which need the fitted model and raw per-sample
    probabilities, not just the aggregated metrics every other caller
    only needed). Default False preserves every existing caller's
    original return."""
    X_fit = scadanet_content_features(edge_attr, fit_idx)
    X_val = scadanet_content_features(edge_attr, val_idx)
    X_test = scadanet_content_features(edge_attr, test_idx)
    y_fit = y[fit_idx].numpy()
    y_val = y[val_idx].numpy()
    y_test = y[test_idx]

    if subtype_balanced:
        subtype_weights = compute_subtype_balanced_weights(df, LABEL_COL, fit_idx)
        fit_labels = df.iloc[fit_idx.numpy()][LABEL_COL].values
        sample_weight_fit = np.array([subtype_weights[l] for l in fit_labels], dtype=np.float32)
    else:
        cw = build_class_weights(y[fit_idx])
        sample_weight_fit = cw[y[fit_idx]].numpy()

    with Timer() as train_timer:
        best, searched = _fit_predict_nongnn(family, X_fit, y_fit, X_val, y_val, X_test,
                                              sample_weight_fit, seed, grid, log_every=log_every,
                                              mlp_epochs=mlp_epochs)

    test_probs = best["model"].predict_proba(X_test)
    test_pred = predict_labels(test_probs, best["threshold"])
    metrics = compute_metrics(torch.as_tensor(test_pred), y_test, torch.as_tensor(test_probs))

    attack_labels = df.iloc[test_idx.numpy()][LABEL_COL].values
    par = per_attack_recall(torch.as_tensor(test_pred), y_test, attack_labels)

    latency = measure_inference_latency(
        lambda: best["model"].predict_proba(X_test), n_repeats=10, warmup=2,
    )

    model_name = {"logistic_regression": "logistic_regression", "mlp": "mlp",
                  "tree": f"tree_{best['model'].backend if family == 'tree' else ''}"}[family]

    return {
        "dataset": "scadanet", "model": model_name, "model_family": family,
        "seed": seed, "train_size": int(len(fit_idx)), "validation_size": int(len(val_idx)),
        "test_size": int(len(test_idx)), "feature_set": "enriched_edge_attr_content_only",
        "graph_mode": "n/a_non_graph", "threshold": best["threshold"], "runtime": train_timer.elapsed,
        "n_trainable_params": best["model"].n_trainable_params(),
        "inference_latency_ms": latency,
        "per_attack_recall": par,
        "best_hyperparams": best["combo"],
        "hyperparam_search_log": searched,
        "notes": {"loss_weighting": "per_sample_subtype_balanced" if subtype_balanced
                  else "binary_class_weighted",
                  "reason_no_topology": "non-GNN model; receives same per-flow content "
                                         "features as the GraphSAGE edge classifier "
                                         "(Issue #3 section 3), no graph structure.",
                  "input_dim": int(X_fit.shape[1]),
                  "mlp_epochs": mlp_epochs if family == "mlp" else None},
        **metrics,
    }
    if return_model:
        return result, best["model"], X_test, test_probs
    return result


# ---------------------------------------------------------------------
# Experiment B1 — SCADANet Track B, all 6 model families
# ---------------------------------------------------------------------

def run_experiment_b1(seed: int = 42, csv_path: str = None, epochs: int = 300,
                       val_fraction: float = 0.15, grid: HParamGrid = DEFAULT_GRID,
                       families: Optional[List[str]] = None,
                       device: Optional[torch.device] = None,
                       on_result=None, log_every: int = 50) -> List[Dict]:
    """`families`: subset of ALL_FAMILIES to run (default: all six) — lets
    you resume a partial run instead of redoing everything. `on_result`:
    optional callback(result_dict) invoked immediately after each family
    finishes, so results are checkpointed to disk as they complete rather
    than only after the whole function returns (see main()'s use of this
    — a multi-hour run that gets interrupted now keeps whatever finished
    before the interruption). `log_every`: print training progress every
    N epochs, matching Issue #1's runner.py logging style; 0 disables it."""
    set_seed(seed)
    device = device or _resolve_device()
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)

    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    families = families or ALL_FAMILIES
    results = []
    for family in NON_GNN_FAMILIES:
        if family not in families:
            continue
        r = nongnn_family(family, gt.edge_attr, gt.y, df, fit_idx, val_idx, test_idx,
                           seed, grid=grid, subtype_balanced=True, log_every=log_every,
                           mlp_epochs=epochs)
        r["split_protocol"] = "track_b"
        r["notes"]["split_meta"] = split_meta
        results.append(r)
        if on_result:
            on_result(r)

    for conv_type in GNN_CONV_TYPES:
        if conv_type not in families:
            continue
        r = gnn_edge_family(conv_type, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr,
                             gt.y, df, fit_idx, val_idx, test_idx, seed, epochs=epochs,
                             grid=grid, subtype_balanced=True, device=device, log_every=log_every)
        r["split_protocol"] = "track_b"
        r["notes"]["split_meta"] = split_meta
        results.append(r)
        if on_result:
            on_result(r)

    return results


# ---------------------------------------------------------------------
# Experiment B2 — strict SCADANet temporal, principal models only
# (tree, MLP, GraphSAGE, best alternative GNN from B1)
# ---------------------------------------------------------------------

def run_experiment_b2(seed: int = 42, csv_path: str = None, n_windows: int = 20,
                       epochs: int = 300, val_fraction: float = 0.15,
                       best_alt_gnn: str = "gcn", grid: HParamGrid = DEFAULT_GRID,
                       families: Optional[List[str]] = None,
                       device: Optional[torch.device] = None,
                       on_result=None, log_every: int = 50) -> List[Dict]:
    """`best_alt_gnn` ("gcn" or "gat") should be picked from B1's result —
    pass whichever of the two scored higher there; defaults to "gcn" only
    as a placeholder when B1 hasn't been run yet in the calling session.
    `families`/`on_result`: see run_experiment_b1 docstring.

    Trains every GNN on `train_topology` (the topology observable at the
    train/test boundary — snapshots[split_window - 1]) rather than the
    fully-streamed final topology, and scores the final test set via
    Issue #5's own `scadanet_window_eval(mode="causal")`, walking test
    windows with only the topology observable before each one. Originally
    (before a later full-requirements re-audit) this trained AND tested
    every GNN against `snapshots[-1]["edge_index"]` — the full streaming
    topology regardless of window — which is exactly the future-topology
    leak Issue #5 was built to fix, silently un-fixed here despite this
    experiment's own "strict" name. MLP/tree don't consume topology at
    all, so they were never affected by this bug — only the GNN rows
    needed the fix. See README_ISSUE3.md's "Fix applied on review"."""
    from src.evaluation.temporal_protocol import scadanet_window_eval  # noqa: F401
    set_seed(seed)
    device = device or _resolve_device()
    all_b2_families = ["mlp", "tree", "graphsage", best_alt_gnn]
    families = families or all_b2_families
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    df_t, snapshots = add_temporal_windows(df, gt.ip_index, n_windows=n_windows)
    train_idx_np, test_idx_np, split_window = temporal_split_positions(df_t)

    # Rebuild tensors in the temporal row order (df_t carries orig_idx).
    orig_idx = torch.as_tensor(df_t["orig_idx"].to_numpy().copy(), dtype=torch.long)
    query_edges_t = gt.query_edges[:, orig_idx]
    edge_attr_t = gt.edge_attr[orig_idx]
    y_t = gt.y[orig_idx]
    # Topology observable at the train/test boundary — used for training
    # AND validation-threshold search (validation flows are all
    # pre-boundary, so this is the causally-correct topology for them
    # too, not a leak). NOT used for the final test score — see
    # strict_temporal_test below, which walks per-window topology instead.
    train_topology = snapshots[split_window - 1]["edge_index"]

    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)
    test_idx = torch.as_tensor(test_idx_np, dtype=torch.long)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    results = []
    for family in ["mlp", "tree"]:
        if family not in families:
            continue
        r = nongnn_family(family, edge_attr_t, y_t, df_t, fit_idx, val_idx, test_idx,
                           seed, grid=grid, subtype_balanced=False, log_every=log_every,
                           mlp_epochs=epochs)
        r["split_protocol"] = "temporal_strict"
        r["notes"]["n_windows"] = n_windows
        r["notes"]["split_window"] = split_window
        results.append(r)
        if on_result:
            on_result(r)

    strict_temporal_test = {"snapshots": snapshots, "split_window": split_window,
                             "n_windows": n_windows}
    for conv_type in ["graphsage", best_alt_gnn]:
        if conv_type not in families:
            continue
        r = gnn_edge_family(conv_type, gt.x, train_topology, query_edges_t, edge_attr_t,
                             y_t, df_t, fit_idx, val_idx, test_idx, seed, epochs=epochs,
                             grid=grid, subtype_balanced=False, device=device, log_every=log_every,
                             strict_temporal_test=strict_temporal_test)
        r["split_protocol"] = "temporal_strict"
        r["notes"]["n_windows"] = n_windows
        r["notes"]["split_window"] = split_window
        results.append(r)
        if on_result:
            on_result(r)

    return results


# ---------------------------------------------------------------------
# Experiment B3 — BATADAL strict temporal, principal models only
# (tabular MLP baseline, GraphSAGE, GCN or GAT)
# ---------------------------------------------------------------------

def run_experiment_b3(seed: int = 42, csv_path: str = "BATADAL_dataset04.csv",
                       epochs: int = 100, val_fraction: float = 0.15,
                       alt_gnn: str = "gcn", grid: HParamGrid = DEFAULT_GRID,
                       families: Optional[List[str]] = None,
                       device: Optional[torch.device] = None,
                       on_result=None, log_every: int = 20,
                       split_frac: float = 0.7) -> List[Dict]:
    """B3 is named "BATADAL strict temporal evaluation" in Issue #3's own
    text — this now actually uses Issue #5's purged split
    (`batadal_purged_split`, Part B) rather than the pre-Issue-#5
    `temporal_split_windows`, whose overlapping 24h/6h-stride windows can
    share up to 18h of raw rows across the train/test boundary. This
    matters for EVERY family here, not just the GNNs: the tabular MLP
    baseline flattens features straight from these window objects too,
    so the old leaky split affected it identically. Fixed during a
    full-requirements re-audit against Issue #3's text, not part of the
    original build — see README_ISSUE3.md's "Fix applied on review"."""
    from src.evaluation.temporal_protocol import batadal_purged_split
    device = device or _resolve_device()
    families = families or ["mlp", "graphsage", alt_gnn]
    df = load_batadal_df(csv_path)
    window_graphs, window_labels, extra = build_windowed_graphs(df)
    starts, window_size = extra["starts"], extra["window_size"]
    train_window_idx, test_window_idx, purged_window_idx, purge_meta = batadal_purged_split(
        len(df), starts, window_size, split_frac)
    train_graphs = [window_graphs[i] for i in train_window_idx]
    test_graphs = [window_graphs[i] for i in test_window_idx]

    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n_train = len(train_graphs)
    perm = torch.randperm(n_train, generator=g).tolist()
    n_val = max(1, int(n_train * val_fraction))
    val_graphs = [train_graphs[i] for i in perm[:n_val]]
    fit_graphs = [train_graphs[i] for i in perm[n_val:]]

    from torch_geometric.loader import DataLoader

    results = []

    # --- tabular MLP baseline: flattened window sensor features ---
    if "mlp" in families:
        X_fit = flatten_window_graph_features(fit_graphs)
        X_val = flatten_window_graph_features(val_graphs)
        X_test = flatten_window_graph_features(test_graphs)
        y_fit = torch.cat([g_.y for g_ in fit_graphs]).numpy()
        y_val = torch.cat([g_.y for g_ in val_graphs]).numpy()
        y_test = torch.cat([g_.y for g_ in test_graphs])
        cw = build_class_weights(torch.as_tensor(y_fit))
        sample_weight_fit = cw[torch.as_tensor(y_fit)].numpy()

        with Timer() as train_timer:
            best, searched = _fit_predict_nongnn("mlp", X_fit, y_fit, X_val, y_val, X_test,
                                                  sample_weight_fit, seed, grid, log_every=log_every,
                                                  mlp_epochs=epochs)
        test_probs = best["model"].predict_proba(X_test)
        test_pred = predict_labels(test_probs, best["threshold"])
        metrics = compute_metrics(torch.as_tensor(test_pred), y_test, torch.as_tensor(test_probs))
        latency = measure_inference_latency(lambda: best["model"].predict_proba(X_test),
                                             n_repeats=10, warmup=2)
        r = {
            "dataset": "batadal", "model": "mlp_flattened_window", "model_family": "mlp",
            "split_protocol": "temporal_purged", "seed": seed,
            "train_size": len(fit_graphs), "validation_size": len(val_graphs),
            "test_size": len(test_graphs), "feature_set": "flattened_window_sensor_features",
            "graph_mode": "n/a_non_graph", "threshold": best["threshold"], "runtime": train_timer.elapsed,
            "n_trainable_params": best["model"].n_trainable_params(),
            "inference_latency_ms": latency,
            "per_attack_recall": {"n/a": None},
            "best_hyperparams": best["combo"], "hyperparam_search_log": searched,
            "notes": {"window_size": extra["window_size"], "stride": extra["stride"],
                      "feature_flattening": "flatten_window_graph_features "
                                             "(see src/models/baselines.py docstring)",
                      "purge_meta": purge_meta},
            **metrics,
        }
        results.append(r)
        if on_result:
            on_result(r)

    # --- GraphSAGE + one alternative GNN, window/graph-level ---
    for conv_type in ["graphsage", alt_gnn]:
        if conv_type not in families:
            continue
        best_gnn = None
        searched_gnn = []
        combos = list(itertools.product(grid.hidden_dim, grid.lr, grid.dropout))
        with Timer() as t:
            for combo_i, (hidden_dim, lr, dropout) in enumerate(combos, start=1):
                prefix = f"[{conv_type}] combo {combo_i}/{len(combos)} (hidden={hidden_dim}, lr={lr}, dropout={dropout})  "
                if log_every:
                    print(f"{prefix}starting...")
                set_seed(seed)
                model = GNNWindowClassifier(conv_type, in_dim=fit_graphs[0].x.shape[1],
                                             hidden_dim=hidden_dim, dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
                fit_loader = DataLoader(fit_graphs, batch_size=16, shuffle=True)
                fit_y_all = torch.cat([g_.y for g_ in fit_graphs])
                cw_gnn = build_class_weights(fit_y_all).to(device)

                model.train()
                for epoch in range(1, epochs + 1):
                    epoch_losses = []
                    for batch in fit_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        out = model(batch.x, batch.edge_index, batch.batch)
                        loss = F.cross_entropy(out, batch.y, weight=cw_gnn)
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())
                    if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                        print(f"{prefix}epoch {epoch:4d}/{epochs}  "
                              f"loss={sum(epoch_losses) / len(epoch_losses):.4f}")

                model.eval()
                val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
                val_probs_list, val_y_list = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        val_probs_list.append(F.softmax(out, dim=1)[:, 1].cpu())
                        val_y_list.append(batch.y.cpu())
                val_probs = torch.cat(val_probs_list).numpy()
                val_y_np = torch.cat(val_y_list).numpy()
                thr, val_f1 = _threshold_search(val_probs, val_y_np)
                searched_gnn.append({"hidden_dim": hidden_dim, "lr": lr, "dropout": dropout,
                                      "threshold": thr, "val_f1": val_f1})
                if log_every:
                    print(f"{prefix}done. best threshold={thr:.2f}, val_f1={val_f1:.4f}")
                if best_gnn is None or val_f1 > best_gnn["val_f1"]:
                    best_gnn = {"model": model, "hidden_dim": hidden_dim, "lr": lr,
                                "dropout": dropout, "threshold": thr, "val_f1": val_f1}

        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        best_gnn["model"].eval()

        def _predict_test():
            probs, ys = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = best_gnn["model"](batch.x, batch.edge_index, batch.batch)
                    probs.append(F.softmax(out, dim=1)[:, 1].cpu())
                    ys.append(batch.y.cpu())
            return torch.cat(probs), torch.cat(ys)

        test_probs_t, test_y_t = _predict_test()
        latency = measure_inference_latency(lambda: _predict_test(), n_repeats=10, warmup=2)
        test_pred_t = predict_labels(test_probs_t.numpy(), best_gnn["threshold"])
        metrics = compute_metrics(torch.as_tensor(test_pred_t), test_y_t, test_probs_t)

        r = {
            "dataset": "batadal", "model": f"{conv_type}_window_classifier",
            "model_family": conv_type, "split_protocol": "temporal_purged", "seed": seed,
            "train_size": len(fit_graphs), "validation_size": len(val_graphs),
            "test_size": len(test_graphs), "feature_set": "sensor_window_stats",
            "graph_mode": "windowed", "threshold": best_gnn["threshold"], "runtime": t.elapsed,
            "n_trainable_params": count_trainable_params(best_gnn["model"]),
            "inference_latency_ms": latency,
            "per_attack_recall": {"n/a": None},
            "best_hyperparams": {"hidden_dim": best_gnn["hidden_dim"], "lr": best_gnn["lr"],
                                  "dropout": best_gnn["dropout"]},
            "hyperparam_search_log": searched_gnn,
            "notes": {"window_size": extra["window_size"], "stride": extra["stride"],
                      "purge_meta": purge_meta},
            **metrics,
        }
        results.append(r)
        if on_result:
            on_result(r)

    return results


# ---------------------------------------------------------------------
# CLI + result-artifact writing
# ---------------------------------------------------------------------

RESULTS_DIR = Path("experiments/results/q1/baselines")


def _result_to_row(r: Dict) -> Dict:
    return {
        "dataset": r["dataset"], "split_protocol": r["split_protocol"],
        "model": r["model"], "model_family": r.get("model_family"), "seed": r["seed"],
        "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
        "auprc": r.get("auprc"), "auroc": r.get("auroc"), "fpr": r["fpr"],
        "n_trainable_params": r.get("n_trainable_params"),
        "inference_latency_ms_mean": (r.get("inference_latency_ms") or {}).get("mean_ms"),
        "train_runtime_s": r["runtime"], "threshold": r["threshold"],
    }


def _append_result_to_csv(r: Dict, out_path: Path) -> None:
    """Append ONE result row immediately (mode='a'), rather than
    accumulating a full run's results in memory and writing once at the
    end. This is the fix for a multi-hour B1 run losing all progress on
    interruption — each family's row lands on disk within seconds of
    that family finishing, not after all six complete."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([_result_to_row(r)])
    header = not out_path.exists()
    row_df.to_csv(out_path, mode="a", header=header, index=False)


def _results_to_csv(results: List[Dict], out_path: Path) -> None:
    """Bulk variant, kept for programmatic (non-CLI) use — writes/append
    a whole batch of results in one go. The CLI path below uses
    `_append_result_to_csv` per-result instead so a run gets checkpointed
    incrementally rather than waiting for the whole experiment to
    finish."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_result_to_row(r) for r in results])
    if out_path.exists():
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[baseline_runner] wrote {out_path} ({len(df)} rows total)")


def _write_per_model_jsons(results: List[Dict], subdir: Path) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fname = f"{r['dataset']}_{r['split_protocol']}_{r['model']}_seed{r['seed']}.json"
        write_result(dict(r), str(subdir / fname))


def _write_one_result(r: Dict, json_subdir: Path, csv_path: Path) -> None:
    json_subdir.mkdir(parents=True, exist_ok=True)
    fname = f"{r['dataset']}_{r['split_protocol']}_{r['model']}_seed{r['seed']}.json"
    write_result(dict(r), str(json_subdir / fname))
    _append_result_to_csv(r, csv_path)
    print(f"[baseline_runner] checkpointed {r['model']} (seed {r['seed']}) "
          f"-> {json_subdir / fname} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Issue #3 baseline experiments B1/B2/B3.")
    parser.add_argument("--experiment", required=True, choices=["b1", "b2", "b3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override default epoch budget (300 for SCADANet GNNs, 100 for BATADAL).")
    parser.add_argument("--families", default=None,
                         help="Comma-separated subset of model families to run this call "
                              "(e.g. --families graphsage,gcn). Default: all families for the "
                              "chosen experiment. Use this to split a long run across multiple "
                              "invocations/Colab cells and resume after an interruption — "
                              "results are checkpointed to disk after EACH family finishes, "
                              "so an interrupted run doesn't lose already-finished families.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Default: auto-detect (uses CUDA/GPU if available, e.g. Colab's "
                              "T4 — neither this nor the Issue #1 runner used the GPU before "
                              "this flag existed).")
    parser.add_argument("--best-alt-gnn", default="gcn", choices=["gcn", "gat"],
                         help="(b2 only) whichever of GCN/GAT scored higher in your B1 run.")
    parser.add_argument("--log-every", type=int, default=None,
                         help="Print train/val progress every N epochs (matches Issue #1's "
                              "runner.py style). Default: 50 for SCADANet GNNs/MLP, 20 for "
                              "BATADAL. Pass 0 to disable all per-epoch logging.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[baseline_runner] using device: {device}")
    families = args.families.split(",") if args.families else None

    kwargs = {"seed": args.seed, "device": device}
    if args.data_path:
        kwargs["csv_path"] = args.data_path
    if args.epochs:
        kwargs["epochs"] = args.epochs
    if families:
        kwargs["families"] = families
    if args.log_every is not None:
        kwargs["log_every"] = args.log_every

    if args.experiment == "b1":
        csv_path = RESULTS_DIR / "scadanet_trackb_baselines.csv"
        json_subdir = RESULTS_DIR / "b1"
        on_result = lambda r: _write_one_result(r, json_subdir, csv_path)
        run_experiment_b1(on_result=on_result, **kwargs)
    elif args.experiment == "b2":
        csv_path = RESULTS_DIR / "scadanet_temporal_baselines.csv"
        json_subdir = RESULTS_DIR / "b2"
        on_result = lambda r: _write_one_result(r, json_subdir, csv_path)
        kwargs["best_alt_gnn"] = args.best_alt_gnn
        run_experiment_b2(on_result=on_result, **kwargs)
    else:
        csv_path = RESULTS_DIR / "batadal_temporal_baselines.csv"
        json_subdir = RESULTS_DIR / "b3"
        on_result = lambda r: _write_one_result(r, json_subdir, csv_path)
        kwargs["alt_gnn"] = args.best_alt_gnn
        run_experiment_b3(on_result=on_result, **kwargs)

    print(f"[baseline_runner] experiment {args.experiment} complete.")


if __name__ == "__main__":
    main()
