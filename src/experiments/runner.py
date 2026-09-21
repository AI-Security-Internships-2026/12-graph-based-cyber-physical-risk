"""
Single entry point for every paper experiment.

    python -m src.experiments.runner \
        --dataset scadanet --split track_b --seed 42 \
        --output experiments/results/q1/scadanet_graphsage_trackb_seed42.json

No notebook cells need to be run first — everything below is the notebook
logic moved into src/, not reinvented. Two architectures are used, matching
what the notebooks actually did (do not merge them — they answer different
questions and use different models):

  - SCADANet Track B and SCADANet Temporal: flow/edge-level,
    EdgeClassifierWithAttr (Week 7 Part G.2 / Week 10 Part M / Week 12 Part X)
  - BATADAL Temporal and BATADAL Static-CV: window/graph-level,
    WindowGraphClassifier (Week 5 Part D / Week 11 Part P)
"""
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.batadal import (
    load_batadal_df, build_windowed_graphs, temporal_split_windows,
)
from src.data.scadanet import (
    load_scadanet_df, build_graph_tensors, add_temporal_windows,
    temporal_split_positions, LABEL_COL,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.splits import (
    scadanet_track_a_split, scadanet_track_b_split, batadal_static_cv_folds,
)
from src.models.gnn import (
    EdgeClassifier, EdgeClassifierWithAttr, WindowGraphClassifier,
    build_class_weights, compute_subtype_balanced_weights,
)
from src.utils.io import Timer, write_result
from src.utils.seed import set_seed


# ---------------------------------------------------------------------
# SCADANet — flow-level (EdgeClassifierWithAttr)
# ---------------------------------------------------------------------

def run_scadanet_track_a(seed: int, csv_path: str = None,
                          epochs: int = 200, lr: float = 0.01, log_every: int = 20) -> Dict:
    """Week 6 Cell 110 `train_eval_scadanet_ip_split` — Track A / IP-held-out
    split. Uses plain EdgeClassifier (topology only, no edge attributes —
    this is the simplest of the three SCADANet models) and plain binary
    class weighting. Note this is the split Issue #2's audit shows only
    covers 2/13 real attack types in practice — see split_audits/."""
    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)

    train_idx, test_idx, split_meta = scadanet_track_a_split(gt.query_edges, gt.all_ips, seed=seed)
    train_y, test_y = gt.y[train_idx], gt.y[test_idx]
    class_weights = build_class_weights(train_y)

    model = EdgeClassifier(in_dim=gt.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    with Timer() as t:
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(gt.x, gt.edge_index, gt.query_edges[:, train_idx])
            loss = F.cross_entropy(out, train_y, weight=class_weights)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % log_every == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    tr_pred = out.argmax(dim=1)
                    tr_m = compute_metrics(tr_pred, train_y)
                    test_out = model(gt.x, gt.edge_index, gt.query_edges[:, test_idx])
                    te_pred = test_out.argmax(dim=1)
                    te_m = compute_metrics(te_pred, test_y)
                model.train()
                print(f"epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.4f}  |  "
                      f"TRAIN acc={tr_m['accuracy']:.4f} f1={tr_m['f1']:.4f}  |  "
                      f"TEST acc={te_m['accuracy']:.4f} f1={te_m['f1']:.4f}")

        model.eval()
        with torch.no_grad():
            logits = model(gt.x, gt.edge_index, gt.query_edges[:, test_idx])
            probs = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

    metrics = compute_metrics(pred, test_y, probs)
    return {
        "dataset": "scadanet", "model": "graphsage_topology_only",
        "split_protocol": "track_a", "seed": seed,
        "train_size": int(len(train_idx)), "validation_size": None,
        "test_size": int(len(test_idx)), "feature_set": "topology_only",
        "graph_mode": "static", "threshold": 0.5, "runtime": t.elapsed,
        "notes": split_meta, **metrics,
    }


def run_scadanet_track_b(seed: int, csv_path: str = None,
                          epochs: int = 400, lr: float = 0.003, log_every: int = 20) -> Dict:
    """Week 8 Cell 123 `train_eval_scadanet_flowlevel` — Track B split
    (Cells 121/122) with PER-SAMPLE subtype-balanced loss weighting, NOT
    plain binary class weighting. Do not swap in build_class_weights()
    here — see compute_subtype_balanced_weights() docstring."""
    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)

    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    train_y, test_y = gt.y[train_idx], gt.y[test_idx]

    subtype_weights = compute_subtype_balanced_weights(df, LABEL_COL, train_idx)
    train_labels = df.iloc[train_idx.numpy()][LABEL_COL].values
    per_flow_weight = torch.tensor(
        [subtype_weights[l] for l in train_labels], dtype=torch.float32
    )

    model = EdgeClassifierWithAttr(in_dim=gt.x.shape[1], edge_attr_dim=gt.edge_attr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    with Timer() as t:
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(gt.x, gt.edge_index, gt.query_edges[:, train_idx], gt.edge_attr[train_idx])
            per_sample_loss = F.cross_entropy(out, train_y, reduction="none")
            loss = (per_sample_loss * per_flow_weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (epoch + 1) % log_every == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    tr_pred = out.argmax(dim=1)
                    tr_m = compute_metrics(tr_pred, train_y)
                    test_out = model(gt.x, gt.edge_index, gt.query_edges[:, test_idx], gt.edge_attr[test_idx])
                    te_pred = test_out.argmax(dim=1)
                    te_m = compute_metrics(te_pred, test_y)
                model.train()
                print(f"epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.4f}  |  "
                      f"TRAIN acc={tr_m['accuracy']:.4f} f1={tr_m['f1']:.4f}  |  "
                      f"TEST acc={te_m['accuracy']:.4f} f1={te_m['f1']:.4f}")

        model.eval()
        with torch.no_grad():
            logits = model(gt.x, gt.edge_index, gt.query_edges[:, test_idx], gt.edge_attr[test_idx])
            probs = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

    metrics = compute_metrics(pred, test_y, probs)
    return {
        "dataset": "scadanet", "model": "graphsage_edge_attr",
        "split_protocol": "track_b", "seed": seed,
        "train_size": int(len(train_idx)), "validation_size": None,
        "test_size": int(len(test_idx)), "feature_set": "enriched_edge_attr",
        "graph_mode": "static", "threshold": 0.5, "runtime": t.elapsed,
        "notes": {**split_meta, "loss_weighting": "per_sample_subtype_balanced",
                  "subtype_weights": subtype_weights},
        **metrics,
    }


def run_scadanet_temporal(seed: int, csv_path: str = None, n_windows: int = 20,
                           epochs: int = 400, lr: float = 0.003, log_every: int = 20) -> Dict:
    """Week 10 Part M / Week 12 Part X (Cell 190):
    train_eval_scadanet_temporal_seeded — trains on the streaming topology
    snapshot as of the split boundary, evaluates on the final (full)
    topology snapshot, query_edges/edge_attr/y reordered chronologically
    to match df_t's row order before indexing by train/test position."""
    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)  # ip_index/all_ips/x are order-independent

    df_t, snapshots = add_temporal_windows(df, gt.ip_index, n_windows=n_windows)
    orig_idx = df_t["orig_idx"].to_numpy()

    query_edges_t = gt.query_edges[:, orig_idx]
    edge_attr_t = gt.edge_attr[orig_idx]
    y_t = gt.y[orig_idx]

    train_pos, test_pos, split_window = temporal_split_positions(df_t)
    train_idx = torch.tensor(train_pos, dtype=torch.long)
    test_idx = torch.tensor(test_pos, dtype=torch.long)
    train_y, test_y = y_t[train_idx], y_t[test_idx]

    n_nodes = len(gt.all_ips)
    x = torch.ones((n_nodes, 4), dtype=torch.float32)
    train_data = Data(x=x, edge_index=snapshots[split_window - 1]["edge_index"])
    eval_data = Data(x=x, edge_index=snapshots[-1]["edge_index"])

    class_weights = build_class_weights(train_y)
    model = EdgeClassifierWithAttr(in_dim=x.shape[1], edge_attr_dim=edge_attr_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    with Timer() as t:
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index,
                        query_edges_t[:, train_idx], edge_attr_t[train_idx])
            loss = F.cross_entropy(out, train_y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (epoch + 1) % log_every == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    tr_pred = out.argmax(dim=1)
                    tr_m = compute_metrics(tr_pred, train_y)
                    test_out = model(eval_data.x, eval_data.edge_index,
                                      query_edges_t[:, test_idx], edge_attr_t[test_idx])
                    te_pred = test_out.argmax(dim=1)
                    te_m = compute_metrics(te_pred, test_y)
                model.train()
                print(f"epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.4f}  |  "
                      f"TRAIN acc={tr_m['accuracy']:.4f} f1={tr_m['f1']:.4f}  |  "
                      f"TEST acc={te_m['accuracy']:.4f} f1={te_m['f1']:.4f}")

        model.eval()
        with torch.no_grad():
            logits = model(eval_data.x, eval_data.edge_index,
                            query_edges_t[:, test_idx], edge_attr_t[test_idx])
            probs = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

    metrics = compute_metrics(pred, test_y, probs)
    return {
        "dataset": "scadanet", "model": "graphsage_edge_attr",
        "split_protocol": "temporal", "seed": seed,
        "train_size": int(len(train_idx)), "validation_size": None,
        "test_size": int(len(test_idx)), "feature_set": "enriched_edge_attr",
        "graph_mode": "streaming_snapshot", "threshold": 0.5, "runtime": t.elapsed,
        "notes": {"n_windows": n_windows, "split_window": split_window},
        **metrics,
    }


# ---------------------------------------------------------------------
# BATADAL — window/graph-level (WindowGraphClassifier)
# ---------------------------------------------------------------------

def _train_eval_window_classifier(train_graphs, test_graphs, seed, epochs=100, lr=0.01,
                                   log_every: int = 20, fold_label: str = ""):
    """Week 5 Part D `run_fold` (Cell 92), used identically by both the
    BATADAL temporal split and the BATADAL static-CV folds. The original
    notebook only printed a one-line summary per fold at the end (visible
    in the calling loop, Cell 92) — this adds optional intra-fold epoch
    printing since 100 epochs over 5 folds can otherwise look silent for
    a while; set log_every=0 to suppress and match the original exactly."""
    set_seed(seed)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    train_y = torch.cat([g.y for g in train_graphs])
    class_weights = build_class_weights(train_y)

    model = WindowGraphClassifier(in_dim=train_graphs[0].x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    with Timer() as t:
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y, weight=class_weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
                print(f"{fold_label}epoch {epoch + 1:4d}/{epochs}  "
                      f"avg_loss={epoch_loss / max(len(train_loader), 1):.4f}")

        model.eval()
        all_probs, all_pred, all_true = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                all_probs.append(F.softmax(out, dim=1)[:, 1])
                all_pred.append(out.argmax(dim=1))
                all_true.append(batch.y)

    pred, true, probs = torch.cat(all_pred), torch.cat(all_true), torch.cat(all_probs)
    return compute_metrics(pred, true, probs), t.elapsed


def run_batadal_temporal(seed: int, csv_path: str = "BATADAL_dataset04.csv",
                          epochs: int = 100, lr: float = 0.01) -> Dict:
    """Week 11 Part P (Cell 171): chronological 70/30 split on windows."""
    df = load_batadal_df(csv_path)
    window_graphs, window_labels, extra = build_windowed_graphs(df)
    train_graphs, test_graphs = temporal_split_windows(window_graphs)

    metrics, runtime = _train_eval_window_classifier(train_graphs, test_graphs, seed, epochs, lr,
                                                       fold_label="[batadal temporal] ")
    return {
        "dataset": "batadal", "model": "graphsage_window_classifier",
        "split_protocol": "temporal", "seed": seed,
        "train_size": len(train_graphs), "validation_size": None,
        "test_size": len(test_graphs), "feature_set": "sensor_window_stats",
        "graph_mode": "windowed", "threshold": 0.5, "runtime": runtime,
        "notes": {"window_size": extra["window_size"], "stride": extra["stride"]},
        **metrics,
    }


def run_batadal_static_cv(seed: int, csv_path: str = "BATADAL_dataset04.csv",
                           n_folds: int = 5, epochs: int = 100, lr: float = 0.01) -> Dict:
    """Week 5 Part D (Cell 92): StratifiedKFold(5, shuffle=True, random_state=seed)."""
    df = load_batadal_df(csv_path)
    window_graphs, window_labels, extra = build_windowed_graphs(df)
    folds = batadal_static_cv_folds(window_labels, n_folds=n_folds, seed=seed)

    fold_metrics = []
    total_runtime = 0.0
    for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
        train_graphs = [window_graphs[i] for i in train_idx]
        test_graphs = [window_graphs[i] for i in test_idx]
        metrics, runtime = _train_eval_window_classifier(
            train_graphs, test_graphs, seed, epochs, lr,
            fold_label=f"[batadal static_cv fold {fold_num}/{n_folds}] ")
        fold_metrics.append(metrics)
        total_runtime += runtime

    avg = {k: float(np.mean([m[k] for m in fold_metrics if m[k] is not None]))
           for k in ("accuracy", "precision", "recall", "f1")}

    return {
        "dataset": "batadal", "model": "graphsage_window_classifier",
        "split_protocol": "static_cv", "seed": seed,
        "train_size": None, "validation_size": None,
        "test_size": len(window_graphs), "feature_set": "sensor_window_stats",
        "graph_mode": "windowed", "threshold": 0.5, "runtime": total_runtime,
        "auprc": None, "fpr": None, "confusion_matrix": None,
        "notes": {"n_folds": n_folds, "per_fold": fold_metrics,
                  "window_size": extra["window_size"], "stride": extra["stride"]},
        **avg,
    }


REGISTRY = {
    ("scadanet", "track_a"): run_scadanet_track_a,
    ("scadanet", "track_b"): run_scadanet_track_b,
    ("scadanet", "temporal"): run_scadanet_temporal,
    ("batadal", "temporal"): run_batadal_temporal,
    ("batadal", "static_cv"): run_batadal_static_cv,
}


def main():
    parser = argparse.ArgumentParser(description="Run one reproducible paper experiment.")
    parser.add_argument("--dataset", required=True, choices=["scadanet", "batadal"])
    parser.add_argument("--split", required=True, choices=["track_a", "track_b", "temporal", "static_cv"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", default=None,
                         help="CSV path (SCADANet defaults to kagglehub download; "
                              "BATADAL defaults to ./BATADAL_dataset04.csv)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    key = (args.dataset, args.split)
    if key not in REGISTRY:
        raise SystemExit(f"No runner for dataset={args.dataset} split={args.split}. "
                          f"Available: {list(REGISTRY.keys())}")

    kwargs = {}
    if args.data_path:
        kwargs["csv_path"] = args.data_path

    result = REGISTRY[key](args.seed, **kwargs)
    write_result(result, args.output)


if __name__ == "__main__":
    main()
