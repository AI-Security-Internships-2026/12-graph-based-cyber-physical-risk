"""
Issue #5 — strict leakage-free temporal evaluation, orchestration.

    python -m src.experiments.temporal_runner --experiment scadanet --seed 42
    python -m src.experiments.temporal_runner --experiment batadal --seed 42

Writes every artifact named in Issue #5 section 3 under
experiments/results/q1/temporal/. Reuses Issue #1/#3's data loading,
splitting, and model code completely unchanged; only the evaluation
loop is new (src/evaluation/temporal_protocol.py) — this issue is about
fixing HOW test-time topology/windows are exposed to the model, not
about a new architecture or a new training procedure.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from src.data.batadal import (
    load_batadal_df, build_windowed_graphs, temporal_split_windows,
    WINDOW_SIZE, STRIDE,
)
from src.data.scadanet import (
    load_scadanet_df, build_graph_tensors, add_temporal_windows,
    temporal_split_positions, LABEL_COL, SRC_IP_COL, DST_IP_COL,
)
from src.evaluation.audit import audit_split
from src.evaluation.metrics import compute_metrics, per_attack_recall
from src.evaluation.temporal_protocol import (
    scadanet_window_eval, scadanet_future_leak_eval, scadanet_graph_growth,
    batadal_purged_split, batadal_window_bounds_df, TRACKED_SCADANET_ATTACKS,
)
from src.experiments.runner import _train_eval_window_classifier
from src.models.gnn import EdgeClassifierWithAttr, build_class_weights
from src.utils.io import Timer, get_git_commit, get_library_versions
from src.utils.seed import set_seed

RESULTS_DIR = Path("experiments/results/q1/temporal")


# ---------------------------------------------------------------------
# Part A — SCADANet
# ---------------------------------------------------------------------

def _train_frozen_scadanet_model(
    x, train_topology, query_edges_t, edge_attr_t, y_t, train_idx,
    epochs: int, lr: float, log_every: int,
) -> torch.nn.Module:
    """Same training loop as `run_scadanet_temporal`/`run_experiment_b2`
    (Issue #1/#3) — unchanged. The model is trained once here and never
    updated again; every T1/T2/T3 eval below reuses these exact frozen
    weights, per the issue's "frozen model + evolving observed topology"
    design choice."""
    train_y = y_t[train_idx]
    class_weights = build_class_weights(train_y)
    model = EdgeClassifierWithAttr(in_dim=x.shape[1], edge_attr_dim=edge_attr_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, train_topology, query_edges_t[:, train_idx], edge_attr_t[train_idx])
        loss = F.cross_entropy(out, train_y, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            with torch.no_grad():
                acc = (out.argmax(dim=1) == train_y).float().mean().item()
            print(f"[temporal_runner][scadanet] epoch {epoch + 1:4d}/{epochs} "
                  f"loss={loss.item():.4f} train_acc={acc:.4f}")
    model.eval()
    return model


def _load_static_trackb_recall(dataset_dir: Path = Path("experiments/results/q1/baselines")
                                ) -> Dict[str, Optional[float]]:
    """Table T2's "Static/Track-B recall" column, pulled from Issue #3's
    own saved results (scadanet_trackb_baselines.csv) if that file
    exists in this run's environment. Returns {} (not an error) when it
    doesn't — Issue #5 does not depend on Issue #3 having been run first
    in this session; Table T2 just leaves that column blank with a note
    when the comparison isn't available, per this repo's
    applicable/reason convention rather than fabricating a number."""
    csv_path = dataset_dir / "scadanet_trackb_baselines.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    gnn_rows = df[df["split_protocol"] == "track_b"]
    if "per_attack_recall" not in gnn_rows.columns or gnn_rows.empty:
        return {}
    # Prefer the graphsage row (the repo's reference GNN family) if present.
    row = gnn_rows[gnn_rows.get("model", "") == "graphsage"]
    row = row.iloc[0] if len(row) else gnn_rows.iloc[0]
    try:
        return json.loads(row["per_attack_recall"].replace("'", '"'))
    except Exception:
        return {}


def run_scadanet_issue5(seed: int = 42, csv_path: str = None, n_windows: int = 20,
                         epochs: int = 300, lr: float = 0.003, log_every: int = 50) -> Dict:
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

    n_nodes = len(gt.all_ips)
    x = torch.ones((n_nodes, 4), dtype=torch.float32)
    train_topology = snapshots[split_window - 1]["edge_index"]

    with Timer() as t:
        model = _train_frozen_scadanet_model(
            x, train_topology, query_edges_t, edge_attr_t, y_t, train_idx,
            epochs=epochs, lr=lr, log_every=log_every,
        )

        # T1 — previous protocol (future topology visible), historical reference.
        t1_pred, t1_true, t1_probs = scadanet_future_leak_eval(
            model, x, query_edges_t, edge_attr_t, y_t, test_idx, snapshots)
        t1_metrics = compute_metrics(t1_pred, t1_true, t1_probs)

        # T2 — strict cumulative-topology (causal) protocol. Primary result.
        t2_records, t2_pred, t2_true, t2_probs = scadanet_window_eval(
            model, x, query_edges_t, edge_attr_t, y_t, df_t, snapshots,
            split_window, n_windows, LABEL_COL, mode="causal")
        t2_metrics = compute_metrics(t2_pred, t2_true, t2_probs)

        # T3 — fixed training-topology protocol.
        t3_records, t3_pred, t3_true, t3_probs = scadanet_window_eval(
            model, x, query_edges_t, edge_attr_t, y_t, df_t, snapshots,
            split_window, n_windows, LABEL_COL, mode="fixed")
        t3_metrics = compute_metrics(t3_pred, t3_true, t3_probs)

    runtime = t.elapsed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # scadanet_temporal_window_metrics.csv — per-window T2 (and T3) diagnostics.
    window_rows = []
    growth_df = scadanet_graph_growth(df_t, snapshots, SRC_IP_COL, DST_IP_COL)
    growth_by_window = growth_df.set_index("window").to_dict(orient="index")
    for rec in t2_records + t3_records:
        row = dict(rec)
        g = growth_by_window.get(rec["window"], {})
        row["new_nodes"] = g.get("new_nodes")
        row["cumulative_nodes"] = g.get("cumulative_nodes")
        row["new_ip_pairs"] = g.get("new_ip_pairs")
        row["cumulative_edges"] = g.get("cumulative_edges")
        row["attack_type_counts"] = json.dumps(row.get("attack_type_counts", {}))
        row["per_attack_recall"] = json.dumps(row.get("per_attack_recall", {}))
        row["tracked_attack_recall"] = json.dumps(row.get("tracked_attack_recall", {}))
        window_rows.append(row)
    window_csv = RESULTS_DIR / "scadanet_temporal_window_metrics.csv"
    pd.DataFrame(window_rows).to_csv(window_csv, index=False)
    print(f"[temporal_runner] wrote {window_csv} ({len(window_rows)} rows)")

    # Save the split boundary with each row for downstream plotting.
    summary_rows = [
        {"dataset": "scadanet", "protocol": "T1_previous_temporal", "seed": seed,
         "leakage_condition": "future test topology possible",
         "precision": t1_metrics["precision"], "recall": t1_metrics["recall"],
         "f1": t1_metrics["f1"], "auprc": t1_metrics["auprc"], "fpr": t1_metrics["fpr"],
         "n_test": int(len(t1_true)), "split_window": split_window, "n_windows": n_windows},
        {"dataset": "scadanet", "protocol": "T2_strict_causal_topology", "seed": seed,
         "leakage_condition": "none identified",
         "precision": t2_metrics["precision"], "recall": t2_metrics["recall"],
         "f1": t2_metrics["f1"], "auprc": t2_metrics["auprc"], "fpr": t2_metrics["fpr"],
         "n_test": int(len(t2_true)), "split_window": split_window, "n_windows": n_windows},
        {"dataset": "scadanet", "protocol": "T3_fixed_training_topology", "seed": seed,
         "leakage_condition": "none identified (no topology update)",
         "precision": t3_metrics["precision"], "recall": t3_metrics["recall"],
         "f1": t3_metrics["f1"], "auprc": t3_metrics["auprc"], "fpr": t3_metrics["fpr"],
         "n_test": int(len(t3_true)), "split_window": split_window, "n_windows": n_windows},
    ]
    summary_csv = RESULTS_DIR / "scadanet_temporal_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[temporal_runner] wrote {summary_csv}")

    # Table T2 (per-attack, static/Track-B vs strict temporal recall).
    static_recall = _load_static_trackb_recall()
    t2_per_attack = per_attack_recall(t2_pred, t2_true,
                                       df_t.iloc[test_pos][LABEL_COL].to_numpy())
    per_attack_rows = []
    for attack, strict_recall in t2_per_attack.items():
        train_count = int((df_t.iloc[train_pos][LABEL_COL] == attack).sum())
        test_count = int((df_t.iloc[test_pos][LABEL_COL] == attack).sum())
        static_r = static_recall.get(attack)
        delta = (strict_recall - static_r) if (strict_recall is not None and static_r is not None) else None
        per_attack_rows.append({
            "attack_type": attack, "train_count": train_count, "test_count": test_count,
            "static_trackb_recall": static_r, "strict_temporal_recall": strict_recall,
            "delta_recall": delta,
        })
    per_attack_csv = RESULTS_DIR / "scadanet_temporal_per_attack.csv"
    pd.DataFrame(per_attack_rows).to_csv(per_attack_csv, index=False)
    print(f"[temporal_runner] wrote {per_attack_csv}")

    growth_csv = RESULTS_DIR / "scadanet_graph_growth.csv"
    growth_df.to_csv(growth_csv, index=False)
    print(f"[temporal_runner] wrote {growth_csv}")

    return {
        "dataset": "scadanet", "seed": seed, "n_windows": n_windows,
        "split_window": split_window, "runtime": runtime,
        "t1": t1_metrics, "t2": t2_metrics, "t3": t3_metrics,
        "git_commit": get_git_commit(), "library_versions": get_library_versions(),
    }


# ---------------------------------------------------------------------
# Part B — BATADAL
# ---------------------------------------------------------------------

def run_batadal_issue5(seed: int = 42, csv_path: str = "BATADAL_dataset04.csv",
                        epochs: int = 100, lr: float = 0.01,
                        split_fracs: Optional[List[float]] = None) -> Dict:
    split_fracs = split_fracs or [0.6, 0.7, 0.8]
    df = load_batadal_df(csv_path)
    window_graphs, window_labels, extra = build_windowed_graphs(df)
    starts, window_size = extra["starts"], extra["window_size"]
    n_rows = len(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    boundary_audit = {}

    # T4 — existing overlapping temporal split (historical reference).
    split_idx = int(len(window_graphs) * 0.7)
    t4_train_idx = list(range(0, split_idx))
    t4_test_idx = list(range(split_idx, len(window_graphs)))
    t4_train_graphs = [window_graphs[i] for i in t4_train_idx]
    t4_test_graphs = [window_graphs[i] for i in t4_test_idx]
    with Timer() as t:
        t4_metrics, t4_runtime = _train_eval_window_classifier(
            t4_train_graphs, t4_test_graphs, seed, epochs, lr, fold_label="[T4] ")
    t4_audit = audit_split(
        batadal_window_bounds_df(starts, window_size, t4_train_idx),
        batadal_window_bounds_df(starts, window_size, t4_test_idx),
        protocol_name="T4_overlapping_temporal",
        window_start_col="window_start", window_end_col="window_end",
    )
    metrics_rows.append({
        "dataset": "batadal", "protocol": "T4_overlapping_temporal", "seed": seed,
        "split": "70/30", "n_train_windows": len(t4_train_idx),
        "n_purged_windows": 0, "n_test_windows": len(t4_test_idx),
        "n_anomalous_train_windows": int(window_labels[t4_train_idx].sum()),
        "n_anomalous_test_windows": int(window_labels[t4_test_idx].sum()),
        "precision": t4_metrics["precision"], "recall": t4_metrics["recall"],
        "f1": t4_metrics["f1"], "auprc": t4_metrics["auprc"], "fpr": t4_metrics["fpr"],
    })
    boundary_audit["T4_overlapping_temporal"] = t4_audit

    # T5/T6 — purged chronological splits at each requested fraction.
    for frac in split_fracs:
        train_idx, test_idx, purged_idx, split_meta = batadal_purged_split(
            n_rows, starts, window_size, frac)
        train_graphs = [window_graphs[i] for i in train_idx]
        test_graphs = [window_graphs[i] for i in test_idx]
        label = "T5_purged_70_30" if abs(frac - 0.7) < 1e-9 else f"T6_purged_{round(frac*100)}_{round((1-frac)*100)}"

        m, runtime = _train_eval_window_classifier(
            train_graphs, test_graphs, seed, epochs, lr, fold_label=f"[{label}] ")

        audit_result = audit_split(
            batadal_window_bounds_df(starts, window_size, train_idx),
            batadal_window_bounds_df(starts, window_size, test_idx),
            protocol_name=label,
            window_start_col="window_start", window_end_col="window_end",
        )
        boundary_audit[label] = {**audit_result, "purge_meta": split_meta}

        metrics_rows.append({
            "dataset": "batadal", "protocol": label, "seed": seed,
            "split": f"{round(frac*100)}/{round((1-frac)*100)}",
            "n_train_windows": len(train_idx), "n_purged_windows": len(purged_idx),
            "n_test_windows": len(test_idx),
            "n_anomalous_train_windows": int(window_labels[train_idx].sum()),
            "n_anomalous_test_windows": int(window_labels[test_idx].sum()),
            "n_anomalous_purged_windows": int(window_labels[purged_idx].sum()) if len(purged_idx) else 0,
            "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "auprc": m["auprc"], "fpr": m["fpr"],
        })

    metrics_csv = RESULTS_DIR / "batadal_purged_temporal_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
    print(f"[temporal_runner] wrote {metrics_csv}")

    audit_json = RESULTS_DIR / "batadal_boundary_audit.json"
    with open(audit_json, "w") as f:
        json.dump(boundary_audit, f, indent=2, default=str)
    print(f"[temporal_runner] wrote {audit_json}")

    return {"dataset": "batadal", "seed": seed, "rows": metrics_rows,
            "git_commit": get_git_commit(), "library_versions": get_library_versions()}


def main():
    parser = argparse.ArgumentParser(description="Issue #5 strict temporal evaluation.")
    parser.add_argument("--experiment", required=True, choices=["scadanet", "batadal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--n-windows", type=int, default=20, help="SCADANet only.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50, help="SCADANet only.")
    parser.add_argument("--splits", default="0.6,0.7,0.8",
                         help="BATADAL only. Comma-separated train fractions for T5/T6.")
    args = parser.parse_args()

    if args.experiment == "scadanet":
        kwargs = {"seed": args.seed, "n_windows": args.n_windows, "log_every": args.log_every}
        if args.data_path:
            kwargs["csv_path"] = args.data_path
        if args.epochs:
            kwargs["epochs"] = args.epochs
        if args.lr:
            kwargs["lr"] = args.lr
        run_scadanet_issue5(**kwargs)
    else:
        kwargs = {"seed": args.seed, "split_fracs": [float(s) for s in args.splits.split(",")]}
        if args.data_path:
            kwargs["csv_path"] = args.data_path
        if args.epochs:
            kwargs["epochs"] = args.epochs
        if args.lr:
            kwargs["lr"] = args.lr
        run_batadal_issue5(**kwargs)

    print(f"[temporal_runner] experiment {args.experiment} complete.")


if __name__ == "__main__":
    main()
