"""Strict leakage-free temporal evaluation, orchestration.

    python -m src.experiments.temporal_runner --experiment scadanet --seed 42
    python -m src.experiments.temporal_runner --experiment batadal --seed 42
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
    batadal_purged_split, batadal_window_bounds_df, TRACKED_SCADANET_ATTACKS, scadanet_topology_audit,
)
from src.experiments.runner import _train_eval_window_classifier
from src.models.gnn import EdgeClassifierWithAttr, build_class_weights
from src.utils.io import Timer, get_git_commit, get_library_versions
from src.utils.seed import set_seed
from src.evaluation.splits import scadanet_track_b_split
from src.utils.reproducibility import fingerprint, USED_MANIFESTS

RESULTS_DIR = Path("experiments/results/q1/temporal")


def _write_csv_with_seed_archive(df: pd.DataFrame, canonical_path: Path, seed: int) -> None:
    """Writes `canonical_path` exactly as before (so build_table_t_and_figures.py's
    fixed filenames keep working unchanged — it always reads the MOST RECENT run),
    but also writes a `..._seed{seed}.csv` copy alongside it. Without this, running
    the "additional seeds" cells after a canonical seed (e.g. 42) silently
    overwrites that seed's results before the tables get built from them, and no
    per-seed history survives a rerun. The archive copy is not read by anything
    yet — it's there so no seed's results are ever lost, and so a later multi-seed
    analysis has real per-seed files to draw from instead of only the last run."""
    df.to_csv(canonical_path, index=False)
    print(f"[temporal_runner] wrote {canonical_path}")
    archive_path = canonical_path.with_name(f"{canonical_path.stem}_seed{seed}{canonical_path.suffix}")
    df.to_csv(archive_path, index=False)
    print(f"[temporal_runner] wrote {archive_path} (per-seed archive)")


# ---------------------------------------------------------------------
# SCADANet
# ---------------------------------------------------------------------

def _train_frozen_scadanet_model(
    x, train_topology, query_edges_t, edge_attr_t, y_t, train_idx,
    epochs: int, lr: float, log_every: int,
) -> torch.nn.Module:
    """Same training loop as `run_scadanet_temporal` / `run_experiment_b2`. The model is
    trained once and never updated; every T1/T2/T3 evaluation reuses these frozen
    weights, so only the observed topology changes between evaluations.
    """
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


def _load_static_trackb_recall(seed, expected_config, dataset_dir=Path("experiments/results/q1/baselines/b1")):
    """Read a seed-specific JSON only when model, threshold and settings match.
    The matched control below is used when a tuned B1 run is incompatible.
    """
    path=dataset_dir / f"scadanet_track_b_graphsage_edge_attr_seed{seed}.json"
    if not path.exists(): return {}
    r=json.loads(path.read_text())
    if r.get("comparison_config") != expected_config: return {}
    if r.get("seed") != seed or r.get("threshold") != 0.5: return {}
    return r.get("per_attack_recall", {})


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
    gt = build_graph_tensors(df, fit_idx=orig_idx[train_idx].numpy())
    edge_attr_t = gt.edge_attr[orig_idx]

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
        from src.evaluation.prediction_diagnostics import diagnostics
        t2_operational={**diagnostics(t2_probs,t2_true,model),**t2_metrics,
             "n_trainable_params":sum(t.numel() for t in model.parameters()),
             "inference_latency_ms":model._last_inference_ms,"model":"graphsage_strict_temporal","seed":seed}

        # T3 — fixed training-topology protocol.
        t3_records, t3_pred, t3_true, t3_probs = scadanet_window_eval(
            model, x, query_edges_t, edge_attr_t, y_t, df_t, snapshots,
            split_window, n_windows, LABEL_COL, mode="fixed")
        t3_metrics = compute_metrics(t3_pred, t3_true, t3_probs)

    runtime = t.elapsed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t2_operational]).to_csv(RESULTS_DIR/f"operational_strict_seed{seed}.csv",index=False)

    config={"model":"graphsage_edge_attr_reference", "seed":seed, "epochs":epochs,
            "lr":lr,"threshold":0.5,"threshold_rule":"fixed_0.5_no_test_selection",
            "weighting":"binary_class_weighted","n_windows":n_windows,
            "preprocessing":"training_only", "data_sha256":fingerprint(df)}
    config_id=fingerprint({k:v for k,v in config.items() if k!="seed"})
    audit_rows=[]
    for mode in ("causal","fixed"):
        audit_rows += scadanet_topology_audit(df_t,snapshots,split_window,n_windows,gt.ip_index,mode)
    _write_csv_with_seed_archive(pd.DataFrame(audit_rows),RESULTS_DIR/"scadanet_topology_audit.csv",seed)
    # Matched Track-B control: same architecture, epochs, lr, loss and fixed threshold.
    # A separately tuned B1 threshold/model is not silently reused here.
    static_train,static_test,_=scadanet_track_b_split(df,LABEL_COL,seed=seed)
    static_gt=build_graph_tensors(df,fit_idx=static_train.numpy())
    set_seed(seed)
    static_model=_train_frozen_scadanet_model(static_gt.x,static_gt.edge_index,static_gt.query_edges,
         static_gt.edge_attr,static_gt.y,static_train,epochs,lr,log_every)
    with torch.no_grad():
        static_probs=F.softmax(static_model(static_gt.x,static_gt.edge_index,
            static_gt.query_edges[:,static_test],static_gt.edge_attr[static_test]),dim=1)[:,1]
    static_pred=(static_probs>0.5).long()
    static_recall=per_attack_recall(static_pred,static_gt.y[static_test],df.iloc[static_test.numpy()][LABEL_COL].to_numpy())
    static_metrics=compute_metrics(static_pred,static_gt.y[static_test],static_probs)
    (RESULTS_DIR/f"matched_trackb_seed{seed}.json").write_text(json.dumps({
        **config,"config_id":config_id,"per_attack_recall":static_recall,
        "metrics":static_metrics,"git_commit":get_git_commit()},indent=2,default=str))
    (RESULTS_DIR/f"scadanet_config_seed{seed}.json").write_text(json.dumps({
        **config,"config_id":config_id,"git_commit":get_git_commit(),"split_manifests":sorted(USED_MANIFESTS)},indent=2))
    pd.DataFrame({"seed":seed,"row_id":df_t.iloc[test_pos]["orig_idx"].to_numpy(),
         "true":t2_true.numpy(),"pred":t2_pred.numpy(),"probability":t2_probs.numpy(),
         "attack_type":df_t.iloc[test_pos][LABEL_COL].to_numpy()}).to_csv(RESULTS_DIR/f"scadanet_predictions_seed{seed}.csv",index=False)

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
    # Same seed-archive fix as summary_csv/per_attack_csv below — this
    # file feeds Figure T1/T3 (per-window F1/recall/FPR trajectories),
    # which vary by seed just like the aggregate summary does, so it was
    # just as exposed to a second seed silently erasing the first one's
    # per-window curve before this fix.
    _write_csv_with_seed_archive(pd.DataFrame(window_rows), window_csv, seed)

    # scadanet_temporal_summary.csv — Table T1 rows. split_window/n_windows are
    # duplicated onto every row (not just returned in-memory) so a later,
    # separate process (build_table_t_and_figures.py) can recover the
    # train/test boundary for Figure T2 without re-running the experiment.
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
    for row in summary_rows: row.update({"config_id":config_id,"threshold":0.5,"model":config["model"]})
    _write_csv_with_seed_archive(pd.DataFrame(summary_rows), summary_csv, seed)

    # Table T2 (per-attack, static/Track-B vs strict temporal recall).
    # static_recall was computed from the explicitly matched Track-B control above.
    t2_per_attack = per_attack_recall(t2_pred, t2_true,
                                       df_t.iloc[test_pos][LABEL_COL].to_numpy())
    per_attack_rows = []
    for attack, strict_recall in t2_per_attack.items():
        train_count = int((df_t.iloc[train_pos][LABEL_COL] == attack).sum())
        test_count = int((df_t.iloc[test_pos][LABEL_COL] == attack).sum())
        static_r = static_recall.get(attack)
        delta = (strict_recall - static_r) if (strict_recall is not None and static_r is not None) else None
        per_attack_rows.append({
            "attack_type": attack, "seed":seed,"config_id":config_id, "train_count": train_count, "test_count": test_count,
            "comparison_source":"matched_track_b_reference",
            "static_trackb_recall": static_r, "strict_temporal_recall": strict_recall,
            "delta_recall": delta,
        })
    per_attack_csv = RESULTS_DIR / "scadanet_temporal_per_attack.csv"
    _write_csv_with_seed_archive(pd.DataFrame(per_attack_rows), per_attack_csv, seed)

    growth_csv = RESULTS_DIR / "scadanet_graph_growth.csv"
    growth_df.to_csv(growth_csv, index=False)
    print(f"[temporal_runner] wrote {growth_csv}")

    return {
        "dataset": "scadanet", "seed": seed, "n_windows": n_windows,
        "split_window": split_window, "runtime": runtime,
        "t1": t1_metrics, "t2": t2_metrics, "t3": t3_metrics, "matched_trackb":static_metrics,
        "git_commit": get_git_commit(), "library_versions": get_library_versions(),
    }


# ---------------------------------------------------------------------
# BATADAL
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
        strict_graphs, _, _ = build_windowed_graphs(df,fit_rows=np.arange(split_meta["train_end_row"]))
        train_graphs = [strict_graphs[i] for i in train_idx]
        test_graphs = [strict_graphs[i] for i in test_idx]
        label = "T5_purged_70_30" if abs(frac - 0.7) < 1e-9 else f"T6_purged_{round(frac*100)}_{round((1-frac)*100)}"

        m, runtime = _train_eval_window_classifier(
            train_graphs, test_graphs, seed, epochs, lr, fold_label=f"[{label}] ")

        audit_result = audit_split(
            batadal_window_bounds_df(starts, window_size, train_idx),
            batadal_window_bounds_df(starts, window_size, test_idx),
            protocol_name=label,
            window_start_col="window_start", window_end_col="window_end",
        )
        assert audit_result["temporal_leakage_windows"]["n_test_windows_sharing_raw_rows_with_train"] == 0
        boundary_audit[label] = {**audit_result, "purge_meta": split_meta, "preprocessing":"train_rows_only"}

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

    for row in metrics_rows:
        row.update({"threshold":0.5,"threshold_rule":"fixed_0.5_no_test_selection",
          "config_id":fingerprint({"data":fingerprint(df),"epochs":epochs,"lr":lr,"splits":split_fracs})})
    metrics_csv = RESULTS_DIR / "batadal_purged_temporal_metrics.csv"
    _write_csv_with_seed_archive(pd.DataFrame(metrics_rows), metrics_csv, seed)

    audit_json = RESULTS_DIR / "batadal_boundary_audit.json"
    with open(audit_json, "w") as f:
        json.dump(boundary_audit, f, indent=2, default=str)
    print(f"[temporal_runner] wrote {audit_json}")
    audit_json_archive = audit_json.with_name(f"batadal_boundary_audit_seed{seed}.json")
    with open(audit_json_archive, "w") as f:
        json.dump(boundary_audit, f, indent=2, default=str)
    print(f"[temporal_runner] wrote {audit_json_archive} (per-seed archive)")

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
