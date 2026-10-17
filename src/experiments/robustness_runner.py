"""Robustness across seeds, paired statistical comparisons, and operational IDS metrics.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.scadanet import load_scadanet_df, build_graph_tensors, LABEL_COL
from src.evaluation.operational_metrics import (
    false_alarms_per_unit_time, group_into_events, event_level_recall, time_to_detect,
)
from src.evaluation.robustness_stats import (
    descriptive_stats, paired_comparison, holm_adjust, expected_calibration_error, brier_score,
)
from src.evaluation.splits import scadanet_track_b_split, carve_validation
from src.experiments.ablation_runner import run_condition_matrix_trackb
from src.experiments.baseline_runner import (
    run_experiment_b1, gnn_edge_family, nongnn_family, _gnn_edge_probs, DEFAULT_GRID,
)
from src.experiments.host_norm_runner import run_trackb_host_normalization,run_temporal_host_normalization
from src.experiments.temporal_runner import run_scadanet_issue5
from src.models.baselines import predict_labels
from src.utils.io import get_git_commit, get_library_versions
from src.utils.seed import set_seed

RESULTS_DIR = Path("experiments/results/q1/robustness")

# Both a random-topology (Condition D) row and a degree-preserving rewired (Condition E)
# row are run, because "rewired topology" could refer to either.
PRINCIPAL_PAIRS = [
    ("GraphSAGE_trackb", "MLP_trackb", "f1"),
    ("GraphSAGE_trackb", "Tree_trackb", "f1"),  # added: real B1 numbers show
    # Tree is included as a second non-GNN baseline. Which of MLP/Tree is better must be
    # decided from the multi-seed results, NOT from any single-seed test score.
    ("GraphSAGE_trackb", "RandomTopology_trackb", "f1"),
    ("GraphSAGE_trackb", "DegreePreservedTopology_trackb", "f1"),
    ("Original_H1_trackb", "HostNormalized_H4_trackb", "fpr"),
    ("MatchedGraphSAGE_trackb", "GraphSAGE_strict_temporal", "recall"),
    ("GraphSAGE_trackb","ContentOnly_trackb","f1"),
    ("Original_H1_temporal","HostNormalized_H4_temporal","fpr"),
]


def _row(seed: int, condition: str, r: Dict) -> Dict:
    return {
        "seed": seed, "condition": condition,
        "precision": r.get("precision"), "recall": r.get("recall"),
        "f1": r.get("f1"), "auprc": r.get("auprc"), "fpr": r.get("fpr"),
        "n_trainable_params": r.get("n_trainable_params"),
        "inference_latency_ms": (r.get("inference_latency_ms") or {}).get("mean_ms")
        if isinstance(r.get("inference_latency_ms"), dict) else r.get("inference_latency_ms"),
    }


# ---------------------------------------------------------------------
# Multi-seed robustness
# ---------------------------------------------------------------------

def run_multi_seed_principal(
    seeds: List[int], csv_path: str = None, epochs: int = 300,
    host_epochs: int = 400, temporal_epochs: int = 300, log_every: int = 0,
) -> pd.DataFrame:
    """Run the principal conditions for each seed: MLP_trackb, Tree_trackb,
    GraphSAGE_trackb (also serves as the full-topology Condition C), ContentOnly_trackb
    (A), RandomTopology_trackb (D), DegreePreservedTopology_trackb,
    GraphSAGE_strict_temporal, and Original_H1_trackb / HostNormalized_H4_trackb.
    Appends to any existing `multi_seed_results.csv` for seeds not already present, so
    seeds can be added incrementally.
    """
    existing = pd.read_csv(RESULTS_DIR / "multi_seed_results.csv") if (
        RESULTS_DIR / "multi_seed_results.csv").exists() else pd.DataFrame(columns=["seed", "condition"])

    rows: List[Dict] = []
    for s in seeds:
        print(f"[robustness_runner][A] ===== seed {s} =====")

        b1 = run_experiment_b1(seed=s, csv_path=csv_path, epochs=epochs,
                                families=["mlp", "tree", "graphsage"], log_every=log_every)
        # Both are run: MLP_trackb and Tree_trackb. Choose between them from the multi-
        # seed results, not from one seed.
        _b1_name = {"mlp": "MLP_trackb", "tree": "Tree_trackb", "graphsage": "GraphSAGE_trackb"}
        for r in b1:
            rows.append(_row(s, _b1_name[r["model_family"]], r))

        abl = run_condition_matrix_trackb(seed=s, csv_path=csv_path, epochs=epochs,
                                           conditions=["A", "B", "D", "E", "F"], log_every=log_every)
        for r in abl:
            name = {"A_content_only": "ContentOnly_trackb", "B_topology_only":"TopologyOnly_trackb","F_no_message_passing":"NoMessagePassing_trackb",
                     "D_random_topology": "RandomTopology_trackb",
                     "E_degree_preserving_rewiring": "DegreePreservedTopology_trackb"}[r["condition"]]
            rows.append(_row(s, name, r))

        t5 = run_scadanet_issue5(seed=s, csv_path=csv_path or None, epochs=temporal_epochs,
                                  log_every=log_every)
        rows.append(_row(s, "GraphSAGE_strict_temporal", t5["t2"]))
        rows.append(_row(s,"MatchedGraphSAGE_trackb",t5["matched_trackb"]))

        for protocol,fn in [("trackb",run_trackb_host_normalization),("temporal",run_temporal_host_normalization)]:
            h_rows=fn(seed=s,csv_path=csv_path,epochs=host_epochs if protocol=="trackb" else temporal_epochs,log_every=log_every)
            for r in h_rows:
                prefix="Original_" if r["condition"].startswith(("H1","H2")) else "HostNormalized_"
                rows.append(_row(s,prefix+r["condition"][:2]+"_"+protocol,r))
        # Checkpoint every completed seed, including both host-normalization protocols.
        RESULTS_DIR.mkdir(parents=True,exist_ok=True)
        pd.concat([existing,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(["seed","condition"],keep="last").to_csv(RESULTS_DIR/"multi_seed_results.csv",index=False)

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["seed", "condition"], keep="last")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "multi_seed_results.csv"
    combined.to_csv(out, index=False)
    print(f"[robustness_runner][A] wrote {out} ({len(combined)} rows, "
          f"{combined['seed'].nunique()} distinct seeds)")
    return combined


def summarize_multi_seed(df: pd.DataFrame = None) -> pd.DataFrame:
    """Table S1 — mean/std/median/95% CI per condition, across whichever
    seeds are in `multi_seed_results.csv`."""
    if df is None:
        df = pd.read_csv(RESULTS_DIR / "multi_seed_results.csv")
    rows = []
    for condition, g in df.groupby("condition"):
        row = {"condition": condition, "n_seeds": g["seed"].nunique()}
        for metric in ["f1", "auprc", "fpr", "precision", "recall"]:
            if metric not in g.columns or g[metric].isna().all():
                continue
            stats = descriptive_stats(g[metric].dropna().to_numpy())
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_ci_95_low"] = stats["ci_95_low"]
            row[f"{metric}_ci_95_high"] = stats["ci_95_high"]
        rows.append(row)
    out_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_DIR / "table_s1_robustness_summary.csv", index=False)
    print(f"[robustness_runner][A] wrote {RESULTS_DIR / 'table_s1_robustness_summary.csv'}")
    return out_df


# ---------------------------------------------------------------------
# Paired statistical comparisons
# ---------------------------------------------------------------------

def compute_paired_statistics(df: pd.DataFrame = None) -> pd.DataFrame:
    """Table S2 — the four required paired comparisons (PRINCIPAL_PAIRS),
    computed only over seeds present for BOTH conditions in a pair (a
    seed missing from one side of a pair is dropped from that pair's
    comparison, not imputed or treated as zero)."""
    if df is None:
        df = pd.read_csv(RESULTS_DIR / "multi_seed_results.csv")

    rows = []
    for cond_a, cond_b, metric in PRINCIPAL_PAIRS:
        a_df = df[df["condition"] == cond_a][["seed", metric]].dropna()
        b_df = df[df["condition"] == cond_b][["seed", metric]].dropna()
        merged = a_df.merge(b_df, on="seed", suffixes=("_a", "_b"))
        if len(merged) < 2:
            rows.append({
                "comparison": f"{cond_a} vs {cond_b}", "metric": metric,
                "n_seeds": len(merged),
                "test_used": f"too few matched seeds ({len(merged)}) — need both "
                              f"conditions run for the same seeds first",
            })
            continue
        rows.append(paired_comparison(
            merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy(),
            cond_a, cond_b, metric))

    _adj = holm_adjust([r.get("p_value") for r in rows])
    for r, a_ in zip(rows, _adj):
        r["p_value_holm"] = a_
    out_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_DIR / "paired_statistics.csv", index=False)
    print(f"[robustness_runner][B] wrote {RESULTS_DIR / 'paired_statistics.csv'}")
    return out_df


# ---------------------------------------------------------------------
# Operational IDS metrics (ONE "final" model, not multi-seed)
# ---------------------------------------------------------------------

def compute_operational_metrics(
    model_choice: str = "graphsage", seed: int = 42, csv_path: str = None, epochs: int = 300,
    val_fraction: float = 0.15, seconds_per_time_unit: float = 3600.0, event_gap_seconds: float = 60.0,
) -> Dict:
    """Table S3: one row for `model_choice` ("graphsage", "mlp" or "tree"), the final Track
    B model of that family. It is trained once, so it is the same model that Table S1's
    *_trackb rows report, with raw probabilities kept for calibration and time-based
    metrics.

    Assumptions (see the module docstring of `operational_metrics.py`): the Time column
    is in seconds and events are grouped by a 60s gap heuristic. Neither is a dataset-
    given fact.
    """
    if model_choice not in ("graphsage", "mlp", "tree"):
        raise ValueError(f"model_choice must be 'graphsage', 'mlp' or 'tree', got {model_choice!r}")

    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)
    gt = build_graph_tensors(df, fit_idx=fit_idx.cpu().numpy())

    if model_choice == "graphsage":
        result, model, device, x_dev, edge_index_dev = gnn_edge_family(
            "graphsage", gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y,
            df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=DEFAULT_GRID,
            subtype_balanced=True, return_model=True)
        test_probs = _gnn_edge_probs(model, x_dev, edge_index_dev, gt.query_edges, gt.edge_attr,
                                      test_idx, device)
    else:
        result, model, X_test, test_probs = nongnn_family(
            model_choice, gt.edge_attr, gt.y, df, fit_idx, val_idx, test_idx, seed, grid=DEFAULT_GRID,
            subtype_balanced=True, mlp_epochs=epochs, return_model=True)

    test_pred = predict_labels(test_probs, result["threshold"])
    test_true = gt.y[test_idx].numpy()

    metrics = {
        "auprc": result["auprc"], "fpr": result["fpr"], "precision": result["precision"],
        "recall": result["recall"], "f1": result["f1"],
        "n_trainable_params": result["n_trainable_params"],
        "inference_latency_ms": result["inference_latency_ms"],
        "ece": expected_calibration_error(test_probs, test_true),
        "brier_score": brier_score(test_probs, test_true),
        "model_size_bytes":result.get("model_size_bytes"),
    }

    # Track B is a randomly sampled test set, not a continuous deployment stream.
    for key in ("false_alarms_per_hour","event_recall","time_to_detect_mean_s","time_to_detect_median_s"):
        metrics[key]=None
    metrics["false_alarms_per_hour_note"]="N/A: Track-B random sample has no continuous exposure denominator"
    metrics["event_metrics_note"]="N/A: sampled flows and no verified event IDs; event reconstruction would be heuristic"
    metrics["model"] = f"{model_choice}_trackb_final"
    metrics["seed"] = seed
    metrics["git_commit"] = get_git_commit()
    metrics["library_versions"] = get_library_versions()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "operational_metrics.csv"
    row_df = pd.DataFrame([{k: v for k, v in metrics.items()
                             if k not in ("git_commit", "library_versions", "inference_latency_ms")}])
    row_df["inference_latency_ms"] = json.dumps(metrics["inference_latency_ms"])
    seed_file=RESULTS_DIR/f"operational_{model_choice}_seed{seed}.csv"
    row_df.to_csv(seed_file,index=False)
    pd.concat([pd.read_csv(f) for f in RESULTS_DIR.glob("operational_*_seed*.csv")],ignore_index=True).to_csv(out_path,index=False)
    print(f"[robustness_runner][C] wrote {out_path}")
    return metrics


def collect_principal_from_artifacts(seeds):
    root=RESULTS_DIR.parent;rows=[]
    for seed in seeds:
        for family,name in [("mlp","MLP_trackb"),("tree","Tree_trackb"),("graphsage","GraphSAGE_trackb")]:
            candidates=[]
            for f in (root/"baselines"/"b1").glob(f"*_seed{seed}.json"):
                r=json.loads(f.read_text())
                if r.get("model_family")==family:candidates.append(r)
            if len(candidates)!=1:raise ValueError(f"Need one B1 {family} result for seed {seed}")
            rows.append(_row(seed,name,candidates[0]))
        for cond,name in [("A_content_only","ContentOnly_trackb"),("D_random_topology","RandomTopology_trackb"),
                          ("E_degree_preserving_rewiring","DegreePreservedTopology_trackb"),("B_topology_only","TopologyOnly_trackb"),("F_no_message_passing","NoMessagePassing_trackb")]:
            found=list((root/"ablations").rglob(f"*{cond}_seed{seed}.json"))
            found=[f for f in found if json.loads(f.read_text()).get("split_protocol")=="track_b"]
            if len(found)!=1:raise ValueError(f"Need one Track-B {cond} result for seed {seed}")
            rows.append(_row(seed,name,json.loads(found[0].read_text())))
        temporal=pd.read_csv(root/"temporal"/f"scadanet_temporal_summary_seed{seed}.csv")
        rows.append(_row(seed,"GraphSAGE_strict_temporal",temporal[temporal.protocol=="T2_strict_causal_topology"].iloc[0].to_dict()))
        matched=json.loads((root/"temporal"/f"matched_trackb_seed{seed}.json").read_text())
        rows.append(_row(seed,"MatchedGraphSAGE_trackb",matched["metrics"]))
        for protocol in ("trackb","temporal"):
            h=pd.read_csv(root/"host_normalization"/f"{protocol}_host_normalization_summary_seed{seed}.csv")
            for _,r in h.iterrows():
                prefix="Original_" if r.condition.startswith(("H1","H2")) else "HostNormalized_"
                rows.append(_row(seed,prefix+r.condition[:2]+"_"+protocol,r.to_dict()))
    df=pd.DataFrame(rows);RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    df.to_csv(RESULTS_DIR/"multi_seed_results.csv",index=False)
    return df

def aggregate_operational():
    from src.evaluation.aggregation import summarize_frame
    frames=[pd.read_csv(f) for f in RESULTS_DIR.glob("operational_*_seed*.csv")]
    if not frames:raise FileNotFoundError("No operational per-seed files")
    for f in (RESULTS_DIR.parent/"temporal").glob("operational_strict_seed*.csv"):
        frames.append(pd.read_csv(f))
    for protocol in ("trackb","temporal"):
        for f in (RESULTS_DIR.parent/"host_normalization").glob(f"{protocol}_host_normalization_summary_seed*.csv"):
            h=pd.read_csv(f);h["model"]=h.condition+"_"+protocol
            h["false_alarms_per_hour_note"]="N/A: exposure/timestamp units not validated"
            h["event_metrics_note"]="N/A: no verified event IDs/onsets"
            frames.append(h)
    raw=pd.concat(frames,ignore_index=True)
    for col in ("inference_latency_ms",):
        raw[col]=raw[col].apply(lambda x: json.loads(x).get("mean_ms") if isinstance(x,str) and x.startswith("{") else x)
    out=summarize_frame(raw,["model"])
    out.to_csv(RESULTS_DIR/"operational_metrics.csv",index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True, choices=["a", "b", "c", "all"])
    parser.add_argument("--seeds", default="42,7,123,1,2024",
                         help="comma-separated seed list, used by Part A")
    parser.add_argument("--seed", type=int, default=None, help="optional one-seed Part C override")
    parser.add_argument("--from-artifacts", action="store_true")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.part in ("a", "all"):
        df = collect_principal_from_artifacts(seeds) if args.from_artifacts else run_multi_seed_principal(seeds=seeds, csv_path=args.csv_path,
                                       epochs=args.epochs or 300, log_every=args.log_every)
        summarize_multi_seed(df)
    if args.part in ("b", "all"):
        compute_paired_statistics()
    if args.part in ("c", "all"):
        for seed in ([args.seed] if args.seed is not None else seeds):
            for model_choice in ("graphsage", "mlp", "tree"):
                compute_operational_metrics(model_choice=model_choice, seed=seed,
                                             csv_path=args.csv_path, epochs=args.epochs or 300)
        aggregate_operational()


if __name__ == "__main__":
    main()
