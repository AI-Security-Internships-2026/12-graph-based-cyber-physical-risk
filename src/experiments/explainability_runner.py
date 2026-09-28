"""python -m src.experiments.explainability_runner --seed 42

Trains ONE SCADANet Track B GraphSAGE model, then runs the stratified-sample Integrated
Gradients analysis and the feature-intervention test against it. Needs `captum`
installed (`pip install captum`) — see `src/evaluation/explainability.py`'s module
docstring for what this file does NOT do (a literal GNNExplainer call) and why.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.data.scadanet import load_scadanet_df, build_graph_tensors, LABEL_COL
from src.evaluation.explainability import (
    stratified_sample, integrated_gradients_content, topology_occlusion_importance,
    run_feature_intervention,
)
from src.evaluation.splits import scadanet_track_b_split, carve_validation
from src.experiments.baseline_runner import gnn_edge_family, _gnn_edge_probs, DEFAULT_GRID
from src.models.baselines import predict_labels
from src.utils.seed import set_seed

RESULTS_DIR = Path("experiments/results/q1/robustness")

# Example intervention set — the top-3 content features by Integrated Gradients
# (computed from this run's TP+FP sample) are appended automatically; see main().
BASE_INTERVENTIONS = {
    "Original": None,
    "Protocol_TCP permuted": ["Protocol_TCP"],
    "Tcp_flags_reset_Set permuted": ["Tcp_flags_reset_Set"],
    "frame_len permuted": ["frame_len"],
}


def run_explainability(
    seed: int = 42, csv_path: str = None, epochs: int = 300, val_fraction: float = 0.15,
    n_per_category: int = 150, n_topology_examples_per_category: int = 5,
) -> Dict:
    set_seed(seed)
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)
    gt = build_graph_tensors(df, fit_idx=fit_idx.cpu().numpy())

    result, model, device, x_dev, edge_index_dev = gnn_edge_family(
        "graphsage", gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y,
        df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=DEFAULT_GRID,
        subtype_balanced=True, return_model=True)

    test_probs = _gnn_edge_probs(model, x_dev, edge_index_dev, gt.query_edges, gt.edge_attr,
                                  test_idx, device)
    test_pred = predict_labels(test_probs, result["threshold"])
    test_true = gt.y[test_idx].numpy()

    # Stratified sample (positions into test_idx).
    sample_positions = stratified_sample(test_pred, test_true, n_per_category=n_per_category, seed=seed,
        strata=(df.iloc[test_idx.numpy()]["Source"].astype(str)+"|"+df.iloc[test_idx.numpy()][LABEL_COL].astype(str)).to_numpy())
    print("[explainability_runner] stratified sample sizes: "
          + ", ".join(f"{k}={len(v)}" for k, v in sample_positions.items()))

    # "Include multiple hosts and attack types" — this session doesn't stratify sampling
    # BY host/attack type (only by TP/FP/TN/FN), so verify diversity rather than assume
    # a random sample of this size happens to cover it.
    diversity_rows = []
    for category, positions in sample_positions.items():
        if len(positions) == 0:
            continue
        cat_df = df.iloc[test_idx[torch.as_tensor(positions, dtype=torch.long)].numpy()]
        n_hosts = cat_df["Source"].nunique() if "Source" in cat_df.columns else None
        n_attack_types = cat_df[LABEL_COL].nunique() if LABEL_COL in cat_df.columns else None
        diversity_rows.append({"prediction_group": category, "n": len(positions),
                                "n_distinct_hosts": n_hosts, "n_distinct_attack_types": n_attack_types})
    diversity_df = pd.DataFrame(diversity_rows)
    print("[explainability_runner] sample diversity (hosts/attack types per category):")
    print(diversity_df.to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    diversity_csv = RESULTS_DIR / "explainability_sample_diversity.csv"
    diversity_df.to_csv(diversity_csv, index=False)
    print(f"[explainability_runner] wrote {diversity_csv}")

    # Integrated Gradients per category.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    ig_results = {}
    for category, positions in sample_positions.items():
        if len(positions) == 0:
            summary_rows.append({"prediction_group": category, "n": 0,
                                  "top_feature_1": None, "top_feature_2": None, "top_feature_3": None,
                                  "rank_stability": None, "note": "no examples in this category"})
            continue
        cat_idx = test_idx[torch.as_tensor(positions, dtype=torch.long)]
        ig = integrated_gradients_content(model, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr,
                                           cat_idx, gt.feature_names, device=device)
        ig_results[category] = ig
        top3 = [name for name, _ in ig["feature_ranking"][:3]]
        while len(top3) < 3:
            top3.append(None)
        summary_rows.append({
            "prediction_group": category, "n": ig["n"],
            "top_feature_1": top3[0], "top_feature_2": top3[1], "top_feature_3": top3[2],
            "rank_stability": ig["rank_stability"],
        })

    summary_csv = RESULTS_DIR / "explainability_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[explainability_runner] wrote {summary_csv}")

    # Full per-feature attribution rankings (needed for Figure S4; not
    # one of the issue's named files but Table S4 alone only keeps top-3).
    ranking_rows = []
    for category, ig in ig_results.items():
        for feature, mean_abs_attr in ig["feature_ranking"]:
            ranking_rows.append({"prediction_group": category, "feature": feature,
                                  "mean_abs_attribution": mean_abs_attr})
    ranking_csv = RESULTS_DIR / "explainability_feature_rankings.csv"
    pd.DataFrame(ranking_rows).to_csv(ranking_csv, index=False)
    print(f"[explainability_runner] wrote {ranking_csv}")

    # Topology occlusion on a handful of examples per category.
    topology_examples = []
    for category, positions in sample_positions.items():
        for pos in positions[:n_topology_examples_per_category]:
            query_pos = int(test_idx[pos])
            edge_scores = topology_occlusion_importance(
                model, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, query_pos, device=device)
            topology_examples.append({
                "prediction_group": category, "query_pos": query_pos,
                "top_5_edges_by_abs_prob_change": edge_scores[:5],
            })
    topology_json = RESULTS_DIR / "topology_occlusion_examples.json"
    with open(topology_json, "w") as f:
        json.dump(topology_examples, f, indent=2, default=str)
    print(f"[explainability_runner] wrote {topology_json} ({len(topology_examples)} examples)")

    # Feature intervention using the top-3 features from TP+FP combined.
    combined_positions = list(sample_positions.get("TP", [])) + list(sample_positions.get("FP", []))
    if not combined_positions:
        combined_positions=list(sample_positions.get("FN",[]))+list(sample_positions.get("TN",[]))
    top3_features: List[str] = []
    if combined_positions:
        combined_idx = test_idx[torch.as_tensor(combined_positions, dtype=torch.long)]
        ig_combined = integrated_gradients_content(model, gt.x, gt.edge_index, gt.query_edges,
                                                     gt.edge_attr, combined_idx, gt.feature_names,
                                                     device=device)
        top3_features = [name for name, _ in ig_combined["feature_ranking"][:3]]

    interventions = dict(BASE_INTERVENTIONS)
    if top3_features:
        interventions["Top-3 jointly perturbed"] = top3_features
    interventions["All content shuffled"] = gt.feature_names

    intervention_rows = run_feature_intervention(
        model, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y, test_idx,
        gt.feature_names, interventions, threshold=result["threshold"], device=device, seed=seed)
    intervention_csv = RESULTS_DIR / "feature_intervention_results.csv"
    pd.DataFrame(intervention_rows).to_csv(intervention_csv, index=False)
    print(f"[explainability_runner] wrote {intervention_csv}")

    # Preserve each seed, then aggregate from these files (including empty prediction groups).
    for stem in ("explainability_sample_diversity","explainability_summary","explainability_feature_rankings","feature_intervention_results"):
        file=RESULTS_DIR/f"{stem}.csv"
        frame=pd.read_csv(file);frame["seed"]=seed
        frame.to_csv(RESULTS_DIR/f"{stem}_seed{seed}.csv",index=False)
    (RESULTS_DIR/f"topology_occlusion_examples_seed{seed}.json").write_text(topology_json.read_text())
    (RESULTS_DIR/f"explainability_config_seed{seed}.json").write_text(json.dumps({"seed":seed,"top3_features":top3_features,"epochs":epochs},indent=2))
    return {"seed": seed, "summary_rows": summary_rows, "intervention_rows": intervention_rows,
            "top3_features": top3_features}


def aggregate_explanations():
    from src.evaluation.aggregation import read_seed_frames,summarize_frame
    from scipy.stats import spearmanr
    import itertools
    import numpy as np
    rankings=read_seed_frames(RESULTS_DIR/"explainability_feature_rankings.csv")
    summary=read_seed_frames(RESULTS_DIR/"explainability_summary.csv")
    rows=[];pairs=[]
    for group,g in rankings.groupby("prediction_group"):
        seed_rankings={seed:part.set_index("feature")["mean_abs_attribution"] for seed,part in g.groupby("seed")}
        for (a,va),(b,vb) in itertools.combinations(seed_rankings.items(),2):
            shared=va.index.intersection(vb.index)
            rho=float(spearmanr(va[shared],vb[shared]).statistic) if len(shared)>1 else np.nan
            topa=set(va.nlargest(3).index);topb=set(vb.nlargest(3).index)
            pairs.append({"prediction_group":group,"seed_a":a,"seed_b":b,"spearman":rho,
              "top3_jaccard":len(topa&topb)/len(topa|topb) if topa|topb else np.nan})
        mean=g.groupby("feature").mean_abs_attribution.mean().sort_values(ascending=False)
        group_pairs=[r for r in pairs if r["prediction_group"]==group]
        top=list(mean.index[:3])+[None]*3
        rows.append({"prediction_group":group,"n_seeds":g.seed.nunique(),
          "n":int(summary[summary.prediction_group==group]["n"].sum()),"top_feature_1":top[0],"top_feature_2":top[1],"top_feature_3":top[2],
          "rank_stability":float(np.nanmean([r["spearman"] for r in group_pairs])) if group_pairs else np.nan,
          "top3_jaccard":float(np.mean([r["top3_jaccard"] for r in group_pairs])) if group_pairs else np.nan,
          "stability_definition":"cross-seed Spearman over common features; top-3 Jaccard"})
    for group in ("TP","FP","TN","FN"):
        if group not in {r["prediction_group"] for r in rows}: rows.append({"prediction_group":group,"n":0,"n_seeds":0,"note":"No examples; stability not estimable"})
    pd.DataFrame(rows).to_csv(RESULTS_DIR/"explainability_summary.csv",index=False)
    pd.DataFrame(pairs).to_csv(RESULTS_DIR/"explainability_cross_seed_stability.csv",index=False)
    summarize_frame(rankings,["prediction_group","feature"],["mean_abs_attribution"]).to_csv(RESULTS_DIR/"explainability_feature_rankings.csv",index=False)
    interventions=read_seed_frames(RESULTS_DIR/"feature_intervention_results.csv")
    metrics=["precision","recall","f1","auprc","fpr","mean_probability_change"]
    for metric in ("f1","auprc","fpr"):
        original=interventions[interventions.intervention=="Original"].set_index("seed")[metric]
        interventions["delta_"+metric]=interventions[metric]-interventions.seed.map(original)
        metrics.append("delta_"+metric)
    summarize_frame(interventions,["intervention"],metrics).to_csv(RESULTS_DIR/"feature_intervention_results.csv",index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="42,7,123,1,2024")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-per-category", type=int, default=150)
    args = parser.parse_args()
    for seed in ([args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]):
        run_explainability(seed=seed, csv_path=args.csv_path, epochs=args.epochs,n_per_category=args.n_per_category)
    aggregate_explanations()


if __name__ == "__main__":
    main()
