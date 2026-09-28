"""Tables S1-S5 and Figures S1-S5, built directly from the saved CSVs/JSON in
experiments/results/q1/robustness/.

Requires, at minimum:
    python -m src.experiments.robustness_runner --part all --seeds 42,7,123
    python -m src.experiments.explainability_runner --seed 42

Usage:
    python -m experiments.build_table_s_and_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results/q1/robustness")
TEMPORAL_DIR = Path("experiments/results/q1/temporal")
HOSTNORM_DIR = Path("experiments/results/q1/host_normalization")

MULTI_SEED_CSV = RESULTS_DIR / "multi_seed_results.csv"
TABLE_S1_CSV = RESULTS_DIR / "table_s1_robustness_summary.csv"
PAIRED_STATS_CSV = RESULTS_DIR / "paired_statistics.csv"
OPERATIONAL_CSV = RESULTS_DIR / "operational_metrics.csv"
EXPLAIN_SUMMARY_CSV = RESULTS_DIR / "explainability_summary.csv"
EXPLAIN_RANKINGS_CSV = RESULTS_DIR / "explainability_feature_rankings.csv"
INTERVENTION_CSV = RESULTS_DIR / "feature_intervention_results.csv"


def _save_md(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"[build_table_s] wrote {path}")


# ---------------------------------------------------------------------
# Table S1 — reformatted to the issue's exact display columns (wide
# per-metric columns from robustness_runner.summarize_multi_seed are the
# authoritative data, kept in the .csv it already wrote; this builds the
# "mean ± std" / single "95% CI" display shape the issue's Table S1
# actually specifies, matching how Table S2/S3 are formatted below —
# S1 was left as a raw re-export in an earlier pass, inconsistent with
# S2/S3's formatting; fixed to match).
# ---------------------------------------------------------------------

def _fmt_mean_std(mean, std):
    if pd.isna(mean):
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f}"


def build_table_s1():
    if not TABLE_S1_CSV.exists():
        print(f"Table S1 skipped: {TABLE_S1_CSV} not found "
              f"(run: python -m src.experiments.robustness_runner --part a ...)")
        return None
    s = pd.read_csv(TABLE_S1_CSV)
    rows = []
    for _, r in s.iterrows():
        f1_ci = (f"[{r['f1_ci_95_low']:.4f}, {r['f1_ci_95_high']:.4f}]"
                  if "f1_ci_95_low" in r and pd.notna(r.get("f1_ci_95_low")) else "N/A")
        rows.append({
            "Model / condition": r["condition"], "n seeds": r.get("n_seeds"),
            "F1 mean ± std": _fmt_mean_std(r.get("f1_mean"), r.get("f1_std")),
            "95% CI (F1)": f1_ci,
            "AUPRC mean ± std": _fmt_mean_std(r.get("auprc_mean"), r.get("auprc_std")),
            "FPR mean ± std": _fmt_mean_std(r.get("fpr_mean"), r.get("fpr_std")),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table_s1_display.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_s1_robustness_summary.md")
    return df


# ---------------------------------------------------------------------
# Table S2 — paired statistics, reformatted to the issue's exact columns
# ---------------------------------------------------------------------

def build_table_s2():
    if not PAIRED_STATS_CSV.exists():
        print(f"Table S2 skipped: {PAIRED_STATS_CSV} not found "
              f"(run: python -m src.experiments.robustness_runner --part b)")
        return None
    s = pd.read_csv(PAIRED_STATS_CSV)
    rows = []
    for _, r in s.iterrows():
        ci = (f"[{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]"
              if pd.notna(r.get("ci_95_low")) else "N/A")
        rows.append({
            "Comparison": r["comparison"], "Metric": r["metric"],
            "Mean difference": r.get("mean_difference"), "95% CI": ci,
            "Effect size (Cohen's d)": r.get("effect_size_cohens_d"),
            "p-value": r.get("p_value"), "Test used": r.get("test_used"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table_s2_statistical_comparison.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_s2_statistical_comparison.md")
    return df


# ---------------------------------------------------------------------
# Table S3 — operational metrics
# ---------------------------------------------------------------------

def build_table_s3():
    if not OPERATIONAL_CSV.exists():
        print(f"Table S3 skipped: {OPERATIONAL_CSV} not found "
              f"(run: python -m src.experiments.robustness_runner --part c)")
        return None
    s = pd.read_csv(OPERATIONAL_CSV)
    rows = []
    for _, r in s.iterrows():
        rows.append({
            "Model": r["model"], "AUPRC": r.get("auprc"), "FPR": r.get("fpr"),
            "False alarms/hour": r.get("false_alarms_per_hour")
            if pd.notna(r.get("false_alarms_per_hour")) else f"N/A — {r.get('false_alarms_per_hour_note')}",
            "Event recall": r.get("event_recall")
            if pd.notna(r.get("event_recall")) else r.get("event_metrics_note","N/A — verified event labels unavailable"),
            "Time-to-detect (s)": r.get("time_to_detect_mean_s"),
            "ECE": r.get("ece"), "Brier": r.get("brier_score"),
            "Inference latency": r.get("inference_latency_ms"), "n_seeds":r.get("n_seeds"),
            "Parameters":r.get("n_trainable_params"), "Model storage bytes":r.get("model_size_bytes"),
            **{k:r[k] for k in r.index if any(k.endswith(x) for x in ("_std","_ci_95_low","_ci_95_high"))},
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table_s3_operational_metrics.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_s3_operational_metrics.md")
    return df


# ---------------------------------------------------------------------
# Table S4 — explainability stability (already close to final shape)
# ---------------------------------------------------------------------

def build_table_s4():
    if not EXPLAIN_SUMMARY_CSV.exists():
        print(f"Table S4 skipped: {EXPLAIN_SUMMARY_CSV} not found "
              f"(run: python -m src.experiments.explainability_runner)")
        return None
    df = pd.read_csv(EXPLAIN_SUMMARY_CSV)
    df.to_csv(RESULTS_DIR / "table_s4_explainability_stability.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_s4_explainability_stability.md")
    return df


# ---------------------------------------------------------------------
# Table S5 — feature intervention
# ---------------------------------------------------------------------

def build_table_s5():
    if not INTERVENTION_CSV.exists():
        print(f"Table S5 skipped: {INTERVENTION_CSV} not found "
              f"(run: python -m src.experiments.explainability_runner)")
        return None
    df = pd.read_csv(INTERVENTION_CSV)
    df.to_csv(RESULTS_DIR / "table_s5_feature_intervention.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_s5_feature_intervention.md")
    return df


# ---------------------------------------------------------------------
# Figure S1 — F1 distribution across seeds, dot plot (no seaborn
# dependency — plain matplotlib scatter/box, "a standard plot available
# in the existing stack" per the issue)
# ---------------------------------------------------------------------

FIG_S1_CONDITIONS = ["MLP_trackb", "Tree_trackb", "GraphSAGE_trackb", "RandomTopology_trackb",
                      "DegreePreservedTopology_trackb"]
# Tree_trackb is drawn as an extra line alongside MLP.
FIG_S1_LABELS = {"MLP_trackb": "MLP", "Tree_trackb": "Tree (XGBoost)",
                  "GraphSAGE_trackb": "GraphSAGE",
                  "RandomTopology_trackb": "Random topology",
                  "DegreePreservedTopology_trackb": "Degree-preserved topology"}


def build_figure_s1():
    if not MULTI_SEED_CSV.exists():
        print(f"Figure S1 skipped: {MULTI_SEED_CSV} not found.")
        return
    df = pd.read_csv(MULTI_SEED_CSV)
    fig, ax = plt.subplots(figsize=(7, 5))
    present = [c for c in FIG_S1_CONDITIONS if c in df["condition"].unique()]
    if not present:
        print("Figure S1 skipped: none of the required conditions found in multi_seed_results.csv.")
        return
    box_data = [df[df["condition"] == c]["f1"].dropna().to_numpy() for c in present]
    ax.boxplot(box_data, labels=[FIG_S1_LABELS[c] for c in present], showmeans=True)
    for i, data in enumerate(box_data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, alpha=0.6, s=20, color="tab:blue")
    ax.set_ylabel("F1 (Track B)")
    ax.set_title("Figure S1 — F1 distribution across seeds")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_s1_seed_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_s] wrote {out}")


# ---------------------------------------------------------------------
# Figure S2 — evaluation-protocol degradation (Track B -> strict
# temporal), with 95% CI. Only GraphSAGE has both variants built in
# this repo (see robustness_runner.py's own docstring) — this figure
# covers that one model, not "each principal model" as literally
# written, and says so on the plot rather than fabricating temporal
# variants of MLP/content-only that were never run.
# ---------------------------------------------------------------------

def build_figure_s2():
    if not TABLE_S1_CSV.exists():
        print(f"Figure S2 skipped: {TABLE_S1_CSV} not found.")
        return
    s = pd.read_csv(TABLE_S1_CSV).set_index("condition")
    pairs = [("MatchedGraphSAGE_trackb", "GraphSAGE_strict_temporal", "Matched GraphSAGE")]
    present = [(a, b, label) for a, b, label in pairs if a in s.index and b in s.index]
    if not present:
        print("Figure S2 skipped: GraphSAGE_trackb / GraphSAGE_strict_temporal rows not found.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (cond_a, cond_b, label) in enumerate(present):
        for x, cond in [(0, cond_a), (1, cond_b)]:
            row = s.loc[cond]
            mean, lo, hi = row["f1_mean"], row["f1_ci_95_low"], row["f1_ci_95_high"]
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", capsize=5,
                        color=f"C{i}", label=label if x == 0 else None)
        row_a, row_b = s.loc[cond_a], s.loc[cond_b]
        ax.plot([0, 1], [row_a["f1_mean"], row_b["f1_mean"]], color=f"C{i}", alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Track B / non-temporal", "Strict temporal"])
    ax.set_ylabel("F1, mean ± 95% CI")
    ax.set_title("Figure S2 — Evaluation-protocol degradation\n"
                  "(only models with both variants implemented are shown)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "figure_s2_protocol_degradation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_s] wrote {out}")


# ---------------------------------------------------------------------
# Figure S3 — per-attack temporal recall heatmap: static/Track-B vs
# strict temporal vs (optionally) host-normalized strict temporal
# ---------------------------------------------------------------------

def build_figure_s3():
    per_attack_csv = TEMPORAL_DIR / "scadanet_temporal_per_attack.csv"
    if not per_attack_csv.exists():
        print(f"Figure S3 skipped: {per_attack_csv} not found "
              f"(run Issue #5's temporal_runner --experiment scadanet first).")
        return
    per_attack = pd.read_csv(per_attack_csv).set_index("attack_type")
    cols = {"Static / Track B": per_attack["static_trackb_recall"],
            "Strict temporal": per_attack["strict_temporal_recall"]}

    host_par_csv = HOSTNORM_DIR / "temporal_per_attack_recall.csv"
    if host_par_csv.exists():
        host_par = pd.read_csv(host_par_csv)
        h4 = host_par[host_par["condition"] == "H4_host_normalized_calibrated_threshold"]
        if not h4.empty:
            cols["Host-normalized strict temporal"] = h4.set_index("attack_type")["recall"].reindex(
                per_attack.index)

    heat = pd.DataFrame(cols)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(heat))))
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Figure S3 — Per-attack recall across protocols")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_s3_per_attack_recall_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_s] wrote {out}")


# ---------------------------------------------------------------------
# Figure S4 — explainability feature ranking, TP/FP/FN
# ---------------------------------------------------------------------

def build_figure_s4(top_n: int = 5):
    if not EXPLAIN_RANKINGS_CSV.exists():
        print(f"Figure S4 skipped: {EXPLAIN_RANKINGS_CSV} not found.")
        return
    df = pd.read_csv(EXPLAIN_RANKINGS_CSV)
    groups = [g for g in ["TP", "FP", "FN"] if g in df["prediction_group"].unique()]
    if not groups:
        print("Figure S4 skipped: none of TP/FP/FN present in explainability_feature_rankings.csv.")
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), sharey=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        g = df[df["prediction_group"] == group].nlargest(top_n, "mean_abs_attribution")
        ax.barh(g["feature"][::-1], g["mean_abs_attribution"][::-1])
        ax.set_title(group)
        ax.set_xlabel("Mean |attribution|")
    fig.suptitle("Figure S4 — Explainability feature ranking by prediction category")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_s4_explainability_ranking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_s] wrote {out}")


# ---------------------------------------------------------------------
# Figure S5 — explanation intervention effect
# ---------------------------------------------------------------------

def build_figure_s5():
    if not INTERVENTION_CSV.exists():
        print(f"Figure S5 skipped: {INTERVENTION_CSV} not found.")
        return
    df = pd.read_csv(INTERVENTION_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(df["intervention"], df["f1"],yerr=(df["f1_ci_95_high"]-df["f1"]).fillna(0),capsize=3)
    axes[0].set_ylabel("F1")
    axes[0].set_title("F1 after intervention")
    axes[0].tick_params(axis="x", rotation=30, labelsize=7)
    axes[1].bar(df["intervention"], df["mean_probability_change"],yerr=(df["mean_probability_change_ci_95_high"]-df["mean_probability_change"]).fillna(0),capsize=3, color="tab:orange")
    axes[1].set_ylabel("Mean |probability change|")
    axes[1].set_title("Prediction shift after intervention")
    axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("Figure S5 — Explanation intervention effect")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_s5_intervention_effect.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_s] wrote {out}")


def main():
    # Rebuild prerequisite reports from immutable per-seed files; never trust a latest-run canonical.
    from experiments.build_table_t_and_figures import main as rebuild_temporal
    from experiments.build_table_h_and_figures import main as rebuild_hosts
    rebuild_temporal()
    rebuild_hosts()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    build_table_s1()
    build_table_s2()
    build_table_s3()
    build_table_s4()
    build_table_s5()
    build_figure_s1()
    build_figure_s2()
    build_figure_s3()
    build_figure_s4()
    build_figure_s5()


if __name__ == "__main__":
    main()
