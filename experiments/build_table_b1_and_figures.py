"""
Issue #3 — Tables B1-B3 and Figures B1-B3, built directly from the saved
baseline result CSVs/JSONs (never manually typed values — same principle
as build_table_a2_and_figures.py for Issue #2).

Requires experiments/results/q1/baselines/*.csv to already exist (run
`python -m src.experiments.baseline_runner --experiment b1/b2/b3` first).

Usage:
    python -m experiments.build_table_b1_and_figures

Writes into experiments/results/q1/baselines/:
    table_b1_main_comparison.csv / .md
    table_b2_temporal_comparison.csv / .md
    table_b3_computational_cost.csv / .md
    figure_b1_trackb_performance.png
    figure_b2_static_vs_temporal.png
    figure_b3_per_attack_heatmap.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results/q1/baselines")

TRACKB_CSV = RESULTS_DIR / "scadanet_trackb_baselines.csv"
TEMPORAL_CSV = RESULTS_DIR / "scadanet_temporal_baselines.csv"
BATADAL_CSV = RESULTS_DIR / "batadal_temporal_baselines.csv"

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
    "tree": "Tree model",
    "graphsage": "GraphSAGE",
    "gcn": "GCN",
    "gat": "GAT",
}


def _mean_over_seeds(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    """Issue #3 item 7: "save per-seed results rather than only the final
    mean" — the CSVs are per-seed; tables/figures aggregate to a mean
    here, at display time, so the per-seed record is never lost."""
    metric_cols = ["precision", "recall", "f1", "auprc", "auroc", "fpr",
                    "n_trainable_params", "inference_latency_ms_mean", "train_runtime_s"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()


# ---------------------------------------------------------------------
# Table B1 — main baseline comparison (SCADANet Track B, all 6 families)
# ---------------------------------------------------------------------

def build_table_b1():
    if not TRACKB_CSV.exists():
        print(f"Table B1 skipped: {TRACKB_CSV} not found.")
        return None
    df = pd.read_csv(TRACKB_CSV)
    agg = _mean_over_seeds(df, ["dataset", "split_protocol", "model_family"])
    agg["model_family"] = pd.Categorical(
        agg["model_family"],
        categories=["logistic_regression", "tree", "mlp", "gcn", "graphsage", "gat"],
        ordered=True,
    )
    agg = agg.sort_values("model_family")
    agg["model"] = agg["model_family"].map(MODEL_DISPLAY_NAMES)

    out = agg[["dataset", "split_protocol", "model", "precision", "recall", "f1", "auprc", "fpr"]]
    out.columns = ["Dataset", "Protocol", "Model", "Precision", "Recall", "F1", "AUPRC", "FPR"]

    csv_path = RESULTS_DIR / "table_b1_main_comparison.csv"
    out.to_csv(csv_path, index=False)
    md_path = RESULTS_DIR / "table_b1_main_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Table B1 — Main baseline comparison (SCADANet Track B)\n\n")
        f.write(out.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"wrote {csv_path}\nwrote {md_path}")
    return out


# ---------------------------------------------------------------------
# Table B2 — temporal baseline comparison, ΔF1 vs static (Track B)
# ---------------------------------------------------------------------

def build_table_b2():
    if not TEMPORAL_CSV.exists():
        print(f"Table B2 skipped: {TEMPORAL_CSV} not found.")
        return None

    temp_df = pd.read_csv(TEMPORAL_CSV)
    temp_agg = _mean_over_seeds(temp_df, ["dataset", "model_family"])

    static_agg = None
    if TRACKB_CSV.exists():
        static_df = pd.read_csv(TRACKB_CSV)
        static_agg = _mean_over_seeds(static_df, ["dataset", "model_family"]).set_index("model_family")["f1"]

    rows = []
    for _, r in temp_agg.iterrows():
        static_f1 = static_agg.get(r["model_family"]) if static_agg is not None else None
        delta = (r["f1"] - static_f1) if static_f1 is not None else None
        rows.append({
            "Dataset": r["dataset"], "Model": MODEL_DISPLAY_NAMES.get(r["model_family"], r["model_family"]),
            "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"],
            "AUPRC": r.get("auprc"), "FPR": r["fpr"],
            "ΔF1 vs static": delta,
        })
    out = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "table_b2_temporal_comparison.csv"
    out.to_csv(csv_path, index=False)
    md_path = RESULTS_DIR / "table_b2_temporal_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Table B2 — Temporal baseline comparison\n\n"
                "ΔF1 vs static is temporal F1 minus the same model's SCADANet "
                "Track B (static) F1; positive means the model held up better "
                "under the strict temporal protocol.\n\n")
        f.write(out.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"wrote {csv_path}\nwrote {md_path}")
    return out


# ---------------------------------------------------------------------
# Table B3 — computational cost, principal models, across all runs found
# ---------------------------------------------------------------------

def build_table_b3():
    frames = []
    for path, protocol_label in [(TRACKB_CSV, "SCADANet Track B"),
                                  (TEMPORAL_CSV, "SCADANet temporal"),
                                  (BATADAL_CSV, "BATADAL temporal")]:
        if path.exists():
            df = pd.read_csv(path)
            df["protocol_label"] = protocol_label
            frames.append(df)
    if not frames:
        print("Table B3 skipped: no baseline CSVs found.")
        return None

    all_df = pd.concat(frames, ignore_index=True)
    agg = _mean_over_seeds(all_df, ["protocol_label", "dataset", "model"])
    out = agg[["protocol_label", "dataset", "model", "n_trainable_params",
               "train_runtime_s", "inference_latency_ms_mean"]]
    out.columns = ["Protocol", "Dataset", "Model", "Parameters", "Train time (s)", "Inference time (ms)"]

    csv_path = RESULTS_DIR / "table_b3_computational_cost.csv"
    out.to_csv(csv_path, index=False)
    md_path = RESULTS_DIR / "table_b3_computational_cost.md"
    with open(md_path, "w") as f:
        f.write("# Table B3 — Computational cost\n\n"
                "Parameters is blank for the tree baseline (not a meaningful "
                "concept for a tree ensemble — see src/models/baselines.py "
                "TreeBaseline.n_trainable_params). Timings are single-process "
                "CPU wall time on whatever machine produced this file; see "
                "library_versions in the per-model result JSON for the "
                "environment.\n\n")
        f.write(out.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"wrote {csv_path}\nwrote {md_path}")
    return out


# ---------------------------------------------------------------------
# Figure B1 — SCADANet Track B, F1/AUPRC/FPR per model
# ---------------------------------------------------------------------

def build_figure_b1():
    if not TRACKB_CSV.exists():
        print(f"Figure B1 skipped: {TRACKB_CSV} not found.")
        return
    df = pd.read_csv(TRACKB_CSV)
    agg = _mean_over_seeds(df, ["model_family"])
    order = ["logistic_regression", "tree", "mlp", "gcn", "graphsage", "gat"]
    agg["model_family"] = pd.Categorical(agg["model_family"], categories=order, ordered=True)
    agg = agg.sort_values("model_family")

    labels = [MODEL_DISPLAY_NAMES[m] for m in agg["model_family"]]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, agg["f1"], width, label="F1", color="#4C72B0")
    ax.bar(x, agg["auprc"], width, label="AUPRC", color="#55A868")
    ax.bar(x + width, agg["fpr"], width, label="FPR", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure B1 — Model performance under the same protocol\n(SCADANet Track B)")
    ax.legend()
    fig.tight_layout()

    out_path = RESULTS_DIR / "figure_b1_trackb_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure B2 — static (Track B) vs strict temporal F1, principal models
# ---------------------------------------------------------------------

def build_figure_b2():
    if not (TRACKB_CSV.exists() and TEMPORAL_CSV.exists()):
        print("Figure B2 skipped: need both Track B and temporal CSVs.")
        return
    static_df = _mean_over_seeds(pd.read_csv(TRACKB_CSV), ["model_family"]).set_index("model_family")
    temporal_df = _mean_over_seeds(pd.read_csv(TEMPORAL_CSV), ["model_family"]).set_index("model_family")

    principal = [m for m in temporal_df.index if m in static_df.index]
    labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in principal]
    static_f1 = [static_df.loc[m, "f1"] for m in principal]
    temporal_f1 = [temporal_df.loc[m, "f1"] for m in principal]

    x = np.arange(len(principal))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, static_f1, width, label="Static (Track B)", color="#4C72B0")
    ax.bar(x + width / 2, temporal_f1, width, label="Strict temporal", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure B2 — Static vs strict-temporal F1, principal models")
    ax.legend()
    fig.tight_layout()

    out_path = RESULTS_DIR / "figure_b2_static_vs_temporal.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure B3 — per-attack recall heatmap (SCADANet Track B, per-model JSONs)
# ---------------------------------------------------------------------

def build_figure_b3():
    json_dir = RESULTS_DIR / "b1"
    if not json_dir.exists():
        print(f"Figure B3 skipped: {json_dir} not found (run Experiment B1 first).")
        return

    per_model = {}
    for jf in sorted(json_dir.glob("scadanet_track_b_*.json")):
        with open(jf) as f:
            r = json.load(f)
        per_model[r["model"]] = r.get("per_attack_recall", {})

    if not per_model:
        print("Figure B3 skipped: no per-attack recall found in saved JSONs.")
        return

    all_labels = sorted(set().union(*[set(d.keys()) for d in per_model.values()]))
    model_names = list(per_model.keys())

    matrix = np.full((len(all_labels), len(model_names)), np.nan)
    for j, model in enumerate(model_names):
        for i, label in enumerate(all_labels):
            v = per_model[model].get(label)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(model_names)), max(4, 0.4 * len(all_labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=8)

    for i in range(len(all_labels)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                     color="black" if np.isnan(val) or val > 0.5 else "white")

    ax.set_title("Figure B3 — Per-attack recall by model (SCADANet Track B)")
    fig.colorbar(im, ax=ax, label="Recall")
    fig.tight_layout()

    out_path = RESULTS_DIR / "figure_b3_per_attack_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    if not RESULTS_DIR.exists() or not TRACKB_CSV.exists():
        raise SystemExit(
            f"No baseline result CSVs found in {RESULTS_DIR}. Run "
            f"`python -m src.experiments.baseline_runner --experiment b1` "
            f"(and b2/b3) first."
        )
    build_table_b1()
    build_table_b2()
    build_table_b3()
    build_figure_b1()
    build_figure_b2()
    build_figure_b3()


if __name__ == "__main__":
    main()
