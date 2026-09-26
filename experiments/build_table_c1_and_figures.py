"""
Issue #4 — Tables C1-C2 and Figures C1-C4, built from the saved ablation
CSVs/JSONs (never manually typed, same principle as Issues #2/#3's
table/figure scripts).

IMPORTANT — this script is the authoritative place ΔF1/ΔAUPRC/ΔFPR are
computed, NOT the raw CSV columns written during the run. See
src/experiments/ablation_runner.py's run_condition_matrix_trackb
docstring: if a run was split across multiple `--conditions` calls for
checkpointing, the CSV's own delta columns can be blank for rows saved
before Condition C existed. This script recomputes every delta fresh by
joining each row against its own (dataset, split_protocol, seed)'s
C_full_model row, so the final tables/figures are correct regardless of
how the run was split.

Requires experiments/results/q1/ablations/*.csv to exist (run
`python -m src.experiments.ablation_runner --experiment c1_trackb` etc.
first, including at least one seed with Condition C run).

Usage:
    python -m experiments.build_table_c1_and_figures
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results/q1/ablations")

TRACKB_CSV = RESULTS_DIR / "scadanet_trackb_topology_ablation.csv"
TEMPORAL_CSV = RESULTS_DIR / "scadanet_temporal_topology_ablation.csv"
FEATURE_CSV = RESULTS_DIR / "feature_counterfactuals.csv"
REWIRING_JSON = RESULTS_DIR / "rewiring_metadata.json"

CONDITION_ORDER = ["A_content_only", "B_topology_only", "C_full_model",
                   "D_random_topology", "E_degree_preserving_rewiring", "F_no_message_passing"]
CONDITION_DISPLAY = {
    "A_content_only": "Content only", "B_topology_only": "Topology only",
    "C_full_model": "Full GraphSAGE", "D_random_topology": "Random topology",
    "E_degree_preserving_rewiring": "Degree-preserved rewiring",
    "F_no_message_passing": "No message passing",
}


def _attach_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Authoritative delta computation — see module docstring. Joins each
    row against the C_full_model row for the SAME (dataset,
    split_protocol, conv_type, seed), then averages over seeds. A row
    whose seed has no matching Condition C result gets NaN deltas rather
    than silently being dropped or compared against the wrong seed."""
    key = ["dataset", "split_protocol", "conv_type", "seed"]
    full = df[df["condition"] == "C_full_model"].set_index(key)

    def _delta(row, col):
        idx = tuple(row[k] for k in key)
        if idx not in full.index:
            return np.nan
        return row[col] - full.loc[idx, col]

    df = df.copy()
    df["delta_f1"] = df.apply(lambda r: _delta(r, "f1"), axis=1)
    df["delta_auprc"] = df.apply(lambda r: _delta(r, "auprc") if pd.notna(r.get("auprc")) else np.nan, axis=1)
    df["delta_fpr"] = df.apply(lambda r: _delta(r, "fpr"), axis=1)
    return df


def _mean_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["precision", "recall", "f1", "auprc", "fpr", "delta_f1", "delta_auprc", "delta_fpr"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(["dataset", "split_protocol", "conv_type", "condition"], dropna=False)[metric_cols].mean().reset_index()


# ---------------------------------------------------------------------
# Table C1 — topology/content ablation
# ---------------------------------------------------------------------

def build_table_c1(csv_path: Path, label: str):
    if not csv_path.exists():
        print(f"Table C1 ({label}) skipped: {csv_path} not found.")
        return None
    df = pd.read_csv(csv_path)
    df = _attach_deltas(df)
    agg = _mean_over_seeds(df)
    agg["condition"] = pd.Categorical(agg["condition"], categories=CONDITION_ORDER, ordered=True)
    agg = agg.sort_values(["conv_type", "condition"])
    agg["condition_display"] = agg["condition"].map(CONDITION_DISPLAY)

    content_flag = {"A_content_only": "✓", "B_topology_only": "✗", "C_full_model": "✓",
                     "D_random_topology": "✓", "E_degree_preserving_rewiring": "✓", "F_no_message_passing": "✓"}
    topology_flag = {"A_content_only": "✗", "B_topology_only": "✓", "C_full_model": "✓",
                      "D_random_topology": "✗ random", "E_degree_preserving_rewiring": "✗ rewired",
                      "F_no_message_passing": "partial"}
    mp_flag = {"A_content_only": "✗", "B_topology_only": "✓", "C_full_model": "✓",
               "D_random_topology": "✓", "E_degree_preserving_rewiring": "✓", "F_no_message_passing": "✗"}

    out = pd.DataFrame({
        "Condition": agg["condition_display"],
        "Content features": agg["condition"].map(content_flag),
        "True topology": agg["condition"].map(topology_flag),
        "Message passing": agg["condition"].map(mp_flag),
        "F1": agg["f1"], "AUPRC": agg["auprc"], "FPR": agg["fpr"], "ΔF1": agg["delta_f1"],
    })

    csv_out = csv_path.parent / f"table_c1_{label}.csv"
    md_out = csv_path.parent / f"table_c1_{label}.md"
    out.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write(f"# Table C1 — Topology/content ablation ({label})\n\n")
        f.write(out.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"wrote {csv_out}\nwrote {md_out}")
    return out


# ---------------------------------------------------------------------
# Table C2 — feature intervention results
# ---------------------------------------------------------------------

def build_table_c2():
    if not FEATURE_CSV.exists():
        print(f"Table C2 skipped: {FEATURE_CSV} not found.")
        return None
    df = pd.read_csv(FEATURE_CSV)
    metric_cols = [c for c in ["f1", "auprc", "fpr", "delta_f1", "delta_auprc", "delta_fpr"] if c in df.columns]
    agg = df.groupby(["removed_or_permuted_feature"], dropna=False)[metric_cols].mean().reset_index()

    order = ["none", "Protocol_TCP", "Tcp_flags_reset_Set", "frame_len", "top-3 jointly", "all content shuffled"]
    agg["removed_or_permuted_feature"] = pd.Categorical(
        agg["removed_or_permuted_feature"], categories=order, ordered=True)
    agg = agg.sort_values("removed_or_permuted_feature")

    out = agg.rename(columns={"removed_or_permuted_feature": "Removed/permuted feature",
                               "f1": "F1", "auprc": "AUPRC", "fpr": "FPR",
                               "delta_f1": "ΔF1", "delta_auprc": "ΔAUPRC", "delta_fpr": "ΔFPR"})

    csv_out = RESULTS_DIR / "table_c2_feature_interventions.csv"
    md_out = RESULTS_DIR / "table_c2_feature_interventions.md"
    out.to_csv(csv_out, index=False)
    skipped = df[df.get("skipped_reason").notna()] if "skipped_reason" in df.columns else pd.DataFrame()
    with open(md_out, "w") as f:
        f.write("# Table C2 — Feature intervention results\n\n")
        f.write(out.round(4).to_markdown(index=False))
        f.write("\n")
        if not skipped.empty:
            f.write("\n**Skipped** (no matching column found in this dataset's one-hot "
                     "feature names):\n\n")
            for _, row in skipped.drop_duplicates("removed_or_permuted_feature").iterrows():
                f.write(f"- `{row['removed_or_permuted_feature']}`: {row['skipped_reason']}\n")
    print(f"wrote {csv_out}\nwrote {md_out}")
    return out


# ---------------------------------------------------------------------
# Figure C1 — where does model performance come from?
# ---------------------------------------------------------------------

def build_figure_c1(csv_path: Path, label: str):
    if not csv_path.exists():
        print(f"Figure C1 ({label}) skipped: {csv_path} not found.")
        return
    df = _attach_deltas(pd.read_csv(csv_path))
    agg = _mean_over_seeds(df)
    agg["condition"] = pd.Categorical(agg["condition"], categories=CONDITION_ORDER, ordered=True)
    agg = agg.sort_values("condition")

    labels = [CONDITION_DISPLAY[c] for c in agg["condition"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, agg["f1"], width, label="F1", color="#4C72B0")
    ax.bar(x + width / 2, agg["auprc"], width, label="AUPRC", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Figure C1 — Where does model performance come from? ({label})")
    ax.legend()
    fig.tight_layout()

    out_path = csv_path.parent / f"figure_c1_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure C2 — performance drop caused by destroying true topology
# ---------------------------------------------------------------------

def build_figure_c2(csv_path: Path, label: str):
    if not csv_path.exists():
        print(f"Figure C2 ({label}) skipped: {csv_path} not found.")
        return
    df = _attach_deltas(pd.read_csv(csv_path))
    agg = _mean_over_seeds(df)
    agg = agg[agg["condition"] != "C_full_model"]
    agg["condition"] = pd.Categorical(
        agg["condition"], categories=[c for c in CONDITION_ORDER if c != "C_full_model"], ordered=True)
    agg = agg.sort_values("condition")

    labels = [CONDITION_DISPLAY[c] for c in agg["condition"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, agg["delta_f1"], width, label="ΔF1", color="#C44E52")
    ax.bar(x + width / 2, agg["delta_auprc"], width, label="ΔAUPRC", color="#8172B2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Δ relative to Condition C (full model)")
    ax.set_title(f"Figure C2 — Performance drop from destroying true topology ({label})")
    ax.legend()
    fig.tight_layout()

    out_path = csv_path.parent / f"figure_c2_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure C3 — per-attack effect of topology (needs per-condition JSONs
# for per_attack_recall; the CSV doesn't carry it)
# ---------------------------------------------------------------------

def build_figure_c3(json_subdir: Path, label: str,
                     conditions_to_show=("C_full_model", "D_random_topology", "A_content_only")):
    if not json_subdir.exists():
        print(f"Figure C3 ({label}) skipped: {json_subdir} not found.")
        return

    per_condition = {}
    for jf in sorted(json_subdir.glob("*.json")):
        with open(jf) as f:
            r = json.load(f)
        cond = r.get("condition")
        if cond in conditions_to_show:
            per_condition.setdefault(cond, r.get("per_attack_recall", {}))

    if not per_condition:
        print(f"Figure C3 ({label}) skipped: none of {conditions_to_show} found in saved JSONs.")
        return

    conds_present = [c for c in conditions_to_show if c in per_condition]
    all_labels = sorted(set().union(*[set(d.keys()) for d in per_condition.values()]))

    matrix = np.full((len(all_labels), len(conds_present)), np.nan)
    for j, cond in enumerate(conds_present):
        for i, atk in enumerate(all_labels):
            v = per_condition[cond].get(atk)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(conds_present)), max(4, 0.4 * len(all_labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(conds_present)))
    ax.set_xticklabels([CONDITION_DISPLAY[c] for c in conds_present], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=8)
    for i in range(len(all_labels)):
        for j in range(len(conds_present)):
            val = matrix[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                     color="black" if np.isnan(val) or val > 0.5 else "white")
    ax.set_title(f"Figure C3 — Per-attack effect of topology ({label})")
    fig.colorbar(im, ax=ax, label="Recall")
    fig.tight_layout()

    out_path = RESULTS_DIR / f"figure_c3_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure C4 — degree distribution validation (original vs degree-
# preserved rewired), from rewiring_metadata.json
# ---------------------------------------------------------------------

def build_figure_c4():
    if not REWIRING_JSON.exists():
        print(f"Figure C4 skipped: {REWIRING_JSON} not found.")
        return
    entries = json.loads(REWIRING_JSON.read_text())
    deg_entries = [e for e in entries if "degree_preserving" in e.get("condition", "")]
    if not deg_entries:
        print("Figure C4 skipped: no degree-preserving-rewiring entries in rewiring_metadata.json.")
        return
    e = deg_entries[-1]  # most recent

    orig, rewired = e["original"], e["rewired"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["min", "mean", "max"]
    orig_vals = [orig["degree_min"], orig["mean_degree"], orig["degree_max"]]
    rewired_vals = [rewired["degree_min"], rewired["mean_degree"], rewired["degree_max"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, orig_vals, width, label="Original", color="#4C72B0")
    ax.bar(x + width / 2, rewired_vals, width, label="Degree-preserved rewired", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Degree")
    ax.set_title(f"Figure C4 — Degree distribution validation\n"
                 f"(seed {e['seed']}, swap_status: {e.get('swap_status')})")
    ax.legend()
    fig.tight_layout()

    out_path = RESULTS_DIR / "figure_c4_degree_validation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} (min/mean/max match exactly = swap preserved degree sequence correctly)")


def main():
    if not (TRACKB_CSV.exists() or TEMPORAL_CSV.exists()):
        raise SystemExit(
            f"No ablation result CSVs found in {RESULTS_DIR}. Run "
            f"`python -m src.experiments.ablation_runner --experiment c1_trackb` first."
        )
    build_table_c1(TRACKB_CSV, "trackb")
    build_table_c1(TEMPORAL_CSV, "temporal")
    build_table_c2()
    build_figure_c1(TRACKB_CSV, "trackb")
    build_figure_c1(TEMPORAL_CSV, "temporal")
    build_figure_c2(TRACKB_CSV, "trackb")
    build_figure_c2(TEMPORAL_CSV, "temporal")
    build_figure_c3(RESULTS_DIR / "c1_trackb", "trackb")
    build_figure_c3(RESULTS_DIR / "c1_temporal", "temporal")
    build_figure_c4()


if __name__ == "__main__":
    main()
