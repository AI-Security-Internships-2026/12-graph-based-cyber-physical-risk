"""Tables H1-H3 and Figures H1-H3, built directly from the saved CSVs in
experiments/results/q1/host_normalization/ (never manually typed values — same principle
as build_table_t_and_figures.py).

Requires `python -m src.experiments.host_norm_runner --protocol both` to
have been run first (Track B alone is enough for H1/H2/Figure H1-H3;
Table H1 will just show its temporal rows blank if only one protocol
was run).

Usage:
    python -m experiments.build_table_h_and_figures

Writes into experiments/results/q1/host_normalization/:
    table_h1_mitigation_comparison.csv / .md
    table_h2_fp_concentration_by_host.csv / .md
    table_h3_attack_recall_before_after.csv / .md
    figure_h1_fp_by_host.png
    figure_h2_precision_fpr_tradeoff.png
    figure_h3_distribution_before_after.png
"""
from pathlib import Path
from src.evaluation.aggregation import aggregate_file,read_seed_frames,summarize_frame

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.host_normalization import KNOWN_PROBLEMATIC_HOSTS

RESULTS_DIR = Path("experiments/results/q1/host_normalization")

TRACKB_SUMMARY_CSV = RESULTS_DIR / "trackb_host_normalization_summary.csv"
TRACKB_FP_CSV = RESULTS_DIR / "trackb_false_positives_by_host.csv"
TRACKB_PAR_CSV = RESULTS_DIR / "trackb_per_attack_recall.csv"
TEMPORAL_SUMMARY_CSV = RESULTS_DIR / "temporal_host_normalization_summary.csv"

CONDITION_ORDER = [
    "H1_original_default_threshold", "H2_original_calibrated_threshold",
    "H3_host_normalized_default_threshold", "H4_host_normalized_calibrated_threshold",
]
CONDITION_LABELS = {
    "H1_original_default_threshold": "Original, threshold 0.50",
    "H2_original_calibrated_threshold": "Original + calibrated threshold",
    "H3_host_normalized_default_threshold": "Host-normalized",
    "H4_host_normalized_calibrated_threshold": "Host-normalized + calibrated threshold",
}


def _save_md(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"[build_table_h] wrote {path}")


# ---------------------------------------------------------------------
# Table H1 — mitigation comparison.
# ---------------------------------------------------------------------

def _build_mitigation_table(summary_csv: Path, out_stem: str):
    if not summary_csv.exists():
        print(f"{out_stem} skipped: {summary_csv} not found.")
        return None
    s = pd.read_csv(summary_csv).set_index("condition")
    rows = []
    for cond in CONDITION_ORDER:
        if cond not in s.index:
            continue
        r = s.loc[cond]
        rows.append({
            "Condition": CONDITION_LABELS[cond], "Precision": r["precision"],
            "Recall": r["recall"], "F1": r["f1"], "AUPRC": r["auprc"],
            "Normal FPR": r["fpr"], "Total FP": r["total_fp"], "n_seeds":r.get("n_seeds"),
            **{k:r[k] for k in r.index if any(k.endswith(x) for x in ("_std","_ci_95_low","_ci_95_high"))},
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{out_stem}.csv", index=False)
    _save_md(df, RESULTS_DIR / f"{out_stem}.md")
    return df


def build_table_h1():
    return _build_mitigation_table(TRACKB_SUMMARY_CSV, "table_h1_mitigation_comparison")


def build_table_h1_temporal():
    """Same schema as Table H1, on the strict SCADANet temporal protocol — not one of the
    issue's named tables by itself, but its own numbers are required and this is the
    only place they land in table form.
    """
    return _build_mitigation_table(TEMPORAL_SUMMARY_CSV, "table_h1_temporal_mitigation_comparison")


# ---------------------------------------------------------------------
# Table H2 — false-positive concentration by host
# ---------------------------------------------------------------------

def build_table_h2():
    if not TRACKB_FP_CSV.exists():
        print(f"Table H2 skipped: {TRACKB_FP_CSV} not found.")
        return None
    fp = pd.read_csv(TRACKB_FP_CSV)
    fp = fp[fp["role"] == "source"]
    if fp.empty:
        print("Table H2 skipped: no source-host FP rows.")
        return None

    pivot = fp.pivot_table(index="host", columns="condition", values="fp_count",
                            aggfunc="sum", fill_value=0)
    for cond in CONDITION_ORDER:
        if cond not in pivot.columns:
            pivot[cond] = 0

    # KNOWN_PROBLEMATIC_HOSTS (module-level, edit it there for a
    # different dataset build than the default two IPs) — falls back to
    # whichever two hosts actually generated FPs in THIS run if none of
    # the configured IPs appear in the data at all.
    tracked_hosts = [h for h in KNOWN_PROBLEMATIC_HOSTS if h in pivot.index]
    if not tracked_hosts:
        tracked_hosts = [h for h in pivot.index
                          if pivot.loc[h, CONDITION_ORDER].sum() > 0][:2]

    rows = []
    other_total = pivot.copy()
    for host in tracked_hosts:
        if host not in pivot.index:
            rows.append({"Host": host, **{CONDITION_LABELS[c]: 0 for c in CONDITION_ORDER}})
            continue
        rows.append({"Host": host, **{CONDITION_LABELS[c]: float(pivot.loc[host, c]) for c in CONDITION_ORDER}})
        other_total = other_total.drop(index=host)
    rows.append({"Host": "other hosts",
                  **{CONDITION_LABELS[c]: float(other_total[c].sum()) for c in CONDITION_ORDER}})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table_h2_fp_concentration_by_host.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_h2_fp_concentration_by_host.md")
    return df


# ---------------------------------------------------------------------
# Table H3 — attack recall before/after mitigation
# ---------------------------------------------------------------------

def build_table_h3():
    if not TRACKB_PAR_CSV.exists():
        print(f"Table H3 skipped: {TRACKB_PAR_CSV} not found.")
        return None
    par = pd.read_csv(TRACKB_PAR_CSV)
    pivot = par.pivot_table(index="attack_type", columns="condition", values="recall", aggfunc="first")
    if "H1_original_default_threshold" not in pivot.columns or "H4_host_normalized_calibrated_threshold" not in pivot.columns:
        print("Table H3 skipped: required conditions not present in per-attack CSV.")
        return None
    df = pd.DataFrame({
        "Attack type": pivot.index,
        "Original recall": pivot.get("H1_original_default_threshold"),
        "Host-normalized recall": pivot.get("H3_host_normalized_default_threshold"),
        "Combined recall": pivot.get("H4_host_normalized_calibrated_threshold"),
    }).reset_index(drop=True)
    df["Delta"] = df["Combined recall"] - df["Original recall"]
    raw=read_seed_frames(TRACKB_PAR_CSV)
    paired=raw.pivot(index=["seed","attack_type"],columns="condition",values="recall").reset_index()
    paired["delta_recall"]=paired["H4_host_normalized_calibrated_threshold"]-paired["H1_original_default_threshold"]
    uncertainty=summarize_frame(paired,["attack_type"],["delta_recall"])
    df=df.merge(uncertainty.rename(columns={"attack_type":"Attack type"}),on="Attack type",how="left")
    df.to_csv(RESULTS_DIR / "table_h3_attack_recall_before_after.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_h3_attack_recall_before_after.md")
    return df


# ---------------------------------------------------------------------
# Figure H1 — false positives by host, top hosts across all 4 conditions
# ---------------------------------------------------------------------

def build_figure_h1(top_n: int = 8):
    if not TRACKB_FP_CSV.exists():
        print(f"Figure H1 skipped: {TRACKB_FP_CSV} not found.")
        return
    fp = pd.read_csv(TRACKB_FP_CSV)
    fp = fp[fp["role"] == "source"]
    if fp.empty:
        print("Figure H1 skipped: no source-host FP rows.")
        return
    totals = fp.groupby("host")["fp_count"].sum().sort_values(ascending=False)
    top_hosts = totals.head(top_n).index.tolist()

    pivot = fp[fp["host"].isin(top_hosts)].pivot_table(
        index="host", columns="condition", values="fp_count", aggfunc="sum", fill_value=0
    ).reindex(top_hosts)
    conds = [c for c in CONDITION_ORDER if c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(top_hosts))))
    y = np.arange(len(top_hosts))
    width = 0.8 / max(len(conds), 1)
    for i, cond in enumerate(conds):
        ax.barh(y + i * width, pivot[cond], height=width, label=CONDITION_LABELS[cond])
    ax.set_yticks(y + width * (len(conds) - 1) / 2)
    ax.set_yticklabels(top_hosts)
    ax.set_xlabel("False positive count")
    ax.set_title("Figure H1 — False positives by host, top FP-generating hosts")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = RESULTS_DIR / "figure_h1_fp_by_host.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_h] wrote {out}")


# ---------------------------------------------------------------------
# Figure H2 — precision/FPR trade-off across the 4 conditions
# ---------------------------------------------------------------------

def build_figure_h2():
    if not TRACKB_SUMMARY_CSV.exists():
        print(f"Figure H2 skipped: {TRACKB_SUMMARY_CSV} not found.")
        return
    s = pd.read_csv(TRACKB_SUMMARY_CSV).set_index("condition")

    fig, ax = plt.subplots(figsize=(6, 5))
    for cond in CONDITION_ORDER:
        if cond not in s.index:
            continue
        r = s.loc[cond]
        ax.scatter(r["fpr"], r["recall"], s=80)
        ax.annotate(CONDITION_LABELS[cond], (r["fpr"], r["recall"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("FPR")
    ax.set_ylabel("Recall")
    ax.set_title("Figure H2 — FPR/Recall trade-off across mitigations\n"
                  "(higher recall and lower FPR are preferred)")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_h2_precision_fpr_tradeoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_h] wrote {out}")


# ---------------------------------------------------------------------
# Figure H3 — distribution before/after host normalization
# ---------------------------------------------------------------------

def save_h3_distribution(df,src_col,feature_col,values,problematic_hosts,attack_mask,seed):
    if feature_col not in df:return
    raw=pd.to_numeric(df[feature_col],errors="coerce").to_numpy()
    normal=~np.asarray(attack_mask,dtype=bool);problem=df[src_col].astype(str).isin(problematic_hosts).to_numpy()
    groups={"Normal: problematic hosts":normal&problem,"Normal: other hosts":normal&~problem,"Attack":~normal}
    rows=[]
    # Fixed bins make distributions comparable across seeds; signed log handles negative normalized values.
    bins=np.arange(-12,12.25,0.25)
    for stage,arr in [("Raw",raw),("Host normalized",np.asarray(values))]:
        z=np.sign(arr)*np.log10(1+np.abs(arr))
        for group,mask in groups.items():
            zgroup=z[mask & np.isfinite(z)];counts,_=np.histogram(zgroup,bins=bins)
            for lo,hi,count in zip(bins[:-1],bins[1:],counts):
                rows.append({"seed":seed,"stage":stage,"group":group,"bin_left":lo,"bin_right":hi,
                  "count":int(count),"n_group":len(zgroup),"outside_bins":int(((zgroup<bins[0])|(zgroup>bins[-1])).sum())})
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR/f"h3_distribution_seed{seed}.csv",index=False)

def build_figure_h3():
    df=read_seed_frames(RESULTS_DIR/"h3_distribution.csv")
    agg=df.groupby(["stage","group","bin_left","bin_right"],as_index=False)[["count","n_group"]].mean()
    agg.to_csv(RESULTS_DIR/"h3_distribution.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,stage in zip(axes,["Raw","Host normalized"]):
        for group,g in agg[agg.stage==stage].groupby("group"):
            if g.n_group.iloc[0]>0:ax.step(g.bin_left,g["count"]/g.n_group,where="post",label=group)
        ax.set_title(stage+" frame_len");ax.set_xlabel("sign(x) × log10(1 + |x|)");ax.set_ylabel("Fraction per bin")
        ax.legend(fontsize=7)
        nonzero=agg[(agg.stage==stage)&(agg["count"]>0)]
        if len(nonzero):ax.set_xlim(nonzero.bin_left.min()-.25,nonzero.bin_right.max()+.25)
    fig.suptitle("Figure H3 — Before and after host normalization; mean across seeds")
    fig.tight_layout();fig.savefig(RESULTS_DIR/"figure_h3_distribution_frame_len.png",dpi=150);plt.close(fig)


def plot_figure_h3(df: pd.DataFrame, src_col: str, feature_col: str,
                    host_normalized_values, problematic_hosts, attack_mask,
                    out_path: Path = None) -> Path:
    """Direct-call variant (see `build_figure_h3` docstring for why this isn't wired into
    `main()`). `host_normalized_values` is the output column for `feature_col` from
    `host_normalized_matrix`, aligned to `df`'s row order.
    """
    is_problematic = df[src_col].astype(str).isin(problematic_hosts).to_numpy()
    is_normal = ~np.asarray(attack_mask, dtype=bool)
    raw = pd.to_numeric(df[feature_col], errors="coerce").to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    groups = {
        "Normal, problematic host": is_normal & is_problematic,
        "Normal, other hosts": is_normal & ~is_problematic,
        "Attack traffic": ~is_normal,
    }
    for label, mask in groups.items():
        if mask.sum() == 0:
            continue
        axes[0].hist(raw[mask], bins=30, alpha=0.5, label=label, density=True)
        axes[1].hist(np.asarray(host_normalized_values)[mask], bins=30, alpha=0.5,
                      label=label, density=True)
    axes[0].set_title(f"Before: raw {feature_col} (global scale)")
    axes[1].set_title(f"After: host-normalized {feature_col}")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Figure H3 — Distribution before/after host normalization")
    fig.tight_layout()
    out_path = out_path or (RESULTS_DIR / f"figure_h3_distribution_{feature_col}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[build_table_h] wrote {out_path}")
    return out_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for protocol in ("trackb","temporal"):
        aggregate_file(RESULTS_DIR/f"{protocol}_host_normalization_summary.csv",["protocol","condition"])
        aggregate_file(RESULTS_DIR/f"{protocol}_per_attack_recall.csv",["protocol","condition","attack_type"])
        fp_path=RESULTS_DIR/f"{protocol}_false_positives_by_host.csv"
        raw=read_seed_frames(fp_path)
        if len(raw):
            seeds=read_seed_frames(RESULTS_DIR/f"{protocol}_host_normalization_summary.csv").seed.unique()
            groups=raw[["protocol","condition","host","role"]].drop_duplicates()
            complete=groups.merge(pd.DataFrame({"seed":seeds}),how="cross").merge(raw,how="left")
            complete["fp_count"]=complete.fp_count.fillna(0)
            summarize_frame(complete,["protocol","condition","host","role"],["fp_count"]).to_csv(fp_path,index=False)
    build_table_h1()
    build_table_h1_temporal()
    build_table_h2()
    build_table_h3()
    build_figure_h1()
    build_figure_h2()
    build_figure_h3()


if __name__ == "__main__":
    main()
