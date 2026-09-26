"""
Issue #5 — Tables T1-T3 and Figures T1-T5, built directly from the saved
CSVs in experiments/results/q1/temporal/ (never manually typed values —
same principle as every other build_table_*_and_figures.py in this repo).

Requires `python -m src.experiments.temporal_runner --experiment scadanet`
and `--experiment batadal` to have been run first.

Usage:
    python -m experiments.build_table_t_and_figures

Writes into experiments/results/q1/temporal/:
    table_t1_old_vs_strict.csv / .md
    table_t2_scadanet_per_attack.csv / .md
    table_t3_batadal_split_sensitivity.csv / .md
    figure_t1_performance_over_time.png
    figure_t2_graph_growth.png
    figure_t3_attack_prevalence.png
    figure_t4_static_vs_strict_heatmap.png
    figure_t5_batadal_purge_illustration.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results/q1/temporal")

SUMMARY_CSV = RESULTS_DIR / "scadanet_temporal_summary.csv"
PER_ATTACK_CSV = RESULTS_DIR / "scadanet_temporal_per_attack.csv"
WINDOW_CSV = RESULTS_DIR / "scadanet_temporal_window_metrics.csv"
GROWTH_CSV = RESULTS_DIR / "scadanet_graph_growth.csv"
BATADAL_CSV = RESULTS_DIR / "batadal_purged_temporal_metrics.csv"
BATADAL_AUDIT_JSON = RESULTS_DIR / "batadal_boundary_audit.json"


def _save_md(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"[build_table_t] wrote {path}")


# ---------------------------------------------------------------------
# Table T1 — old vs strict temporal evaluation (SCADANet + BATADAL)
# ---------------------------------------------------------------------

def build_table_t1():
    rows = []
    if SUMMARY_CSV.exists():
        s = pd.read_csv(SUMMARY_CSV)
        name_map = {
            "T1_previous_temporal": "Previous temporal",
            "T2_strict_causal_topology": "Strict causal topology",
            "T3_fixed_training_topology": "Fixed training topology",
        }
        for _, r in s.iterrows():
            rows.append({
                "Dataset": "SCADANet", "Protocol": name_map.get(r["protocol"], r["protocol"]),
                "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"],
                "AUPRC": r["auprc"], "FPR": r["fpr"], "Leakage condition": r["leakage_condition"],
            })
    else:
        print(f"Table T1 (SCADANet half) skipped: {SUMMARY_CSV} not found.")

    if BATADAL_CSV.exists():
        b = pd.read_csv(BATADAL_CSV)
        b70 = b[b["protocol"].isin(["T4_overlapping_temporal", "T5_purged_70_30"])]
        name_map_b = {"T4_overlapping_temporal": "Previous temporal",
                      "T5_purged_70_30": "Purged temporal"}
        for _, r in b70.iterrows():
            leakage = ("overlapping raw rows" if r["protocol"] == "T4_overlapping_temporal"
                       else "none identified")
            rows.append({
                "Dataset": "BATADAL", "Protocol": name_map_b[r["protocol"]],
                "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"],
                "AUPRC": r["auprc"], "FPR": r["fpr"], "Leakage condition": leakage,
            })
    else:
        print(f"Table T1 (BATADAL half) skipped: {BATADAL_CSV} not found.")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table_t1_old_vs_strict.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_t1_old_vs_strict.md")
    return df


# ---------------------------------------------------------------------
# Table T2 — SCADANet temporal performance per attack
# ---------------------------------------------------------------------

def build_table_t2():
    if not PER_ATTACK_CSV.exists():
        print(f"Table T2 skipped: {PER_ATTACK_CSV} not found.")
        return None
    df = pd.read_csv(PER_ATTACK_CSV)
    df = df.rename(columns={
        "attack_type": "Attack type", "train_count": "Train count", "test_count": "Test count",
        "static_trackb_recall": "Static/Track-B recall", "strict_temporal_recall": "Strict temporal recall",
        "delta_recall": "Delta recall",
    })
    df.to_csv(RESULTS_DIR / "table_t2_scadanet_per_attack.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_t2_scadanet_per_attack.md")
    return df


# ---------------------------------------------------------------------
# Table T3 — BATADAL split sensitivity after purging
# ---------------------------------------------------------------------

def build_table_t3():
    if not BATADAL_CSV.exists():
        print(f"Table T3 skipped: {BATADAL_CSV} not found.")
        return None
    b = pd.read_csv(BATADAL_CSV)
    b = b[b["protocol"] != "T4_overlapping_temporal"]
    rename_map = {
        "split": "Split", "n_train_windows": "Train windows",
        "n_purged_windows": "Purged windows", "n_test_windows": "Test windows",
        "precision": "Precision", "recall": "Recall", "f1": "F1", "auprc": "AUPRC",
    }
    cols = ["Split", "Train windows", "Purged windows", "Test windows",
            "Precision", "Recall", "F1", "AUPRC"]
    # Older result files may lack anomalous-window counts.
    if "n_anomalous_train_windows" in b.columns:
        rename_map["n_anomalous_train_windows"] = "Anomalous train windows"
        rename_map["n_anomalous_test_windows"] = "Anomalous test windows"
        cols[2:2] = ["Anomalous train windows", "Anomalous test windows"]
    df = b.rename(columns=rename_map)[cols]
    df.to_csv(RESULTS_DIR / "table_t3_batadal_split_sensitivity.csv", index=False)
    _save_md(df, RESULTS_DIR / "table_t3_batadal_split_sensitivity.md")
    return df


# ---------------------------------------------------------------------
# Figure T1 — SCADANet performance over time
# ---------------------------------------------------------------------

def build_figure_t1():
    if not WINDOW_CSV.exists():
        print(f"Figure T1 skipped: {WINDOW_CSV} not found.")
        return
    w = pd.read_csv(WINDOW_CSV)
    w = w[w["mode"] == "causal"].sort_values("window")
    if w.empty:
        print("Figure T1 skipped: no causal-mode window rows.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w["window"], w["f1"], marker="o", label="F1")
    ax.plot(w["window"], w["recall"], marker="s", label="Recall")
    if "auprc" in w.columns:
        ax.plot(w["window"], w["auprc"], marker="^", label="AUPRC")
    ax.set_xlabel("Chronological test window")
    ax.set_ylabel("Score")
    ax.set_title("Figure T1 — Strict causal-topology performance over time (SCADANet)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "figure_t1_performance_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_t] wrote {out}")


# ---------------------------------------------------------------------
# Figure T2 — graph growth over time
# ---------------------------------------------------------------------

def build_figure_t2(split_window: int = None):
    if not GROWTH_CSV.exists():
        print(f"Figure T2 skipped: {GROWTH_CSV} not found.")
        return
    g = pd.read_csv(GROWTH_CSV)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g["window"], g["cumulative_nodes"], marker="o", label="Cumulative nodes")
    ax2 = ax.twinx()
    ax2.plot(g["window"], g["cumulative_edges"], marker="s", color="tab:orange",
             label="Cumulative IP pairs (edges)")
    if split_window is not None:
        ax.axvline(split_window - 0.5, color="gray", linestyle="--", label="train/test boundary")
    ax.set_xlabel("Window")
    ax.set_ylabel("Cumulative nodes")
    ax2.set_ylabel("Cumulative edges")
    ax.set_title("Figure T2 — Graph growth over time (SCADANet)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_t2_graph_growth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_t] wrote {out}")


# ---------------------------------------------------------------------
# Figure T3 — attack prevalence over time
# ---------------------------------------------------------------------

def build_figure_t3():
    if not WINDOW_CSV.exists():
        print(f"Figure T3 skipped: {WINDOW_CSV} not found.")
        return
    w = pd.read_csv(WINDOW_CSV)
    w = w[w["mode"] == "causal"].sort_values("window")
    if w.empty:
        print("Figure T3 skipped: no causal-mode window rows.")
        return

    counts_by_window = {}
    all_attacks = set()
    for _, row in w.iterrows():
        try:
            counts = json.loads(row["attack_type_counts"]) if pd.notna(row["attack_type_counts"]) else {}
        except (json.JSONDecodeError, TypeError):
            counts = {}
        counts_by_window[row["window"]] = counts
        all_attacks |= set(counts.keys())

    fig, ax = plt.subplots(figsize=(7, 4))
    windows = sorted(counts_by_window.keys())
    for attack in sorted(all_attacks):
        series = [counts_by_window[wnd].get(attack, 0) for wnd in windows]
        style = dict(linewidth=2.5) if attack == "vuln_scan" else dict(linewidth=1)
        ax.plot(windows, series, marker="o", label=attack, **style)
    ax.set_xlabel("Chronological test window")
    ax.set_ylabel("Attack flow count")
    ax.set_title("Figure T3 — Attack prevalence over time (SCADANet, vuln_scan highlighted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "figure_t3_attack_prevalence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_t] wrote {out}")


# ---------------------------------------------------------------------
# Figure T4 — static/Track-B vs strict temporal per-attack recall
# ---------------------------------------------------------------------

def build_figure_t4():
    if not PER_ATTACK_CSV.exists():
        print(f"Figure T4 skipped: {PER_ATTACK_CSV} not found.")
        return
    df = pd.read_csv(PER_ATTACK_CSV)
    if df.empty:
        print("Figure T4 skipped: no per-attack rows.")
        return

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(df))))
    y = np.arange(len(df))
    has_static = df["static_trackb_recall"].notna().any()
    if has_static:
        ax.scatter(df["static_trackb_recall"], y, label="Static/Track-B recall", marker="o")
    ax.scatter(df["strict_temporal_recall"], y, label="Strict temporal recall", marker="x")
    ax.set_yticks(y)
    ax.set_yticklabels(df["attack_type"])
    ax.set_xlabel("Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Figure T4 — Static vs strict-temporal per-attack recall")
    ax.legend()
    if not has_static:
        ax.text(0.5, -0.15,
                "Static/Track-B recall not available in this run (Issue #3 results not found).",
                transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_t4_static_vs_strict_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_t] wrote {out}")


# ---------------------------------------------------------------------
# Figure T5 — BATADAL purge illustration
# ---------------------------------------------------------------------

def build_figure_t5():
    if not BATADAL_AUDIT_JSON.exists():
        print(f"Figure T5 skipped: {BATADAL_AUDIT_JSON} not found.")
        return
    audit = json.loads(BATADAL_AUDIT_JSON.read_text())
    if "T5_purged_70_30" not in audit or "purge_meta" not in audit["T5_purged_70_30"]:
        print("Figure T5 skipped: T5_purged_70_30 purge metadata not found.")
        return
    meta = audit["T5_purged_70_30"]["purge_meta"]
    window_size = meta["window_size"]
    boundary = meta["train_end_row"]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    # Illustrative windows straddling the boundary at stride=6h (documented default).
    stride = 6
    starts = list(range(boundary - 3 * window_size, boundary + 2 * window_size, stride))
    for i, s in enumerate(starts):
        e = s + window_size
        straddles = s < boundary < e
        color = "tab:red" if straddles else ("tab:blue" if e <= boundary else "tab:green")
        ax.plot([s, e], [i, i], color=color, linewidth=4,
                label=("purged (straddles boundary)" if straddles and
                       "purged (straddles boundary)" not in ax.get_legend_handles_labels()[1]
                       else None))
    ax.axvline(boundary, color="black", linestyle="--", label="train/test boundary")
    ax.set_xlabel("Raw row index")
    ax.set_yticks([])
    ax.set_title("Figure T5 — BATADAL purge illustration (red windows dropped)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper left")
    fig.tight_layout()
    out = RESULTS_DIR / "figure_t5_batadal_purge_illustration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[build_table_t] wrote {out}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    build_table_t1()
    build_table_t2()
    build_table_t3()

    split_window = None
    if SUMMARY_CSV.exists():
        s = pd.read_csv(SUMMARY_CSV)
        if "split_window" in s.columns and s["split_window"].notna().any():
            split_window = int(s["split_window"].dropna().iloc[0])
        else:
            print("Figure T2 boundary line skipped: split_window column not "
                  "found in scadanet_temporal_summary.csv (regenerate it with "
                  "the current temporal_runner.py to get this).")
    build_figure_t1()
    build_figure_t2(split_window=split_window)
    build_figure_t3()
    build_figure_t4()
    build_figure_t5()


if __name__ == "__main__":
    main()
