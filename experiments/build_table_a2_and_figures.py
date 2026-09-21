"""
Issue #2 — Table A2 and Figures A1-A3, built directly from the saved
audit JSONs (never manually typed values, per the issue's own acceptance
criteria).

Requires the 5 audit JSONs to already exist (run
`python -m experiments.run_split_audits` first).

Usage:
    python -m experiments.build_table_a2_and_figures

Writes into experiments/results/q1/split_audits/:
    table_a2_scadanet_coverage.csv
    table_a2_scadanet_coverage.md
    figure_a1_coverage_heatmap.png
    figure_a2_overlap_novelty.png
    figure_a3_base_rate_shift.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT_DIR = Path("experiments/results/q1/split_audits")

SCADANET_PROTOCOLS = [
    ("Track A", "scadanet_track_a_audit.json"),
    ("Track B", "scadanet_track_b_audit.json"),
    ("Temporal", "scadanet_temporal_audit.json"),
]
ALL_PROTOCOLS = SCADANET_PROTOCOLS + [
    ("BATADAL Static", "batadal_static_audit.json"),
    ("BATADAL Temporal", "batadal_temporal_audit.json"),
]


def _load(fname):
    with open(AUDIT_DIR / fname) as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Table A2 — per-attack-type test coverage across all 3 SCADANet splits
# ---------------------------------------------------------------------

def build_table_a2():
    audits = {name: _load(fname) for name, fname in SCADANET_PROTOCOLS}

    all_labels = set()
    for audit in audits.values():
        cov = audit["attack_coverage"]
        if cov.get("applicable"):
            all_labels |= set(cov["per_label"].keys())
    all_labels = sorted(all_labels)

    rows = []
    for label in all_labels:
        row = {"attack_type": label}
        unique_src = None
        train_samples_seen = None
        for name, _ in SCADANET_PROTOCOLS:
            cov = audits[name]["attack_coverage"]
            entry = cov["per_label"].get(label, {}) if cov.get("applicable") else {}
            row[f"train_{name.lower().replace(' ', '_')}"] = entry.get("train_count", 0)
            row[f"test_{name.lower().replace(' ', '_')}"] = entry.get("test_count", 0)
            if entry.get("unique_src_ips_overall") is not None:
                unique_src = entry["unique_src_ips_overall"]
        row["unique_source_ips"] = unique_src
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = ["attack_type", "unique_source_ips"]
    for name, _ in SCADANET_PROTOCOLS:
        key = name.lower().replace(" ", "_")
        ordered_cols += [f"train_{key}", f"test_{key}"]
    df = df[ordered_cols]

    csv_path = AUDIT_DIR / "table_a2_scadanet_coverage.csv"
    df.to_csv(csv_path, index=False)

    md_path = AUDIT_DIR / "table_a2_scadanet_coverage.md"
    with open(md_path, "w") as f:
        f.write("# Table A2 — SCADANet per-attack split coverage\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return df


# ---------------------------------------------------------------------
# Figure A1 — attack-type coverage by split (presence/absence matrix)
# ---------------------------------------------------------------------

def build_figure_a1():
    audits = {name: _load(fname) for name, fname in SCADANET_PROTOCOLS}
    all_labels = sorted(set().union(*[
        set(a["attack_coverage"]["per_label"].keys())
        for a in audits.values() if a["attack_coverage"].get("applicable")
    ]))
    protocol_names = [name for name, _ in SCADANET_PROTOCOLS]

    matrix = np.zeros((len(all_labels), len(protocol_names)))
    for j, name in enumerate(protocol_names):
        cov = audits[name]["attack_coverage"]["per_label"]
        for i, label in enumerate(all_labels):
            matrix[i, j] = cov.get(label, {}).get("test_count", 0)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(all_labels))))
    display_matrix = np.log1p(matrix)  # log scale so a handful of samples is still visible
    im = ax.imshow(display_matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(protocol_names)))
    ax.set_xticklabels(protocol_names)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=8)

    for i in range(len(all_labels)):
        for j in range(len(protocol_names)):
            count = int(matrix[i, j])
            label = str(count) if count > 0 else "✗"
            color = "white" if display_matrix[i, j] > display_matrix.max() * 0.6 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Figure A1 — Test-set sample count by attack type and protocol\n"
                  "(color = log scale, ✗ = zero test samples)")
    fig.colorbar(im, ax=ax, label="log(1 + test count)")
    fig.tight_layout()

    out_path = AUDIT_DIR / "figure_a1_coverage_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure A2 — train/test overlap and novelty, all 5 protocols
# ---------------------------------------------------------------------

def build_figure_a2():
    audits = {name: _load(fname) for name, fname in ALL_PROTOCOLS}
    names, seen_pct, unseen_pct = [], [], []
    for name, _ in ALL_PROTOCOLS:
        identity = audits[name]["identity_topology_overlap"]
        if not identity.get("applicable"):
            continue
        names.append(name)
        seen_pct.append(identity["host_overlap"]["pct_test_seen_in_train"] * 100)
        unseen_pct.append(identity["pct_test_nodes_unseen"] * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, seen_pct, width, label="% test hosts seen in train", color="#4C72B0")
    ax.bar(x + width / 2, unseen_pct, width, label="% test hosts unseen", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("% of test hosts")
    ax.set_title("Figure A2 — Train/test host overlap and novelty by protocol\n"
                  "(BATADAL excluded: no per-record host identity)")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()

    out_path = AUDIT_DIR / "figure_a2_overlap_novelty.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Figure A3 — base-rate shift across time, SCADANet temporal
# ---------------------------------------------------------------------

def build_figure_a3():
    audit = _load("scadanet_temporal_audit.json")
    per_label = audit["distribution_shift"].get("per_label_base_rates")
    if not per_label:
        print("Figure A3 skipped: scadanet_temporal_audit.json has no per_label_base_rates "
              "(distribution_shift not applicable or no positive_label_check was passed)")
        return

    labels = sorted(per_label.keys())
    train_rates = [per_label[l]["train_rate"] * 100 for l in labels]
    test_rates = [per_label[l]["test_rate"] * 100 for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.35
    colors_train = ["#C44E52" if l == "vuln_scan" else "#4C72B0" for l in labels]
    colors_test = ["#C44E52" if l == "vuln_scan" else "#DD8452" for l in labels]

    ax.bar(x - width / 2, train_rates, width, label="train prevalence", color=colors_train)
    ax.bar(x + width / 2, test_rates, width, label="test prevalence", color=colors_test)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of split")
    ax.set_title("Figure A3 — Base-rate shift across time, SCADANet temporal split\n"
                  "(red = vuln_scan, highlighted per issue spec)")
    ax.legend()
    fig.tight_layout()

    out_path = AUDIT_DIR / "figure_a3_base_rate_shift.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    if not AUDIT_DIR.exists() or not (AUDIT_DIR / "scadanet_track_a_audit.json").exists():
        raise SystemExit(
            f"No audit JSONs found in {AUDIT_DIR}. Run "
            f"`python -m experiments.run_split_audits` first."
        )
    build_table_a2()
    build_figure_a1()
    build_figure_a2()
    build_figure_a3()


if __name__ == "__main__":
    main()
