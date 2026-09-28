"""Run the split audit on all 5 protocols named in the issue and write the required
artifacts:

    experiments/results/q1/split_audits/scadanet_track_a_audit.json
    experiments/results/q1/split_audits/scadanet_track_b_audit.json
    experiments/results/q1/split_audits/scadanet_temporal_audit.json
    experiments/results/q1/split_audits/batadal_static_audit.json
    experiments/results/q1/split_audits/batadal_temporal_audit.json
    experiments/results/q1/split_audits/split_audit_summary.csv

Usage:
    python -m experiments.run_split_audits         --scadanet-path /path/to/scadanet.csv         --batadal-path BATADAL_dataset04.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.batadal import load_batadal_df, build_windowed_graphs
from src.data.scadanet import load_scadanet_df, build_graph_tensors, LABEL_COL, SRC_IP_COL, DST_IP_COL
from src.evaluation.audit import audit_split
from src.evaluation.temporal_protocol import batadal_purged_split, scadanet_topology_audit
from src.data.scadanet import NUMERIC_FEATURE_COLS
from src.evaluation.splits import (
    scadanet_track_a_split, scadanet_track_b_split, batadal_static_cv_folds,
)
from src.data.scadanet import add_temporal_windows, temporal_split_positions

OUT_DIR = Path("experiments/results/q1/split_audits")
IS_NOT_NORMAL = lambda label: "normal" not in str(label).strip().lower()


def audit_scadanet(csv_path: str) -> dict:
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    results = {}

    # --- Track A (IP-held-out) ---
    train_idx, test_idx, _ = scadanet_track_a_split(gt.query_edges, gt.all_ips, seed=42)
    train_df, test_df = df.iloc[train_idx.numpy()], df.iloc[test_idx.numpy()]
    results["track_a"] = audit_split(
        train_df, test_df, protocol_name="scadanet_track_a",
        src_col=SRC_IP_COL, dst_col=DST_IP_COL, label_col=LABEL_COL,
        positive_label_check=IS_NOT_NORMAL, full_taxonomy=df[LABEL_COL].unique(), numeric_cols=NUMERIC_FEATURE_COLS,
    )

    # --- Track B (single-fixed-IP subtypes vs multi-IP subtypes) ---
    train_idx, test_idx, _ = scadanet_track_b_split(df, LABEL_COL, seed=42)
    train_df, test_df = df.iloc[train_idx.numpy()], df.iloc[test_idx.numpy()]
    results["track_b"] = audit_split(
        train_df, test_df, protocol_name="scadanet_track_b",
        src_col=SRC_IP_COL, dst_col=DST_IP_COL, label_col=LABEL_COL,
        positive_label_check=IS_NOT_NORMAL, full_taxonomy=df[LABEL_COL].unique(), numeric_cols=NUMERIC_FEATURE_COLS,
    )

    # --- Temporal (equal-count windows, chronological) ---
    df_t, snapshots = add_temporal_windows(df, gt.ip_index, n_windows=20)
    train_pos, test_pos, split_window = temporal_split_positions(df_t)
    train_df, test_df = df_t.iloc[train_pos], df_t.iloc[test_pos]
    results["temporal"] = audit_split(
        train_df, test_df, protocol_name="scadanet_temporal",
        src_col=SRC_IP_COL, dst_col=DST_IP_COL, label_col=LABEL_COL,
        time_col="Time", positive_label_check=IS_NOT_NORMAL, full_taxonomy=df[LABEL_COL].unique(), numeric_cols=NUMERIC_FEATURE_COLS,
    )

    for name,mode in [("strict_causal","causal"),("fixed_training","fixed")]:
        import copy
        rec=copy.deepcopy(results["temporal"])
        rec["protocol_name"]="scadanet_"+name
        rec["topology_audit"]=scadanet_topology_audit(df_t,snapshots,split_window,20,gt.ip_index,mode)
        rec["preprocessing_requirement"]="fit on training rows only"
        results[name]=rec
    for rec in results.values():
        for attack,entry in rec["attack_coverage"]["per_label"].items():
            original=df[df[LABEL_COL]==attack]
            entry["unique_src_ips_overall"]=int(original[SRC_IP_COL].nunique())
            entry["unique_dst_ips_overall"]=int(original[DST_IP_COL].nunique())
            entry["diversity_scope"]="full original dataset before Track-B filtering"
    return results


def _batadal_window_df(window_labels, extra) -> pd.DataFrame:
    starts = extra["starts"]
    window_size = extra["window_size"]
    return pd.DataFrame({
        "window_start": starts,
        "window_end": [s + window_size for s in starts],
        "label": window_labels,
    })


def audit_batadal(csv_path: str) -> dict:
    df = load_batadal_df(csv_path)
    window_graphs, window_labels, extra = build_windowed_graphs(df)
    window_df = _batadal_window_df(window_labels, extra)
    results = {}

    folds = batadal_static_cv_folds(window_labels, n_folds=5, seed=42)
    for fold,(train_idx,test_idx) in enumerate(folds):
        results[f"static_fold{fold}"]=audit_split(window_df.iloc[train_idx],window_df.iloc[test_idx],
            protocol_name=f"batadal_static_cv_fold{fold}",window_start_col="window_start",window_end_col="window_end",
            label_col="label",positive_label_check=lambda x: x==1,full_taxonomy=[1])
    results["static_cv"] = dict(results["static_fold0"])
    results["static_cv"]["all_folds"]=[results[f"static_fold{i}"] for i in range(5)]
    results["static_cv"]["notes"]="All five folds audited; representative fields are fold 0. See all_folds and summary rows."

    # --- Temporal 70/30 chronological ---
    split_idx = int(len(window_graphs) * 0.7)
    train_df = window_df.iloc[:split_idx]
    test_df = window_df.iloc[split_idx:]
    results["temporal"] = audit_split(
        train_df, test_df, protocol_name="batadal_temporal",
        window_start_col="window_start", window_end_col="window_end",
        label_col="label",
    )

    for frac in (0.6,0.7,0.8):
        tr,te,purged,meta=batadal_purged_split(len(df),extra["starts"],extra["window_size"],frac)
        key=f"purged_{round(frac*100)}_{round((1-frac)*100)}"
        rec=audit_split(window_df.iloc[tr],window_df.iloc[te],protocol_name="batadal_"+key,
            window_start_col="window_start",window_end_col="window_end",label_col="label",
            positive_label_check=lambda x:x==1,full_taxonomy=[1])
        rec["purge_meta"]=meta
        assert rec["temporal_leakage_windows"]["n_test_windows_sharing_raw_rows_with_train"]==0
        results[key]=rec
    return results


def build_summary_csv(scadanet_results: dict, batadal_results: dict, out_path: Path):
    rows = []
    combined = {**{"scadanet_"+k:v for k,v in scadanet_results.items()},
                **{"batadal_"+k:v for k,v in batadal_results.items()}}
    for name, audit in combined.items():
        identity = audit["identity_topology_overlap"]
        coverage = audit["attack_coverage"]
        temporal_w = audit["temporal_leakage_windows"]
        temporal_f = audit["temporal_leakage_flow"]

        rows.append({
            "protocol": name,
            "train_size": audit["train_size"],
            "test_size": audit["test_size"],
            "host_overlap_pct": (
                identity["host_overlap"]["pct_test_seen_in_train"]
                if identity.get("applicable") else None
            ),
            "pair_overlap_pct": (
                identity["pair_overlap"]["pct_test_seen_in_train"]
                if identity.get("applicable") else None
            ),
            "attack_types_covered": (
                f"{coverage['n_labels_evaluable_in_test']}/{coverage['n_labels_total']}"
                if coverage.get("applicable") else "N/A"
            ),
            "pct_test_nodes_unseen": (
                identity.get("pct_test_nodes_unseen") if identity.get("applicable") else None
            ),
            "pct_test_edges_unseen": (
                identity.get("pct_test_edges_unseen") if identity.get("applicable") else None
            ),
            "temporal_overlap": (
                temporal_w.get("pct_test_windows_overlapping_train")
                if temporal_w.get("applicable")
                else (temporal_f.get("raw_record_time_overlap") if temporal_f.get("applicable") else "N/A")
            ),
            "n_warnings": len(audit["warnings"]),
        })

    for name,rec in combined.items():
        (out_path.parent/f"{name}_audit.json").write_text(json.dumps(rec,indent=2,default=str))
    table=pd.DataFrame(rows)
    table.to_csv(out_path,index=False)
    table.to_csv(out_path.parent/"table_a1_protocol_audit.csv",index=False)
    (out_path.parent/"table_a1_protocol_audit.md").write_text(table.to_markdown(index=False))
    drift=[{"protocol":name,"feature":feature,**entry} for name,rec in combined.items() for feature,entry in rec["distribution_shift"].get("numeric_feature_drift",{}).items()]
    pd.DataFrame(drift).to_csv(out_path.parent/"numeric_feature_drift.csv",index=False)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scadanet-path", default=None)
    parser.add_argument("--batadal-path", default="BATADAL_dataset04.csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scadanet_results = audit_scadanet(args.scadanet_path)
    for key, fname in [("track_a", "scadanet_track_a_audit.json"),
                        ("track_b", "scadanet_track_b_audit.json"),
                        ("temporal", "scadanet_temporal_audit.json")]:
        path = OUT_DIR / fname
        with open(path, "w") as f:
            json.dump(scadanet_results[key], f, indent=2, default=str)
        print(f"wrote {path}")
        for w in scadanet_results[key]["warnings"]:
            print(f"  {w}")

    batadal_results = audit_batadal(args.batadal_path)
    for key, fname in [("static_cv", "batadal_static_audit.json"),
                        ("temporal", "batadal_temporal_audit.json")]:
        path = OUT_DIR / fname
        with open(path, "w") as f:
            json.dump(batadal_results[key], f, indent=2, default=str)
        print(f"wrote {path}")
        for w in batadal_results[key]["warnings"]:
            print(f"  {w}")

    build_summary_csv(scadanet_results, batadal_results, OUT_DIR / "split_audit_summary.csv")


if __name__ == "__main__":
    main()
