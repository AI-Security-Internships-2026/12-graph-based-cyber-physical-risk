"""Artifact contract checks after the full seven-issue workflow."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

def validate_results(seeds,root=Path("experiments/results/q1")):
    root=Path(root);checks=[]
    def check(name,ok):
        checks.append({"check":name,"passed":bool(ok)})
        if not ok:raise AssertionError(name)
    expected=set(seeds)
    for stem,families in [("scadanet_trackb_baselines",6),("scadanet_temporal_baselines",4),("batadal_temporal_baselines",3)]:
        df=pd.read_csv(root/"baselines"/(stem+".csv"))
        check(stem+" has every model for every seed",set(df.seed)==expected and all(df.groupby("seed").model_family.nunique()==families))
    for protocol in ("trackb","temporal"):
        df=pd.read_csv(root/"ablations"/f"scadanet_{protocol}_topology_ablation.csv")
        check("A-F paired: "+protocol,set(df.seed)==expected and all(df.groupby("seed").condition.nunique()==6))
        df=pd.read_csv(root/"host_normalization"/f"{protocol}_host_normalization_summary.csv")
        check("H1-H4 aggregate: "+protocol,len(df)==4 and all(df.n_seeds==len(seeds)))
    for seed in seeds:
        t=pd.read_csv(root/"temporal"/f"scadanet_temporal_summary_seed{seed}.csv")
        check(f"T1/T2/T3 matched config seed {seed}",len(t)==3 and t.config_id.nunique()==1 and t.threshold.nunique()==1)
        a=pd.read_csv(root/"temporal"/f"scadanet_topology_audit_seed{seed}.csv")
        check(f"causal topology audit seed {seed}",a.uses_only_prior_windows.all() and (a.n_unobserved_edges==0).all())
        per=pd.read_csv(root/"temporal"/f"scadanet_temporal_per_attack_seed{seed}.csv")
        comparable=per.static_trackb_recall.notna() & per.strict_temporal_recall.notna()
        check(f"matched recall and deltas seed {seed}",comparable.any() and np.allclose(per.loc[comparable,'delta_recall'],per.loc[comparable,'strict_temporal_recall']-per.loc[comparable,'static_trackb_recall']))
        audit=json.loads((root/"temporal"/f"batadal_boundary_audit_seed{seed}.json").read_text())
        purged=[v for k,v in audit.items() if "purged" in k]
        check(f"3 zero-overlap BATADAL audits seed {seed}",len(purged)==3 and all(x['temporal_leakage_windows']['n_test_windows_sharing_raw_rows_with_train']==0 for x in purged))
    tables=list(root.rglob("table_*.csv"));figures=list(root.rglob("figure_*.png"))
    for letter in "abcths":
        check("tables "+letter,any(f.name.startswith("table_"+letter) for f in tables))
        check("figures "+letter,any(f.name.startswith("figure_"+letter) for f in figures))
    for f in list((root/"baselines").rglob("*_seed*.json"))+list((root/"ablations").rglob("scadanet*_seed*.json")):
        r=json.loads(f.read_text());check("provenance "+f.name,bool(r.get("git_commit")) and bool(r.get("split_manifests")))
    all_folds=json.loads((root/"split_audits"/"batadal_static_audit.json").read_text())
    check("all BATADAL folds audited",len(all_folds["all_folds"])==5)
    s1=pd.read_csv(root/"robustness"/"table_s1_robustness_summary.csv")
    check("all principal comparisons have every seed",all(s1.n_seeds==len(seeds)))
    s4=pd.read_csv(root/"robustness"/"explainability_cross_seed_stability.csv")
    check("cross-seed explanation pairs",len(s4)>0)
    report={"checks":checks,"n_checks":len(checks),"n_tables":len(tables),"n_figures":len(figures),"seeds":seeds}
    (root/"validation_report.json").write_text(json.dumps(report,indent=2))
    return report

def write_interpretation(root=Path("experiments/results/q1")):
    root=Path(root);s=pd.read_csv(root/"temporal"/"scadanet_temporal_summary.csv").set_index("protocol")
    old=s.loc["T1_previous_temporal"];strict=s.loc["T2_strict_causal_topology"]
    lines=["# Temporal and mitigation evidence for the paper update", "", "Generated from saved runs. Verify the run profile; synthetic results are not research findings.","",
       f"T1 mean F1: {old.f1:.6f}; T2 mean F1: {strict.f1:.6f}; change: {strict.f1-old.f1:+.6f}.",
       f"T1 mean AUPRC: {old.auprc:.6f}; T2 mean AUPRC: {strict.auprc:.6f}; change: {strict.auprc-old.auprc:+.6f}.",
       "Leakage removal need not increase either metric. Inspect the paired variation before describing the change as material.","",
       "## Per-attack temporal behavior",pd.read_csv(root/"temporal"/"table_t2_scadanet_per_attack.csv")[["Attack type","Static/Track-B recall","Strict temporal recall","Delta recall"]].to_markdown(index=False),
       "", "## Host normalization trade-offs", "Retain lower attack recall and higher temporal FPR wherever observed. Evaluate .140 and .141 separately using H2 and the full per-host seed artifacts.",
       pd.read_csv(root/"host_normalization"/"table_h3_attack_recall_before_after.csv").to_markdown(index=False),"",
       "The draft paper was not supplied. Issue 29 must integrate these real-data results and remove obsolete future-work claims after validation."]
    out=root/"PAPER_UPDATE_EVIDENCE.md";out.write_text("\n".join(lines));return out
