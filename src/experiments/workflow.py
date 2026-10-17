"""One configuration, all seven issues. CLI and notebook use this exact code."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from src.utils.io import write_result,get_git_commit

DEFAULT_SEEDS=[42,7,123,1,2024,13,21,99,2025,314]

def configure(config):
    import torch
    torch.set_num_threads(int(config.get("cpu_threads",2)))
    os.environ["EXPERIMENT_CONFIG_JSON"]=json.dumps(config)
    for key in ("scadanet_path","batadal_path"):
        if not Path(config[key]).is_file(): raise FileNotFoundError(f"Set {key} to your CSV: {config[key]}")
    root=Path("experiments/results/q1");root.mkdir(parents=True,exist_ok=True)
    from src.utils.reproducibility import fingerprint,source_hashes
    import hashlib
    def _sha256(path):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
        return h.hexdigest()
    data_hashes={k:_sha256(config[k]) for k in ("scadanet_path","batadal_path")}
    run={**config,"data_sha256":data_hashes,"git_commit":get_git_commit(),"source_sha256":fingerprint(source_hashes(Path(__file__).resolve().parents[2]))}
    path=root/"run_config.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old!=run: raise ValueError("Result directory belongs to a different configuration/source. Move experiments/results aside before a new run.")
    path.write_text(json.dumps(run,indent=2))
    return root

def run_issue(issue,config):
    root=configure(config);seeds=config.get("seeds",DEFAULT_SEEDS)
    ep=int(config.get("epochs",300));be=int(config.get("batadal_epochs",100));he=int(config.get("host_epochs",400))
    sc=config["scadanet_path"];ba=config["batadal_path"]
    import time
    from src.utils.reproducibility import fingerprint
    start=time.time()
    def stage(name,fn):
        done=root/"completed"/(name+".json");done.parent.mkdir(exist_ok=True)
        if config.get("resume",True) and done.exists():
            print("Reusing completed stage:",name);return
        print("Running:",name,flush=True);fn()
        done.write_text(json.dumps({"stage":name,"git_commit":get_git_commit(),"config_id":fingerprint(config)}))
    if issue==1:
        from src.experiments.runner import REGISTRY
        for dataset,split in [("scadanet","track_b"),("scadanet","temporal"),("batadal","temporal"),("batadal","static_cv")]:
            def run(dataset=dataset,split=split):
                r=REGISTRY[dataset,split](seeds[0],csv_path=sc if dataset=="scadanet" else ba,epochs=ep if dataset=="scadanet" else be)
                r.setdefault("notes",{})["role"]="historical_reference; strict results are Issue 5"
                write_result(r,str(root/"references"/f"{dataset}_{split}_seed{seeds[0]}.json"))
            stage(f"issue1_{dataset}_{split}",run)
    elif issue==2:
        from experiments.run_split_audits import audit_scadanet,audit_batadal,build_summary_csv,OUT_DIR
        OUT_DIR.mkdir(parents=True,exist_ok=True)
        a,b=audit_scadanet(sc),audit_batadal(ba)
        build_summary_csv(a,b,OUT_DIR/"split_audit_summary.csv")
        (OUT_DIR/"batadal_static_audit.json").write_text(json.dumps(b["static_cv"],indent=2,default=str))
        from experiments.build_table_a2_and_figures import main
        main()
    elif issue==3:
        from src.experiments.baseline_runner import run_experiment_b1,run_experiment_b2,run_experiment_b3,_write_one_result
        folder=root/"baselines"
        for seed in seeds:
            for name,fn,path,outfile in [("b1",run_experiment_b1,sc,"scadanet_trackb_baselines.csv"),
                ("b2",run_experiment_b2,sc,"scadanet_temporal_baselines.csv"),
                ("b3",run_experiment_b3,ba,"batadal_temporal_baselines.csv")]:
                stage(f"issue3_{name}_seed{seed}",lambda seed=seed,name=name,fn=fn,path=path,outfile=outfile:
                  fn(seed=seed,csv_path=path,epochs=be if name=="b3" else ep,log_every=0,
                    on_result=lambda r:_write_one_result(r,folder/name,folder/outfile)))
        from experiments.build_table_b1_and_figures import main
        main()
    elif issue==4:
        from src.experiments.ablation_runner import run_condition_matrix_trackb,run_condition_matrix_temporal,run_feature_counterfactuals,_write_one_condition,_feature_counterfactuals_to_csv
        folder=root/"ablations"
        for seed in seeds:
            for name,fn,outfile in [("c1_trackb",run_condition_matrix_trackb,"scadanet_trackb_topology_ablation.csv"),
                                    ("c1_temporal",run_condition_matrix_temporal,"scadanet_temporal_topology_ablation.csv")]:
                stage(f"issue4_{name}_seed{seed}",lambda seed=seed,name=name,fn=fn,outfile=outfile:
                  fn(seed=seed,csv_path=sc,epochs=ep,log_every=0,
                    on_result=lambda r:_write_one_condition(r,folder/name,folder/outfile,folder/"rewiring_metadata.json")))
            def interventions(seed=seed):
                rows=run_feature_counterfactuals(seed=seed,csv_path=sc,epochs=ep,log_every=0)
                _feature_counterfactuals_to_csv(rows,folder/"feature_counterfactuals.csv")
                (folder/f"feature_interventions_seed{seed}.json").write_text(json.dumps(rows,indent=2,default=str))
            stage(f"issue4_interventions_seed{seed}",interventions)
        from experiments.build_table_c1_and_figures import main
        main()
    elif issue==5:
        from src.experiments.temporal_runner import run_scadanet_issue5,run_batadal_issue5
        for seed in seeds:
            stage(f"issue5_scadanet_seed{seed}",lambda seed=seed:run_scadanet_issue5(seed,sc,epochs=ep,log_every=0))
            stage(f"issue5_batadal_seed{seed}",lambda seed=seed:run_batadal_issue5(seed,ba,epochs=be))
        from experiments.build_table_t_and_figures import main
        main()
    elif issue==6:
        from src.experiments.host_norm_runner import run_trackb_host_normalization,run_temporal_host_normalization
        for seed in seeds:
            stage(f"issue6_trackb_seed{seed}",lambda seed=seed:run_trackb_host_normalization(seed,sc,epochs=he,log_every=0))
            stage(f"issue6_temporal_seed{seed}",lambda seed=seed:run_temporal_host_normalization(seed,sc,epochs=ep,log_every=0))
        from experiments.build_table_h_and_figures import main
        main()
        for protocol in ("trackb","temporal"):
            report=pd.read_csv(root/"host_normalization"/f"{protocol}_host_normalization_summary.csv")
            assert "n_seeds" in report, f"Missing seed aggregation: {protocol}"
    elif issue==7:
        from src.experiments.robustness_runner import collect_principal_from_artifacts,summarize_multi_seed,compute_paired_statistics,compute_operational_metrics,aggregate_operational
        raw=collect_principal_from_artifacts(seeds);summarize_multi_seed(raw);compute_paired_statistics(raw)
        from src.experiments.explainability_runner import run_explainability,aggregate_explanations
        for seed in seeds:
            for model in ("graphsage","mlp","tree"):
                stage(f"issue7_operational_{model}_seed{seed}",lambda seed=seed,model=model:compute_operational_metrics(model,seed,sc,epochs=ep))
            stage(f"issue7_explain_seed{seed}",lambda seed=seed:run_explainability(seed,sc,epochs=ep,
              n_per_category=config.get("explanation_samples",150),n_topology_examples_per_category=config.get("topology_examples",5)))
        aggregate_operational();aggregate_explanations()
        from experiments.build_table_s_and_figures import main
        main()
    else:raise ValueError("issue must be 1..7")
    print(f"Issue {issue} completed in {time.time()-start:.1f}s",flush=True)

def run_configured_experiment(config):
    """Common model and threshold interface; historical reference selection is explicit."""
    os.environ["EXPERIMENT_CONFIG_JSON"]=json.dumps(config)
    from src.experiments.baseline_runner import run_experiment_b1,run_experiment_b2,run_experiment_b3
    dataset=config["dataset"];split=config["split"];model=config.get("model","graphsage")
    kwargs={"seed":config.get("seed",42),"csv_path":config.get("data_path"),"epochs":config.get("epochs",300),"families":[model]}
    if dataset=="scadanet" and split=="track_b":return run_experiment_b1(**kwargs)
    if dataset=="scadanet" and split in ("temporal","temporal_strict"):return run_experiment_b2(**kwargs,best_alt_gnn=model if model in ("gat","gcn") else "gcn")
    if dataset=="batadal" and split in ("temporal","temporal_strict"):return run_experiment_b3(**kwargs,alt_gnn=model if model in ("gat","gcn") else "gcn")
    raise ValueError("Common baseline interface supports Track B and strict temporal. Use legacy reference runner for historical Track A/static CV.")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--issue",default="all")
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    for issue in (range(1,8) if args.issue=="all" else [int(args.issue)]):run_issue(issue,config)

if __name__=="__main__":main()
