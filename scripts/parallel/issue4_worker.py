"""Run Issue 4 stages for a subset of seeds inside one worker copy of the repository.

Run from a worker worktree (same commit as the main run) with the main run's
workflow_config.json, run_config.json and split manifests copied in. The stage bodies and
completion markers are the same as the Issue 4 block of src/experiments/workflow.py; only
the seed loop is restricted and the table build is left to the main run after merging.

    python -u issue4_worker.py 42,7
"""
import json
import sys
from pathlib import Path

from src.experiments import workflow
from src.experiments.ablation_runner import (run_condition_matrix_trackb, run_condition_matrix_temporal,
                                             run_feature_counterfactuals, _write_one_condition,
                                             _feature_counterfactuals_to_csv)
from src.utils.io import get_git_commit
from src.utils.reproducibility import fingerprint

config = json.loads(Path("workflow_config.json").read_text())
seeds = [int(s) for s in sys.argv[1].split(",")]
assert set(seeds) <= set(config["seeds"]), "seed not in the main run's configuration"
root = workflow.configure(config)  # refuses to run if config/source differ from the main run
ep = int(config.get("epochs", 300))
sc = config["scadanet_path"]
folder = root / "ablations"


def stage(name, fn):
    done = root / "completed" / (name + ".json")
    done.parent.mkdir(exist_ok=True)
    if config.get("resume", True) and done.exists():
        print("Reusing completed stage:", name, flush=True)
        return
    print("Running:", name, flush=True)
    fn()
    done.write_text(json.dumps({"stage": name, "git_commit": get_git_commit(), "config_id": fingerprint(config)}))


for seed in seeds:
    for name, fn, outfile in [("c1_trackb", run_condition_matrix_trackb, "scadanet_trackb_topology_ablation.csv"),
                              ("c1_temporal", run_condition_matrix_temporal, "scadanet_temporal_topology_ablation.csv")]:
        stage(f"issue4_{name}_seed{seed}", lambda seed=seed, name=name, fn=fn, outfile=outfile:
              fn(seed=seed, csv_path=sc, epochs=ep, log_every=0,
                 on_result=lambda r: _write_one_condition(r, folder / name, folder / outfile,
                                                          folder / "rewiring_metadata.json")))

    def interventions(seed=seed):
        rows = run_feature_counterfactuals(seed=seed, csv_path=sc, epochs=ep, log_every=0)
        _feature_counterfactuals_to_csv(rows, folder / "feature_counterfactuals.csv")
        (folder / f"feature_interventions_seed{seed}.json").write_text(json.dumps(rows, indent=2, default=str))
    stage(f"issue4_interventions_seed{seed}", interventions)
print("WORKER_DONE", seeds, flush=True)
