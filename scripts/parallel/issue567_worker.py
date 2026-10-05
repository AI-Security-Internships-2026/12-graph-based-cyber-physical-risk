"""Run the per-seed stages of Issues 5, 6 and 7 for a subset of seeds in one worker copy.

Same stage names, calls and completion markers as src/experiments/workflow.py; the issue
order is kept (all of this worker's seeds for Issue 5, then 6, then 7). The cross-seed steps
(Issue 7's principal collection, paired statistics and aggregation, and every table build)
are left to the main run after merging.

    python -u issue567_worker.py 42,7
"""
import json
import sys
from pathlib import Path

from src.experiments import workflow
from src.utils.io import get_git_commit
from src.utils.reproducibility import fingerprint

config = json.loads(Path("workflow_config.json").read_text())
seeds = [int(s) for s in sys.argv[1].split(",")]
assert set(seeds) <= set(config["seeds"]), "seed not in the main run's configuration"
root = workflow.configure(config)  # refuses to run if config/source differ from the main run
ep = int(config.get("epochs", 300))
be = int(config.get("batadal_epochs", 100))
he = int(config.get("host_epochs", 400))
sc = config["scadanet_path"]
ba = config["batadal_path"]


def stage(name, fn):
    done = root / "completed" / (name + ".json")
    done.parent.mkdir(exist_ok=True)
    if config.get("resume", True) and done.exists():
        print("Reusing completed stage:", name, flush=True)
        return
    print("Running:", name, flush=True)
    fn()
    done.write_text(json.dumps({"stage": name, "git_commit": get_git_commit(), "config_id": fingerprint(config)}))


from src.experiments.temporal_runner import run_scadanet_issue5, run_batadal_issue5
for seed in seeds:
    stage(f"issue5_scadanet_seed{seed}", lambda seed=seed: run_scadanet_issue5(seed, sc, epochs=ep, log_every=0))
    stage(f"issue5_batadal_seed{seed}", lambda seed=seed: run_batadal_issue5(seed, ba, epochs=be))

from src.experiments.host_norm_runner import run_trackb_host_normalization, run_temporal_host_normalization
for seed in seeds:
    stage(f"issue6_trackb_seed{seed}", lambda seed=seed: run_trackb_host_normalization(seed, sc, epochs=he, log_every=0))
    stage(f"issue6_temporal_seed{seed}", lambda seed=seed: run_temporal_host_normalization(seed, sc, epochs=ep, log_every=0))

from src.experiments.robustness_runner import compute_operational_metrics
from src.experiments.explainability_runner import run_explainability
for seed in seeds:
    for model in ("graphsage", "mlp", "tree"):
        stage(f"issue7_operational_{model}_seed{seed}",
              lambda seed=seed, model=model: compute_operational_metrics(model, seed, sc, epochs=ep))
    stage(f"issue7_explain_seed{seed}", lambda seed=seed: run_explainability(
        seed, sc, epochs=ep, n_per_category=config.get("explanation_samples", 150),
        n_topology_examples_per_category=config.get("topology_examples", 5)))
print("WORKER_DONE", seeds, flush=True)
