"""Shared optional configuration used by CLI and notebook."""
import json
import os
from pathlib import Path

def current(): return json.loads(os.environ.get("EXPERIMENT_CONFIG_JSON","{}"))
def fixed_threshold():
    t=current().get("threshold","validation")
    if t=="validation": return None
    t=float(t)
    if not 0 <= t <= 1: raise ValueError("threshold must lie in [0,1]")
    return t

def active_grid(default):
    from src.models.baselines import HParamGrid
    config=current().get("grid")
    return HParamGrid(**config) if config else default

def save_baseline_config(grid,seed,epochs,experiment):
    from src.utils.io import get_git_commit
    root=Path("experiments/results/q1/baselines");root.mkdir(parents=True,exist_ok=True)
    path=root/"baseline_configs.json"
    all_configs=json.loads(path.read_text()) if path.exists() else {}
    all_configs[f"{experiment}_seed{seed}"]={"seed":seed,"epochs":epochs,"grid":grid.as_dict(),
        "threshold_policy":current().get("threshold","validation"),"git_commit":get_git_commit()}
    path.write_text(json.dumps(all_configs,indent=2))
