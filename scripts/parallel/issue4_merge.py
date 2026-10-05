"""Merge Issue 4 results from worker copies into the main run's results folder.

    python issue4_merge.py <main_q1_root> <worker_q1_root> [<worker_q1_root> ...]

Copies per-seed JSONs and completion markers, concatenates the shared CSVs and the
rewiring metadata with the same de-duplication keys the ablation runner uses, and adds new
split manifests. Fails before writing anything if a worker is missing a stage.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

main = Path(sys.argv[1])
workers = [Path(p) for p in sys.argv[2:]]
config = json.loads((main / "run_config.json").read_text())
expected = {f"issue4_{kind}_seed{s}.json" for s in config["seeds"]
            for kind in ("c1_trackb", "c1_temporal", "interventions")}
found = {m.name: w for w in workers for m in (w / "completed").glob("issue4_*.json")}
missing = sorted(expected - set(found))
if missing:
    sys.exit(f"Not merging, missing stages: {missing}")
if (main / "ablations").exists() or any((main / "completed").glob("issue4_*.json")):
    sys.exit("Main results already contain Issue 4 output; move it aside before merging")

dst = main / "ablations"
for sub in ("c1_trackb", "c1_temporal"):
    (dst / sub).mkdir(parents=True, exist_ok=True)
csv_keys = {"scadanet_trackb_topology_ablation.csv": ["dataset", "split_protocol", "condition", "conv_type", "seed"],
            "scadanet_temporal_topology_ablation.csv": ["dataset", "split_protocol", "condition", "conv_type", "seed"],
            "feature_counterfactuals.csv": ["dataset", "split_protocol", "seed", "conv_type", "removed_or_permuted_feature"]}
frames = {name: [] for name in csv_keys}
rewiring = []
for w in workers:
    src = w / "ablations"
    for sub in ("c1_trackb", "c1_temporal"):
        for f in (src / sub).glob("*.json"):
            shutil.copy2(f, dst / sub / f.name)
    for f in src.glob("feature_interventions_seed*.json"):
        shutil.copy2(f, dst / f.name)
    for name in csv_keys:
        if (src / name).exists():
            frames[name].append(pd.read_csv(src / name))
    if (src / "rewiring_metadata.json").exists():
        rewiring += json.loads((src / "rewiring_metadata.json").read_text())
    for f in (w / "splits").glob("*.json"):
        if not (main / "splits" / f.name).exists():
            shutil.copy2(f, main / "splits" / f.name)

for name, keys in csv_keys.items():
    df = pd.concat(frames[name], ignore_index=True).drop_duplicates(keys, keep="last")
    df.sort_values([k for k in ("seed", "condition") if k in df]).to_csv(dst / name, index=False)
    print(f"{name}: {len(df)} rows, seeds {sorted(df.seed.unique().tolist())}")
(dst / "rewiring_metadata.json").write_text(json.dumps(rewiring, indent=2, default=str))
print(f"rewiring_metadata.json: {len(rewiring)} entries")
for name, w in found.items():
    shutil.copy2(w / "completed" / name, main / "completed" / name)
print(f"merged {len(found)} completion markers")
