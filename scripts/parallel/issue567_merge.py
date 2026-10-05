"""Merge Issues 5-7 results from worker copies into the main run's results folder.

    python issue567_merge.py <main_q1_root> <last_worker_q1_root> <worker_q1_root> [...]

The second argument is the worker that ran the configuration's last seed: in a sequential
run the fixed-name ("canonical") files of each issue hold the last seed's output, so they are
taken from that worker. Files named *_seed<N>.* come from the worker that produced them.
Fails before writing anything if a stage is missing or two workers produced the same file.
"""
import json
import re
import shutil
import sys
from pathlib import Path

main = Path(sys.argv[1])
last = Path(sys.argv[2])
workers = [Path(p) for p in sys.argv[2:]]
DIRS = ("temporal", "host_normalization", "robustness")
config = json.loads((main / "run_config.json").read_text())
seeds = config["seeds"]
expected = {f"{s}.json" for seed in seeds for s in
            [f"issue5_scadanet_seed{seed}", f"issue5_batadal_seed{seed}", f"issue6_trackb_seed{seed}",
             f"issue6_temporal_seed{seed}", f"issue7_explain_seed{seed}"]
            + [f"issue7_operational_{m}_seed{seed}" for m in ("graphsage", "mlp", "tree")]}
markers = {}
for w in workers:
    for m in (w / "completed").glob("issue[567]_*.json"):
        markers.setdefault(m.name, w)
missing = sorted(expected - set(markers))
if missing:
    sys.exit(f"Not merging, missing stages: {missing}")
if any((main / d).exists() for d in DIRS) or any((main / "completed").glob("issue[567]_*.json")):
    sys.exit("Main results already contain Issue 5-7 output; move it aside before merging")
if any(name.endswith(f"_seed{seeds[-1]}.json") and w != last for name, w in markers.items()):
    sys.exit(f"{last} did not run the last seed {seeds[-1]}")

plan = {}
for w in workers:
    for d in DIRS:
        for f in (w / d).rglob("*") if (w / d).exists() else []:
            if not f.is_file():
                continue
            rel = f.relative_to(w)
            if re.search(r"seed\d+", f.name):
                if rel in plan and plan[rel] != f:
                    sys.exit(f"Two workers produced {rel}")
                plan[rel] = f
            elif w == last:
                plan[rel] = f
for rel, f in plan.items():
    (main / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, main / rel)
print(f"copied {len(plan)} result files into {', '.join(DIRS)}")
for w in workers:
    for f in (w / "splits").glob("*.json"):
        if not (main / "splits" / f.name).exists():
            shutil.copy2(f, main / "splits" / f.name)
for name, w in markers.items():
    shutil.copy2(w / "completed" / name, main / "completed" / name)
print(f"merged {len(markers)} completion markers")
