"""
Fill reproducibility_check.csv by comparing the F1 already recorded in
week12_metricsnew.json against the F1 in the freshly-produced result JSONs
from src.experiments.runner.

Usage:
    python experiments/results/q1/compare.py \
        --old path/to/week12_metricsnew.json \
        --results-dir experiments/results/q1 \
        --map scadanet_trackb_f1=scadanet.track_b.f1:scadanet_graphsage_trackb_seed42.json \
        ...

Kept deliberately simple (no framework) — this is a one-off audit script,
not part of the reusable pipeline.
"""
import argparse
import csv
import json
from pathlib import Path

# Tolerance for "reproduced within normal seed/run variation" (Issue #1,
# acceptance criteria). Tighten/loosen once real numbers are in.
TOLERANCE = 0.02


def get_nested(d, dotted_key):
    for part in dotted_key.split("."):
        d = d[part]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="Path to week12_metricsnew.json")
    parser.add_argument("--results-dir", default="experiments/results/q1")
    args = parser.parse_args()

    old = json.load(open(args.old))
    results_dir = Path(args.results_dir)

    # experiment_id -> (old_json_dotted_path, new_result_filename)
    # TODO: fill in the dotted paths once week12_metricsnew.json's schema
    # is confirmed against the old F1 values already cited in the paper.
    mapping = {
        "scadanet_trackb_f1": (None, "scadanet_graphsage_trackb_seed42.json"),
        "scadanet_temporal_f1": (None, "scadanet_graphsage_temporal_seed42.json"),
        "batadal_temporal_f1": (None, "batadal_graphsage_temporal_seed42.json"),
        "batadal_static_cv_f1": (None, "batadal_graphsage_staticcv_seed42.json"),
    }

    rows = []
    for exp_id, (old_key, result_file) in mapping.items():
        old_val = get_nested(old, old_key) if old_key else None
        result_path = results_dir / result_file
        new_val = None
        if result_path.exists():
            new_val = json.load(open(result_path)).get("f1")

        if old_val is not None and new_val is not None:
            diff = abs(old_val - new_val)
            status = "PASS" if diff <= TOLERANCE else "FAIL"
        else:
            diff, status = "", "pending"

        rows.append([exp_id, old_val or "", new_val or "", diff, status])

    out_path = results_dir / "reproducibility_check.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment_id", "old_result", "reproduced_result",
                     "absolute_difference", "status"])
        w.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
