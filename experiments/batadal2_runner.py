"""Window-graph detectors on GPU: graph topology vs. no-graph baselines under grouped CV.

For every seed and fold: one grouped fold of runs is the test set, one more grouped fold is
held out of training for threshold selection, and the rest train. Models (BATADAL 2.0 names;
WaDi uses graphsage_stage for the process-stage graph instead of graphsage_plc):
  graphsage_plc          WindowGraphClassifier on the PLC + control-loop graph
  graphsage_correlation  WindowGraphClassifier on BATADAL v1's correlation graph
  graphsage_none         WindowGraphClassifier with no edges (topology ablation)
  mlp                    MLP on the same node features, flattened (no graph)
Results go to <out>/results.csv (one row per seed/fold/model/test slice) and summary.csv.

    python -m experiments.batadal2_runner --root ~/datasets/batadal2/extracted --out experiments/results/batadal2
    python -m experiments.batadal2_runner --dataset wadi --root "~/datasets/itrust/WaDi.A2_19 Nov 2019" --out experiments/results/wadi
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

from src.data import batadal2, wadi
from src.models.gnn import WindowGraphClassifier, build_class_weights
from src.utils.seed import set_seed

SEEDS = [42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314]
# dataset -> (loader module, model name -> topology); the MLP ignores edges
DATASETS = {"batadal2": (batadal2, {"graphsage_plc": "plc", "graphsage_correlation": "correlation",
                                    "graphsage_none": "none", "mlp": "plc"}),
            "wadi": (wadi, wadi.TOPOLOGIES)}


class FlatMLP(torch.nn.Module):
    def __init__(self, n_nodes, in_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_nodes * in_dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 2))
        self.n_nodes = n_nodes

    def forward(self, x, edge_index, batch):
        return self.net(x.view(-1, self.n_nodes * x.shape[1]))


def predict(model, graphs, device):
    model.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=2048):
            b = b.to(device)
            out.append(F.softmax(model(b.x, b.edge_index, b.batch), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train(model, graphs, labels, epochs, device, batch_size=256):
    weights = build_class_weights(torch.tensor(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y, weight=weights).backward()
            opt.step()


def best_threshold(y, p):
    """F1-maximising threshold on validation windows; 0.5 if validation has one class."""
    if y.min() == y.max():
        return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 201)))
    return float(max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def metrics(y, p, t):
    pred = p >= t
    row = {"n": len(y), "n_attack": int(y.sum()), "threshold": t,
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "fpr": float(pred[y == 0].mean()) if (y == 0).any() else np.nan}
    row["ap"] = average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="batadal2", choices=sorted(DATASETS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="experiments/results/batadal2")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", default=None, help="comma-separated; default: all of the dataset's models")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this runner is GPU-only")
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    ds, MODELS = DATASETS[args.dataset]
    args.models = args.models or ",".join(MODELS)
    args.root = os.path.expanduser(args.root)

    runs, frames = ds.load(args.root)
    runs["fold"] = ds.grouped_folds(runs, args.folds, seed=0)  # fixed across seeds: paired
    runs.drop(columns="path").to_csv(os.path.join(args.out, "runs_and_folds.csv"), index=False)
    print(f"{len(runs)} runs ({(runs.kind == 'attack').sum()} attack, "
          f"{runs.loc[runs.kind == 'attack', 'signature'].nunique()} attack signatures), {args.folds} folds", flush=True)
    kind = dict(zip(runs.run, runs.kind))
    conceal = dict(zip(runs.run, runs.concealment))
    targets = dict(zip(runs.run, runs.targets))

    results_path = os.path.join(args.out, "results.csv")
    done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        done = set(zip(prev.seed, prev.fold, prev.model))
    for seed in map(int, args.seeds.split(",")):
        for fold in range(args.folds):
            val_fold = (fold + 1) % args.folds
            test_runs = runs[runs.fold == fold]
            val_runs = runs[runs.fold == val_fold]
            train_runs = runs[~runs.fold.isin([fold, val_fold])]
            datasets = {}
            for model_name in args.models.split(","):
                if (seed, fold, model_name) in done:
                    continue
                topology = MODELS[model_name]
                if topology not in datasets:
                    datasets[topology] = ds.build_dataset(args.root, runs, train_runs.run.tolist(),
                                                          topology=topology, frames=frames, seed=seed)
                graphs, info = datasets[topology]
                wr = info["window_runs"]
                y_all = np.array([int(g.y) for g in graphs])
                idx = {name: np.flatnonzero(np.isin(wr, part.run.to_numpy()))
                       for name, part in (("train", train_runs), ("val", val_runs), ("test", test_runs))}
                set_seed(seed)
                n_nodes, in_dim = graphs[0].x.shape
                model = (FlatMLP(n_nodes, in_dim) if model_name == "mlp"
                         else WindowGraphClassifier(in_dim=in_dim)).to(device)
                t0 = time.time()
                train(model, [graphs[i] for i in idx["train"]], y_all[idx["train"]], args.epochs, device)
                elapsed = time.time() - t0
                p_val = predict(model, [graphs[i] for i in idx["val"]], device)
                thr = best_threshold(y_all[idx["val"]], p_val)
                p_test = predict(model, [graphs[i] for i in idx["test"]], device)
                y_test, run_test = y_all[idx["test"]], wr[idx["test"]]
                slices = {"all": np.ones(len(y_test), bool),
                          "normal_runs": np.array([kind[r] == "normal" for r in run_test]),
                          "attack_no_concealment": np.array([kind[r] == "attack" and not conceal[r] for r in run_test]),
                          "attack_concealment": np.array([kind[r] == "attack" and conceal[r] for r in run_test])}
                for plc in sorted({t for r in run_test for t in targets[r].split(",") if t}):
                    slices[f"target_{plc}"] = np.array([plc in targets[r].split(",") for r in run_test])
                rows = []
                for name, mask in slices.items():
                    if mask.any():
                        for thr_name, t in (("val_selected", thr), ("fixed_0.5", 0.5)):
                            rows.append({"seed": seed, "fold": fold, "model": model_name, "slice": name,
                                         "threshold_rule": thr_name, "n_edges": info["n_edges"],
                                         "train_seconds": round(elapsed, 1),
                                         **metrics(y_test[mask], p_test[mask], t)})
                pd.DataFrame(rows).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)
                main_row = rows[0]
                print(f"seed={seed} fold={fold} {model_name:22s} edges={info['n_edges']:3d} "
                      f"f1={main_row['f1']:.3f} ap={main_row['ap']:.3f} fpr={main_row['fpr']:.3f} "
                      f"thr={thr:.3f} {elapsed:.0f}s", flush=True)

    res = pd.read_csv(results_path)
    summary = (res.groupby(["model", "slice", "threshold_rule"])[["precision", "recall", "f1", "fpr", "ap"]]
                  .agg(["mean", "std"]).round(4))
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(summary.xs("all", level="slice").to_string())


if __name__ == "__main__":
    main()
