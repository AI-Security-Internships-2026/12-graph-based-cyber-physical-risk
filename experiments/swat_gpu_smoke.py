"""GPU smoke run: SWaT windows -> WindowGraphClassifier on CUDA, strict chronological split.

Development check for `src.data.swat`, not a reported experiment. Every statistic is fitted on
rows before the split boundary, and windows that straddle it are purged.

    python -m experiments.swat_gpu_smoke --csv ~/datasets/swat_kaggle_devonly/merged.csv
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

from src.data import swat
from src.models.gnn import WindowGraphClassifier, build_class_weights
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--official-attack", default=None,
                        help="official attack file; --csv is then the official normal v1 file")
    parser.add_argument("--topology", default="correlation", choices=["correlation", "process"])
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script is GPU-only")
    device = torch.device("cuda")
    set_seed(args.seed)

    df = (swat.load_swat_official(args.csv, args.official_attack) if args.official_attack
          else swat.load_swat_kaggle_merged(args.csv))
    boundary = int(len(df) * args.split)
    graphs, labels, extra = swat.build_windowed_graphs(df, fit_rows=np.arange(boundary),
                                                       topology=args.topology, seed=args.seed)
    starts = np.asarray(extra["starts"])
    ends = starts + extra["window_size"]
    train_idx = np.flatnonzero(ends <= boundary)
    test_idx = np.flatnonzero(starts >= boundary)
    print(f"rows={len(df)} components={len(extra['sensor_list'])} edges={extra['n_edges']} "
          f"windows train={len(train_idx)} (attack {labels[train_idx].sum()}) "
          f"test={len(test_idx)} (attack {labels[test_idx].sum()}) "
          f"purged={len(graphs) - len(train_idx) - len(test_idx)}")

    train = [graphs[i] for i in train_idx]
    test = [graphs[i] for i in test_idx]
    weights = build_class_weights(torch.tensor(labels[train_idx])).to(device)
    model = WindowGraphClassifier(in_dim=train[0].x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=1024)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch.x, batch.edge_index, batch.batch), batch.y, weight=weights)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}/{args.epochs} loss={total / len(train_loader):.4f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            probs.append(F.softmax(model(batch.x, batch.edge_index, batch.batch), dim=1)[:, 1].cpu())
    p = torch.cat(probs).numpy()
    y = labels[test_idx]
    pred = (p >= 0.5).astype(int)
    print(f"device={torch.cuda.get_device_name(0)} train_time={time.time() - t0:.1f}s "
          f"peak_gpu_mem={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"test @0.5: precision={precision_score(y, pred, zero_division=0):.3f} "
          f"recall={recall_score(y, pred, zero_division=0):.3f} f1={f1_score(y, pred, zero_division=0):.3f} "
          f"AP={average_precision_score(y, p):.3f}")


if __name__ == "__main__":
    main()
