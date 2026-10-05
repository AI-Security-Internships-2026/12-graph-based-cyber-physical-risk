"""Host-identity intervention for the SCADANet strict temporal protocol.

Question: is the GCN / topology-only vuln_scan detection under the strict temporal split
recognising attack behaviour, or recognising the attacker host 192.168.119.140, which emits
every vuln_scan flow in both periods and 99% attack traffic in training?

For each seed the B2 / C1-temporal setup is rebuilt exactly (same windows, split, validation
carve, edge-attribute fitting), each model is retrained once with that seed's saved best
hyperparameters, and the frozen model is scored with `scadanet_window_eval` (causal mode)
under three test-time conditions:

  baseline    unchanged (reproduces the saved Issue 3/4 numbers)
  alias_<ip>  every test flow touching <ip> is re-attached to one new node with no edges;
              packet content is unchanged, only the host's graph identity is removed.
              Applied to 192.168.119.140 and, as a control, the busiest other test host.

    python -m experiments.host_heldout --out experiments/results/host_heldout
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.scadanet import (LABEL_COL, SRC_IP_COL, DST_IP_COL, add_temporal_windows,
                               build_graph_tensors, load_scadanet_df, temporal_split_positions)
from src.evaluation.metrics import compute_metrics, per_attack_recall
from src.evaluation.splits import carve_validation
from src.evaluation.temporal_protocol import scadanet_window_eval
from src.experiments.baseline_runner import _resolve_device, _threshold_search, _train_gnn_edge, _gnn_edge_probs
from src.models.baselines import predict_labels
from src.models.gnn import build_class_weights

SEEDS = [42, 7, 123, 1, 2024, 13, 21, 99, 2025, 314]
ATTACKER = "192.168.119.140"
# model name -> (conv type, uses real edge attributes, file holding its saved hyperparameters)
MODELS = {"gcn": ("gcn", True, "baselines/b2/scadanet_temporal_strict_gcn_edge_attr_seed{s}.json"),
          "graphsage": ("graphsage", True, "baselines/b2/scadanet_temporal_strict_graphsage_edge_attr_seed{s}.json"),
          "topology_only": ("graphsage", False, "ablations/c1_temporal/scadanet_temporal_strict_B_topology_only_seed{s}.json")}


def setup(seed, csv_path, n_windows=20, val_fraction=0.15):
    """Same steps as run_experiment_b2 up to model training."""
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    df_t, snapshots = add_temporal_windows(df, gt.ip_index, n_windows=n_windows)
    train_idx_np, test_idx_np, split_window = temporal_split_positions(df_t)
    orig_idx = torch.as_tensor(df_t["orig_idx"].to_numpy().copy(), dtype=torch.long)
    query_edges_t = gt.query_edges[:, orig_idx]
    y_t = gt.y[orig_idx]
    train_topology = snapshots[split_window - 1]["edge_index"]
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)
    test_idx = torch.as_tensor(test_idx_np, dtype=torch.long)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)
    gt_fit = build_graph_tensors(df, fit_idx=orig_idx[fit_idx].cpu().numpy())
    edge_attr_t = gt_fit.edge_attr[orig_idx]
    return dict(df_t=df_t, snapshots=snapshots, split_window=split_window, n_windows=n_windows,
                x=gt.x, ip_index=gt.ip_index, query_edges_t=query_edges_t, y_t=y_t,
                edge_attr_t=edge_attr_t, train_topology=train_topology,
                fit_idx=fit_idx, val_idx=val_idx, test_idx=test_idx)


def aliased(s, ip):
    """(x, query_edges) with test flows touching `ip` moved to a new, edgeless node."""
    node = s["ip_index"][ip]
    alias = s["x"].shape[0]
    x = torch.cat([s["x"], s["x"][:1]])  # node features are constant, so the new row matches
    q = s["query_edges_t"].clone()
    cols = s["test_idx"]
    sub = q[:, cols]
    sub[sub == node] = alias
    q[:, cols] = sub
    return x, q


def evaluate(model, s, x, q, attr, device, threshold):
    snaps = [{"edge_index": snap["edge_index"].to(device)} for snap in s["snapshots"]]
    _, _, true, probs = scadanet_window_eval(model, x.to(device), q.to(device), attr.to(device),
                                             s["y_t"].to(device), s["df_t"], snaps, s["split_window"],
                                             s["n_windows"], LABEL_COL, mode="causal")
    probs = probs.cpu().numpy()
    true = true.cpu()
    pred = predict_labels(probs, threshold)
    test_rows = s["df_t"].iloc[s["test_idx"].numpy()]
    m = compute_metrics(torch.as_tensor(pred), true, torch.as_tensor(probs))
    par = per_attack_recall(torch.as_tensor(pred), true, test_rows[LABEL_COL].values)
    y = true.numpy().astype(bool)
    p = np.asarray(pred).astype(bool)
    touches = ((test_rows[SRC_IP_COL].astype(str) == ATTACKER) | (test_rows[DST_IP_COL].astype(str) == ATTACKER)).to_numpy()
    fp = p & ~y
    return {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"],
            "auprc": m.get("auprc"), "vuln_scan_recall": par.get("vuln_scan"),
            "modbus_fdi_recall": par.get("modbus_fdi"), "insider_threat_recall": par.get("insider_threat"),
            "fp_total": int(fp.sum()), "fp_on_attacker_host": int((fp & touches).sum()),
            "fpr_attacker_host_normal": float(fp[touches & ~y].sum() / max((touches & ~y).sum(), 1)),
            "fpr_other_hosts_normal": float(fp[~touches & ~y].sum() / max((~touches & ~y).sum(), 1)),
            "per_attack_recall": json.dumps(par)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/scada_dataset_V01.csv")
    parser.add_argument("--out", default="experiments/results/host_heldout")
    parser.add_argument("--q1", default="experiments/results/q1",
                        help="results folder of the Issue 1-7 run, for each seed's saved hyperparameters")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    device = _resolve_device()
    os.makedirs(args.out, exist_ok=True)
    out_csv = Path(args.out) / "results.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.seed, prev.model))

    for seed in map(int, args.seeds.split(",")):
        s = setup(seed, args.csv)
        test_rows = s["df_t"].iloc[s["test_idx"].numpy()]
        hosts = pd.concat([test_rows[SRC_IP_COL], test_rows[DST_IP_COL]]).astype(str).value_counts()
        control = next(h for h in hosts.index if h != ATTACKER)
        for name in args.models.split(","):
            if (seed, name) in done:
                continue
            conv, real_attr, saved_path = MODELS[name]
            saved = json.loads((Path(args.q1) / saved_path.format(s=seed)).read_text())
            hp = saved["best_hyperparams"]
            attr = s["edge_attr_t"] if real_attr else torch.ones((s["edge_attr_t"].shape[0], 1))
            fit_y = s["y_t"][s["fit_idx"]]
            weights = build_class_weights(fit_y)[fit_y].numpy()
            model = _train_gnn_edge(conv, s["x"].to(device), s["train_topology"].to(device), s["query_edges_t"],
                                    attr, s["y_t"], s["fit_idx"], weights, hp["hidden_dim"], hp["lr"],
                                    hp["dropout"], args.epochs, seed, device, val_idx=None, log_every=0)
            val_probs = _gnn_edge_probs(model, s["x"].to(device), s["train_topology"].to(device),
                                        s["query_edges_t"], attr, s["val_idx"], device)
            threshold, _ = _threshold_search(val_probs, s["y_t"][s["val_idx"]].numpy())
            rows = []
            for cond, ip in (("baseline", None), ("alias_attacker", ATTACKER), ("alias_control", control)):
                x, q = (s["x"], s["query_edges_t"]) if ip is None else aliased(s, ip)
                r = evaluate(model, s, x, q, attr, device, threshold)
                rows.append({"seed": seed, "model": name, "condition": cond, "aliased_host": ip or "",
                             "threshold": threshold, "saved_f1": saved["f1"],
                             "saved_vuln_scan_recall": (saved.get("per_attack_recall") or {}).get("vuln_scan"), **r})
                print(f"seed={seed} {name:14s} {cond:15s} f1={r['f1']:.3f} vuln={r['vuln_scan_recall']:.3f} "
                      f"fpr={r['fpr']:.3f} fp_on_.140={r['fp_on_attacker_host']}/{r['fp_total']} "
                      f"(saved f1={saved['f1']:.3f})", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


if __name__ == "__main__":
    main()
