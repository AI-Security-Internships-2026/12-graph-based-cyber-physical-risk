"""READ BEFORE USING THE GRAPH-EXPLAINER FUNCTION IN THIS FILE:

This file does NOT call `torch_geometric.explain.GNNExplainer` — that class's soft edge-
masking approach needs a GNN layer that accepts a per-edge `edge_weight` in its forward
pass (GCNConv does; this repo's `SAGEEmbedder` uses `SAGEConv`, whose PyG forward
signature is `forward(x, edge_index, size=None)` — NO edge_weight parameter), so wiring
the standard `Explainer`/`GNNExplainer` API onto this specific model would need either
changing the model architecture or a nontrivial custom adaptation, and getting that
right with no way to execute or test it felt like a worse bet than a small, fully-hand-
traceable alternative. `topology_occlusion_importance` below implements the SAME GOAL —
which topology edges matter for one query edge's prediction — via discrete edge removal
(occlusion) instead of a learned soft mask: pure `edge_index` subsetting, no gradients,
no layer-specific assumptions. It is slower (one forward pass per candidate edge instead
of one optimization run) but every step of it is ordinary tensor indexing, which is why
it was chosen over a from-scratch, untested GNNExplainer port.

Integrated Gradients (for content/edge_attr features) has no such
mismatch — captum's `IntegratedGradients` just needs a plain
`forward_func`, which is trivial to build around this model's actual
signature — so that part uses the real captum implementation as the
issue asks. `captum` is NOT in this repo's existing dependency list;
add it (`pip install captum`) before running anything in this file that
calls `integrated_gradients_content`.
"""
from typing import Dict, List, Optional

import numpy as np
import torch

from src.evaluation.metrics import compute_metrics
from src.models.baselines import predict_labels


# ---------------------------------------------------------------------
# Stratified sampling across TP/FP/TN/FN
# ---------------------------------------------------------------------

def stratified_sample(pred, true, n_per_category: int = 150, seed: int = 42, strata=None) -> Dict[str, np.ndarray]:
    """Positions (into whatever array `pred`/`true` were computed on — typically
    `test_idx`, so the caller must map these back through `test_idx[positions]` to get
    real row positions into `df`) for each of TP/FP/TN/FN, up to `n_per_category` each.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    rng = np.random.default_rng(seed)
    categories = {
        "TP": np.where((pred == 1) & (true == 1))[0],
        "FP": np.where((pred == 1) & (true == 0))[0],
        "TN": np.where((pred == 0) & (true == 0))[0],
        "FN": np.where((pred == 0) & (true == 1))[0],
    }
    sample = {}
    for cat, positions in categories.items():
        n = min(n_per_category, len(positions))
        if strata is None or n==0:
            sample[cat] = rng.choice(positions, size=n, replace=False) if n > 0 else np.array([], dtype=int)
        else:
            groups={}
            for pos in positions:groups.setdefault(str(strata[pos]),[]).append(int(pos))
            buckets=[list(rng.permutation(v)) for v in groups.values()];rng.shuffle(buckets)
            selected=[]
            while len(selected)<n:
                for bucket in buckets:
                    if bucket and len(selected)<n:selected.append(bucket.pop())
            sample[cat]=np.asarray(selected,dtype=int)
    return sample


# ---------------------------------------------------------------------
# Integrated Gradients over content/edge_attr features
# ---------------------------------------------------------------------

def integrated_gradients_content(
    model, x, edge_index, query_edges, edge_attr, idx, feature_names: List[str],
    n_steps: int = 50, baseline: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None, n_boot: int = 30, boot_seed: int = 42,
) -> Dict:
    """Mean |attribution| per content feature, over the query edges at `idx`, via captum's
    IntegratedGradients — topology (x, edge_index, query_edges) held fixed, only
    edge_attr[idx] is the differentiable input. Also computes a bootstrap-based rank-
    stability score for Table S4's "Rank stability" column: resamples `idx` with
    replacement `n_boot` times, recomputes the top-1 feature each time, and reports the
    fraction of resamples whose top-1 feature matches the full-sample top-1 feature.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError(
            "integrated_gradients_content needs the 'captum' package "
            "(not in this repo's existing dependencies) — pip install captum"
        ) from e

    device = device or torch.device("cpu")
    model = model.to(device).eval()
    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    idx_t = idx if torch.is_tensor(idx) else torch.as_tensor(idx, dtype=torch.long)
    sub_query_edges = query_edges[:, idx_t].to(device)
    inputs = edge_attr[idx_t].clone().to(device)
    baseline_t = (baseline if baseline is not None else torch.zeros_like(inputs)).to(device)

    def forward_fn(edge_attr_batch):
        logits = model(x_dev, edge_index_dev, sub_query_edges, edge_attr_batch)
        return logits[:, 1]

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(inputs, baselines=baseline_t, n_steps=n_steps, internal_batch_size=max(1,len(inputs))).detach().cpu().numpy()
    mean_abs = np.abs(attributions).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs.tolist()), key=lambda kv: -kv[1])
    top1_feature = ranking[0][0] if ranking else None

    rank_stability = None
    if len(idx_t) >= 5 and top1_feature is not None:
        rng = np.random.default_rng(boot_seed)
        matches = 0
        for _ in range(n_boot):
            boot_pos = rng.choice(len(attributions), size=len(attributions), replace=True)
            boot_mean_abs = np.abs(attributions[boot_pos]).mean(axis=0)
            boot_top1 = feature_names[int(np.argmax(boot_mean_abs))]
            matches += int(boot_top1 == top1_feature)
        rank_stability = matches / n_boot

    return {"attributions": attributions, "feature_ranking": ranking,
            "top1_feature": top1_feature, "rank_stability": rank_stability, "n": int(len(idx_t))}


# ---------------------------------------------------------------------
# Topology importance via edge occlusion (see module docstring for why this replaces a
# literal GNNExplainer call)
# ---------------------------------------------------------------------

def topology_occlusion_importance(
    model, x, edge_index, query_edges, edge_attr, query_pos: int,
    max_edges_to_test: int = 200, device: Optional[torch.device] = None,
) -> List[Dict]:
    """For ONE query edge (`query_pos`: a single integer position into
    `query_edges`/`edge_attr`), scores every topology edge incident to
    either endpoint of the query edge by how much removing it changes
    the predicted probability. Returns a list of
    {topology_edge_position, src, dst, prob_change} sorted by
    |prob_change| descending, capped at `max_edges_to_test` candidate
    edges (a busy hub node can have thousands of incident edges;
    testing all of them one-by-one is O(degree) forward passes, so this
    caps the candidate set rather than silently running for a very long
    time — if a node's degree exceeds the cap, the tested subset is a
    random sample of its incident edges, not the full neighborhood)."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    q_edge = query_edges[:, query_pos:query_pos + 1].to(device)
    q_attr = edge_attr[query_pos:query_pos + 1].to(device)

    with torch.no_grad():
        orig_prob = torch.softmax(model(x_dev, edge_index_dev, q_edge, q_attr), dim=1)[0, 1].item()

    src_node, dst_node = int(query_edges[0, query_pos]), int(query_edges[1, query_pos])
    ei_np = edge_index.numpy()
    incident_mask = (ei_np[0] == src_node) | (ei_np[1] == src_node) | \
                     (ei_np[0] == dst_node) | (ei_np[1] == dst_node)
    candidate_positions = np.where(incident_mask)[0]
    if len(candidate_positions) > max_edges_to_test:
        rng = np.random.default_rng(0)
        candidate_positions = rng.choice(candidate_positions, size=max_edges_to_test, replace=False)

    results = []
    with torch.no_grad():
        for pos in candidate_positions:
            keep = np.ones(edge_index.shape[1], dtype=bool)
            keep[pos] = False
            occluded_edge_index = edge_index_dev[:, torch.as_tensor(keep, device=device)]
            prob = torch.softmax(model(x_dev, occluded_edge_index, q_edge, q_attr), dim=1)[0, 1].item()
            results.append({
                "topology_edge_position": int(pos),
                "src": int(ei_np[0, pos]), "dst": int(ei_np[1, pos]),
                "prob_with_edge": orig_prob, "prob_without_edge": prob,
                "prob_change": prob - orig_prob,
            })
    results.sort(key=lambda r: -abs(r["prob_change"]))
    return results


# ---------------------------------------------------------------------
# Explanation intervention test (feature permutation)
# ---------------------------------------------------------------------

def permute_columns(edge_attr: torch.Tensor, idx, col_indices: List[int], seed: int = 42) -> torch.Tensor:
    """Copy of `edge_attr` with the given columns shuffled ACROSS `idx`
    rows only (each column's own marginal distribution over those rows
    is preserved; the row-to-row pairing with the label/other features
    is broken) — the standard permutation-importance perturbation."""
    rng = np.random.default_rng(seed)
    out = edge_attr.clone()
    idx_np = idx.numpy() if torch.is_tensor(idx) else np.asarray(idx)
    perm = rng.permutation(len(idx_np))
    for c in col_indices:
        out[idx_np, c] = out[idx_np[perm], c]
    return out


def run_feature_intervention(
    model, x, edge_index, query_edges, edge_attr, y, idx,
    feature_names: List[str], interventions: Dict[str, Optional[List[str]]],
    threshold: float = 0.5, device: Optional[torch.device] = None, seed: int = 42,
) -> List[Dict]:
    """`interventions`: e.g. {"Original": None, "Protocol_TCP permuted": ["Protocol_TCP"],
    "Top-3 jointly perturbed": [f1, f2, f3]} — one row per entry, `None` meaning the
    unperturbed baseline. Table S5's exact columns: F1/AUPRC/FPR after the intervention,
    and mean absolute change in predicted probability vs the unperturbed baseline.
    """
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    idx_t = idx if torch.is_tensor(idx) else torch.as_tensor(idx, dtype=torch.long)
    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    sub_query_edges = query_edges[:, idx_t].to(device)

    def _probs(attr: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = model(x_dev, edge_index_dev, sub_query_edges, attr[idx_t].to(device))
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    baseline_probs = _probs(edge_attr)
    true_np = y[idx_t].numpy()

    rows = []
    for label, cols in interventions.items():
        missing=[]
        if cols is None:
            probs = baseline_probs
        else:
            col_indices = [feature_names.index(c) for c in cols if c in feature_names]
            missing = [c for c in cols if c not in feature_names]
            if missing:
                print(f"[explainability] intervention {label!r}: feature(s) not found, skipped: {missing}")
            perturbed = permute_columns(edge_attr, idx_t, col_indices, seed=seed)
            probs = _probs(perturbed)
        pred = predict_labels(probs, threshold)
        m = compute_metrics(torch.as_tensor(pred), torch.as_tensor(true_np), torch.as_tensor(probs))
        rows.append({
            "intervention": label, "missing_features":str(missing), "applicable":not missing, "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "auprc": m["auprc"], "fpr": m["fpr"],
            "mean_probability_change": float(np.mean(np.abs(probs - baseline_probs))),
        })
    return rows
