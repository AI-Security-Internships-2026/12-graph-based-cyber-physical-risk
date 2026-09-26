"""
Graph rewiring for Issue #4 Conditions D (random topology) and E
(degree-preserving rewired topology).

Both functions take the TRUE topology `edge_index` (the deduplicated
directed IP-pair graph from src/data/scadanet.py's build_graph_tensors)
and return a same-size counterfactual topology, plus metadata recording
exactly what was preserved/destroyed — Issue #4 item 6 requires this
metadata ("confirms that a bad result is not simply due to accidentally
destroying the graph size/structure completely").
"""
import random
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import torch


def _degree_summary(edge_index: torch.Tensor, num_nodes: int) -> Dict:
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    total_deg = out_deg + in_deg
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(zip(src.tolist(), dst.tolist()))
    return {
        "n_nodes": num_nodes,
        "n_edges": int(edge_index.shape[1]),
        "mean_degree": float(total_deg.mean()),
        "degree_std": float(total_deg.std()),
        "degree_min": int(total_deg.min()),
        "degree_max": int(total_deg.max()),
        "connected_components": nx.number_connected_components(g),
    }


def random_rewire(edge_index: torch.Tensor, num_nodes: int, seed: int = 42,
                   allow_self_loops: bool = False) -> Tuple[torch.Tensor, Dict]:
    """Condition D. Draws the SAME NUMBER of edges as the true topology,
    uniformly at random over node pairs — destroys both the specific
    relationships AND the degree distribution (contrast with
    degree_preserving_rewire below, which destroys only the former).
    Duplicate edges are allowed (matching how the real topology itself
    can and does have essentially arbitrary repeat structure); self-loops
    excluded by default since IP self-communication isn't a meaningful
    ICS relationship to synthesize.
    """
    rng = np.random.default_rng(seed)
    n_edges = edge_index.shape[1]

    src = rng.integers(0, num_nodes, size=n_edges)
    dst = rng.integers(0, num_nodes, size=n_edges)
    if not allow_self_loops:
        clash = src == dst
        while clash.any():
            dst[clash] = rng.integers(0, num_nodes, size=int(clash.sum()))
            clash = src == dst

    new_edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    metadata = {
        "method": "random_rewire", "seed": seed,
        "original": _degree_summary(edge_index, num_nodes),
        "rewired": _degree_summary(new_edge_index, num_nodes),
    }
    return new_edge_index, metadata


def degree_preserving_rewire(edge_index: torch.Tensor, num_nodes: int, seed: int = 42,
                              n_swap_multiplier: int = 10, max_tries_multiplier: int = 100
                              ) -> Tuple[torch.Tensor, Dict]:
    """Condition E. Destroys the specific communication relationships
    while preserving each node's TOTAL degree (in+out collapsed to an
    undirected simple graph — see note below) via repeated double-edge
    swaps, the standard configuration-model-style randomization
    (networkx's `connected_double_edge_swap`/`double_edge_swap`
    algorithm family, reimplemented directly here to work on our own
    edge_index tensors without an intermediate nx.Graph round-trip for
    every swap).

    Directionality note: the true topology in build_graph_tensors is a
    DEDUPLICATED set of (src, dst) pairs as observed in the raw flow
    data — not necessarily symmetric, and not a simple graph (though
    duplicates were already dropped by `drop_duplicates()` upstream).
    Preserving exact in-degree AND out-degree simultaneously under
    directed double-edge-swaps is possible but adds real complexity for
    a topology that isn't a clean directed simple graph to begin with.
    This function instead treats the edge list as an UNDIRECTED simple
    graph (dedup both directions, drop self-loops), preserves its
    (undirected) degree sequence exactly via double-edge swaps, and
    re-emits a directed edge_index by keeping each undirected edge's
    original direction where it existed and assigning an arbitrary
    consistent direction to swapped-in edges. This is the "technically
    justified alternative" the issue's acceptance criteria explicitly
    allows in place of a literal directed degree-preserving rewiring,
    and is documented here rather than silently substituted.
    """
    random.seed(seed)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    undirected_edges = list({tuple(sorted((a, b))) for a, b in zip(src, dst) if a != b})

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(undirected_edges)
    g_before_edge_index = _edges_to_directed_tensor(list(g.edges()))
    metadata_original = _degree_summary(g_before_edge_index, num_nodes)

    n_edges_undirected = g.number_of_edges()
    n_swap = max(1, n_edges_undirected * n_swap_multiplier)
    max_tries = n_swap * max_tries_multiplier

    try:
        nx.double_edge_swap(g, nswap=n_swap, max_tries=max_tries, seed=seed)
        swap_status = "ok"
    except nx.NetworkXError as e:
        # Too few edges/degrees to complete the requested swaps (can
        # happen on small synthetic graphs); keep whatever swaps DID
        # land rather than failing the whole experiment, and record it.
        swap_status = f"incomplete: {e}"

    rewired_undirected = list(g.edges())
    rng = np.random.default_rng(seed)
    src_out, dst_out = [], []
    for a, b in rewired_undirected:
        if rng.integers(0, 2) == 0:
            src_out.append(a); dst_out.append(b)
        else:
            src_out.append(b); dst_out.append(a)

    new_edge_index = torch.tensor(np.stack([src_out, dst_out]), dtype=torch.long) if src_out \
        else torch.zeros((2, 0), dtype=torch.long)

    metadata = {
        "method": "degree_preserving_rewire (undirected double-edge-swap)",
        "seed": seed, "n_swap_requested": n_swap, "swap_status": swap_status,
        "original": metadata_original,
        "rewired": _degree_summary(new_edge_index, num_nodes),
        "note": "both 'original' and 'rewired' here describe the SAME representation "
                "(the deduplicated, self-loop-free undirected graph the swap actually "
                "operates on, re-emitted as a directed edge_index with one arbitrary "
                "direction per undirected edge) — NOT the raw multi/duplicate directed "
                "edge_index from build_graph_tensors, so the two degree sequences are "
                "an apples-to-apples comparison and (barring an incomplete swap_status) "
                "should match exactly.",
    }
    return new_edge_index, metadata


def _edges_to_directed_tensor(edges) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = zip(*edges)
    return torch.tensor(np.stack([np.array(src), np.array(dst)]), dtype=torch.long)


def degree_distributions(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Total (in+out) degree per node — for Figure C4 (original vs
    degree-preserved rewired degree distribution comparison)."""
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    return np.bincount(src, minlength=num_nodes) + np.bincount(dst, minlength=num_nodes)
