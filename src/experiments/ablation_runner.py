"""
Issue #4 — topology-versus-content ablations and graph counterfactual
experiments.

Reuses Issue #3's split/training/metrics machinery unchanged; this file
adds only what's specific to Issue #4: the six experimental conditions
(A-F), graph rewiring counterfactuals (src/graph/rewiring.py), and
feature-permutation counterfactuals.

Every condition, for a given seed, is trained/evaluated on the EXACT
SAME fit/val/test split (Issue #4's "same train/validation/test samples
... where possible" requirement) — the split is built once per seed and
threaded into every condition function.
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.data.scadanet import (
    load_scadanet_df, build_graph_tensors, add_temporal_windows,
    temporal_split_positions, LABEL_COL,
)
from src.evaluation.metrics import compute_metrics, per_attack_recall
from src.evaluation.splits import scadanet_track_b_split, carve_validation
from src.experiments.baseline_runner import (
    gnn_edge_family, nongnn_family, _gnn_edge_probs, _threshold_search,
    _resolve_device, DEFAULT_GRID,
)
from src.graph.rewiring import random_rewire, degree_preserving_rewire, degree_distributions
from src.models.baselines import HParamGrid, predict_labels
from src.utils.io import write_result
from src.utils.seed import set_seed

RESULTS_DIR = Path("experiments/results/q1/ablations")

# Content features selected for the permutation experiments.
TOP_FEATURES = ["Protocol_TCP", "Tcp_flags_reset_Set", "frame_len"]


def find_feature_columns(feature_names: List[str], requested_name: str) -> List[int]:
    """Exact match first (numeric columns, or a one-hot column that
    happens to match exactly); else every column whose name starts with
    `requested_name` + "_" (all dummy columns of a one-hot-expanded
    categorical, permuted together as one block since they jointly
    represent one underlying variable). Returns [] if nothing matches —
    callers must handle this by skipping with a note, not crashing."""
    if requested_name in feature_names:
        return [feature_names.index(requested_name)]
    prefix = requested_name + "_"
    return [i for i, n in enumerate(feature_names) if n.startswith(prefix)]


# ---------------------------------------------------------------------
# Condition A — content only (MLP, same content features, no graph)
# ---------------------------------------------------------------------

def condition_a_content_only(edge_attr, y, df, fit_idx, val_idx, test_idx, seed,
                              grid: HParamGrid, subtype_balanced: bool, mlp_epochs: int) -> Dict:
    r = nongnn_family("mlp", edge_attr, y, df, fit_idx, val_idx, test_idx, seed,
                       grid=grid, subtype_balanced=subtype_balanced, mlp_epochs=mlp_epochs)
    r["condition"] = "A_content_only"
    r["notes"]["condition_definition"] = "MLP, real content features, no graph structure at all."
    return r


# ---------------------------------------------------------------------
# Condition B — topology only (true graph + minimal node features,
# constant dummy edge_attr so the model sees no content signal)
# ---------------------------------------------------------------------

def condition_b_topology_only(conv_type, x, edge_index, query_edges, y, df,
                               fit_idx, val_idx, test_idx, seed, epochs,
                               grid: HParamGrid, subtype_balanced: bool,
                               device, log_every: int,
                               strict_temporal_test: Optional[Dict] = None) -> Dict:
    dummy_edge_attr = torch.ones((query_edges.shape[1], 1), dtype=torch.float32)
    r = gnn_edge_family(conv_type, x, edge_index, query_edges, dummy_edge_attr, y,
                         df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=grid,
                         subtype_balanced=subtype_balanced, device=device, log_every=log_every,
                         strict_temporal_test=strict_temporal_test)
    r["condition"] = "B_topology_only"
    r["feature_set"] = "none_constant_dummy"
    r["notes"]["condition_definition"] = (
        "True graph + true node features (already constant/non-identifying "
        "by construction, per src/data/scadanet.py) + a CONSTANT dummy "
        "edge_attr (all ones, 1 column) in place of real content features. "
        "Any signal here comes only from graph structure/degree via message "
        "passing, not from packet/flow content."
    )
    return r


# ---------------------------------------------------------------------
# Condition C — full model (true graph + node features + edge/content
# features) — the reference condition every other condition is compared
# against (ΔF1/ΔAUPRC/ΔFPR are always relative to this).
# ---------------------------------------------------------------------

def condition_c_full(conv_type, x, edge_index, query_edges, edge_attr, y, df,
                      fit_idx, val_idx, test_idx, seed, epochs, grid: HParamGrid,
                      subtype_balanced: bool, device, log_every: int,
                      return_model: bool = False, strict_temporal_test: Optional[Dict] = None):
    out = gnn_edge_family(conv_type, x, edge_index, query_edges, edge_attr, y,
                           df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=grid,
                           subtype_balanced=subtype_balanced, device=device, log_every=log_every,
                           return_model=return_model, strict_temporal_test=strict_temporal_test)
    if return_model:
        r, model, dev, x_dev, edge_index_dev = out
    else:
        r = out
    r["condition"] = "C_full_model"
    r["notes"]["condition_definition"] = "True graph, true node features, true content features. Reference condition."
    if return_model:
        return r, model, dev, x_dev, edge_index_dev
    return r


# ---------------------------------------------------------------------
# Condition D — randomly shuffled topology (real content, real query
# edges, but message passing runs over a fully random graph)
# ---------------------------------------------------------------------

def condition_d_random_topology(conv_type, x, edge_index, query_edges, edge_attr, y, df,
                                 fit_idx, val_idx, test_idx, seed, epochs, grid: HParamGrid,
                                 subtype_balanced: bool, device, log_every: int):
    rewired_edge_index, rewire_meta = random_rewire(edge_index, num_nodes=x.shape[0], seed=seed)
    r = gnn_edge_family(conv_type, x, rewired_edge_index, query_edges, edge_attr, y,
                         df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=grid,
                         subtype_balanced=subtype_balanced, device=device, log_every=log_every)
    r["condition"] = "D_random_topology"
    r["notes"]["condition_definition"] = ("Same samples/content features/query edges as Condition C, "
                                           "but message passing runs over a FULLY RANDOM graph of the "
                                           "same size (src/graph/rewiring.py random_rewire).")
    r["notes"]["rewiring_metadata"] = rewire_meta
    return r


# ---------------------------------------------------------------------
# Condition E — degree-preserving rewired topology
# ---------------------------------------------------------------------

def condition_e_degree_preserving(conv_type, x, edge_index, query_edges, edge_attr, y, df,
                                   fit_idx, val_idx, test_idx, seed, epochs, grid: HParamGrid,
                                   subtype_balanced: bool, device, log_every: int):
    rewired_edge_index, rewire_meta = degree_preserving_rewire(edge_index, num_nodes=x.shape[0], seed=seed)
    r = gnn_edge_family(conv_type, x, rewired_edge_index, query_edges, edge_attr, y,
                         df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=grid,
                         subtype_balanced=subtype_balanced, device=device, log_every=log_every)
    r["condition"] = "E_degree_preserving_rewiring"
    r["notes"]["condition_definition"] = ("Same as Condition D, but the rewired graph preserves the "
                                           "true topology's (undirected) degree sequence exactly "
                                           "(src/graph/rewiring.py degree_preserving_rewire) — "
                                           "isolates degree/topological-statistics effects from "
                                           "genuine specific-relationship effects.")
    r["notes"]["rewiring_metadata"] = rewire_meta
    return r


# ---------------------------------------------------------------------
# Condition F — no-message-passing / endpoint-only ablation
# ---------------------------------------------------------------------

def condition_f_no_message_passing(x, edge_index, query_edges, edge_attr, y, df,
                                    fit_idx, val_idx, test_idx, seed, epochs, grid: HParamGrid,
                                    subtype_balanced: bool, device, log_every: int) -> Dict:
    r = gnn_edge_family("endpoint_only", x, edge_index, query_edges, edge_attr, y,
                         df, fit_idx, val_idx, test_idx, seed, epochs=epochs, grid=grid,
                         subtype_balanced=subtype_balanced, device=device, log_every=log_every)
    r["condition"] = "F_no_message_passing"
    r["notes"]["condition_definition"] = (
        "Learnable per-node embedding table (no graph convolution, no "
        "neighborhood aggregation) + real content features. See "
        "EndpointOnlyEdgeClassifier docstring in src/models/gnn.py for "
        "why a learnable embedding rather than a plain MLP is used here — "
        "the repo's node features are constant, so an MLP endpoint encoder "
        "would carry zero endpoint-identity signal and collapse into "
        "Condition A."
    )
    return r


# ---------------------------------------------------------------------
# Table C1 orchestration — Conditions A-F, one seed, one protocol
# ---------------------------------------------------------------------

def run_condition_matrix_trackb(
    seed: int, conv_type: str = "graphsage", csv_path: str = None,
    epochs: int = 300, val_fraction: float = 0.15, grid: HParamGrid = DEFAULT_GRID,
    device: Optional[torch.device] = None, log_every: int = 0,
    conditions: Optional[List[str]] = None, on_result=None,
) -> List[Dict]:
    """SCADANet Track B, all six conditions (or a subset via `conditions`,
    using the letter codes: A, B, C, D, E, F), all sharing one fit/val/test
    split for this seed.

    Note on delta_f1_vs_full/delta_auprc_vs_full/delta_fpr_vs_full: these
    are computed in-memory after this call's own condition loop finishes,
    so they are only meaningful when Condition C was included in THIS
    call. If you split conditions across multiple CLI calls (e.g.
    `--conditions A,B,C` then later `--conditions D,E,F`) for
    checkpointing/resumability, the second call's saved rows will have
    blank deltas — this is expected, not a bug: experiments/
    build_table_c1_and_figures.py recomputes deltas authoritatively from
    the saved CSV by joining every row against its own seed's
    C_full_model row, so the final table/figures are correct regardless
    of how the run was split across calls."""
    set_seed(seed)
    device = device or _resolve_device()
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)

    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    conditions = conditions or ["A", "B", "C", "D", "E", "F"]
    results = []

    def _emit(r):
        r["dataset"] = "scadanet"
        r["split_protocol"] = "track_b"
        r["notes"]["split_meta"] = split_meta
        r["notes"]["conv_type"] = conv_type
        results.append(r)
        if on_result:
            on_result(r)

    if "A" in conditions:
        _emit(condition_a_content_only(gt.edge_attr, gt.y, df, fit_idx, val_idx, test_idx,
                                        seed, grid, subtype_balanced=True, mlp_epochs=epochs))
    if "B" in conditions:
        _emit(condition_b_topology_only(conv_type, gt.x, gt.edge_index, gt.query_edges, gt.y,
                                         df, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                         subtype_balanced=True, device=device, log_every=log_every))
    if "C" in conditions:
        _emit(condition_c_full(conv_type, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y,
                                df, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                subtype_balanced=True, device=device, log_every=log_every))
    if "D" in conditions:
        _emit(condition_d_random_topology(conv_type, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr,
                                           gt.y, df, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                           subtype_balanced=True, device=device, log_every=log_every))
    if "E" in conditions:
        _emit(condition_e_degree_preserving(conv_type, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr,
                                             gt.y, df, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                             subtype_balanced=True, device=device, log_every=log_every))
    if "F" in conditions:
        _emit(condition_f_no_message_passing(gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y,
                                              df, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                              subtype_balanced=True, device=device, log_every=log_every))

    # Compare metrics with Condition C when it was run.
    full = next((r for r in results if r["condition"] == "C_full_model"), None)
    if full is not None:
        for r in results:
            r["delta_f1_vs_full"] = r["f1"] - full["f1"]
            r["delta_auprc_vs_full"] = (r["auprc"] - full["auprc"]
                                         if r.get("auprc") is not None and full.get("auprc") is not None
                                         else None)
            r["delta_fpr_vs_full"] = r["fpr"] - full["fpr"]

    return results


def run_condition_matrix_temporal(
    seed: int, conv_type: str = "graphsage", csv_path: str = None, n_windows: int = 20,
    epochs: int = 300, val_fraction: float = 0.15, grid: HParamGrid = DEFAULT_GRID,
    device: Optional[torch.device] = None, log_every: int = 0,
    conditions: Optional[List[str]] = None, on_result=None,
) -> List[Dict]:
    """Strict SCADANet temporal protocol, same six conditions.

    Conditions B and C train on the train-boundary topology
    (`snapshots[split_window - 1]`) and are scored on the final test set
    via Issue #5's own `scadanet_window_eval(mode="causal")` — walking
    real per-window topology, no future leak. Conditions D/E (random /
    degree-preserving rewired topology) and F (no message passing) also
    now train on the train-boundary topology instead of the leaky final
    one, but are NOT walked per-window at test time: `scadanet_window_eval`
    reads REAL snapshots internally, and there is no equivalent
    "rewired snapshot per window" built anywhere in this repo, so D/E are
    scored once against the same rewired train-boundary graph for every
    test flow (fixed, not leaky, just not as elaborately time-evolving as
    B/C — a real scoping limitation, not a design choice, flagged in
    README_ISSUE4.md rather than silently narrowed).

    Before a later full-requirements re-audit, EVERY condition here
    (including B and C) trained AND tested against `snapshots[-1]`
    (the fully-streamed final topology) regardless of window — the exact
    future-topology leak Issue #5 was built to fix, silently un-fixed
    here despite this function's own "temporal_strict" label. This
    docstring itself used to say "if Issue #5 changes the temporal
    protocol, update this function... " as a to-do that was never done
    even after Issue #5 shipped in this same repo. See
    README_ISSUE4.md's "Fix applied on review"."""
    set_seed(seed)
    device = device or _resolve_device()
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    df_t, snapshots = add_temporal_windows(df, gt.ip_index, n_windows=n_windows)
    train_idx_np, test_idx_np, split_window = temporal_split_positions(df_t)

    orig_idx = torch.as_tensor(df_t["orig_idx"].to_numpy().copy(), dtype=torch.long)
    query_edges_t = gt.query_edges[:, orig_idx]
    edge_attr_t = gt.edge_attr[orig_idx]
    y_t = gt.y[orig_idx]
    # Train-boundary topology — used for training EVERY condition below
    # (B-F), and for B/C's validation-threshold search too (validation
    # flows are all pre-boundary, so this is causally correct for them).
    train_topology = snapshots[split_window - 1]["edge_index"]
    strict_temporal_test = {"snapshots": snapshots, "split_window": split_window,
                             "n_windows": n_windows}

    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)
    test_idx = torch.as_tensor(test_idx_np, dtype=torch.long)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    conditions = conditions or ["A", "B", "C", "D", "E", "F"]
    results = []

    def _emit(r):
        r["dataset"] = "scadanet"
        r["split_protocol"] = "temporal_strict"
        r["notes"]["n_windows"] = n_windows
        r["notes"]["split_window"] = split_window
        r["notes"]["conv_type"] = conv_type
        results.append(r)
        if on_result:
            on_result(r)

    if "A" in conditions:
        _emit(condition_a_content_only(edge_attr_t, y_t, df_t, fit_idx, val_idx, test_idx,
                                        seed, grid, subtype_balanced=False, mlp_epochs=epochs))
    if "B" in conditions:
        _emit(condition_b_topology_only(conv_type, gt.x, train_topology, query_edges_t, y_t,
                                         df_t, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                         subtype_balanced=False, device=device, log_every=log_every,
                                         strict_temporal_test=strict_temporal_test))
    if "C" in conditions:
        _emit(condition_c_full(conv_type, gt.x, train_topology, query_edges_t, edge_attr_t, y_t,
                                df_t, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                subtype_balanced=False, device=device, log_every=log_every,
                                strict_temporal_test=strict_temporal_test))
    if "D" in conditions:
        r = condition_d_random_topology(conv_type, gt.x, train_topology, query_edges_t, edge_attr_t,
                                         y_t, df_t, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                         subtype_balanced=False, device=device, log_every=log_every)
        r["notes"]["temporal_scoping_limitation"] = (
            "scored once against the rewired train-boundary graph for every test flow, "
            "not walked per-window like Conditions B/C — see run_condition_matrix_temporal docstring")
        _emit(r)
    if "E" in conditions:
        r = condition_e_degree_preserving(conv_type, gt.x, train_topology, query_edges_t, edge_attr_t,
                                           y_t, df_t, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                           subtype_balanced=False, device=device, log_every=log_every)
        r["notes"]["temporal_scoping_limitation"] = (
            "scored once against the rewired train-boundary graph for every test flow, "
            "not walked per-window like Conditions B/C — see run_condition_matrix_temporal docstring")
        _emit(r)
    if "F" in conditions:
        _emit(condition_f_no_message_passing(gt.x, train_topology, query_edges_t, edge_attr_t, y_t,
                                              df_t, fit_idx, val_idx, test_idx, seed, epochs, grid,
                                              subtype_balanced=False, device=device, log_every=log_every))

    full = next((r for r in results if r["condition"] == "C_full_model"), None)
    if full is not None:
        for r in results:
            r["delta_f1_vs_full"] = r["f1"] - full["f1"]
            r["delta_auprc_vs_full"] = (r["auprc"] - full["auprc"]
                                         if r.get("auprc") is not None and full.get("auprc") is not None
                                         else None)
            r["delta_fpr_vs_full"] = r["fpr"] - full["fpr"]

    return results


# ---------------------------------------------------------------------
# Feature counterfactuals (Table C2) — reuse a trained Condition C model,
# perturb edge_attr at inference time only, no retraining.
# ---------------------------------------------------------------------

def run_feature_counterfactuals(
    seed: int = 42, conv_type: str = "graphsage", csv_path: str = None,
    epochs: int = 300, val_fraction: float = 0.15, grid: HParamGrid = DEFAULT_GRID,
    device: Optional[torch.device] = None, log_every: int = 0,
) -> List[Dict]:
    """Trains ONE Condition-C model (real graph + real features), then
    evaluates it under: no perturbation, each of TOP_FEATURES permuted
    alone, all three jointly, and all content features permuted at once
    (item 2's "true topology + meaningless content" complementary test).
    Every row uses the SAME trained model/threshold — only the TEST-set
    edge_attr changes, so this is a real permutation-importance test, not
    a re-trained ablation."""
    set_seed(seed)
    device = device or _resolve_device()
    df = load_scadanet_df(csv_path)
    gt = build_graph_tensors(df)
    train_idx, test_idx, split_meta = scadanet_track_b_split(df, LABEL_COL, seed=seed)
    fit_idx, val_idx = carve_validation(train_idx, val_fraction=val_fraction, seed=seed)

    full_result, model, dev, x_dev, edge_index_dev = condition_c_full(
        conv_type, gt.x, gt.edge_index, gt.query_edges, gt.edge_attr, gt.y, df,
        fit_idx, val_idx, test_idx, seed, epochs, grid, subtype_balanced=True,
        device=device, log_every=log_every, return_model=True,
    )
    threshold = full_result["threshold"]
    test_y = gt.y[test_idx]
    attack_labels = df.iloc[test_idx.numpy()][LABEL_COL].values

    def _eval(edge_attr_variant, label, removed_or_permuted, columns_found):
        probs = _gnn_edge_probs(model, x_dev, edge_index_dev, gt.query_edges, edge_attr_variant, test_idx, dev)
        pred = predict_labels(probs, threshold)
        metrics = compute_metrics(torch.as_tensor(pred), test_y, torch.as_tensor(probs))
        par = per_attack_recall(torch.as_tensor(pred), test_y, attack_labels)
        return {
            "dataset": "scadanet", "split_protocol": "track_b", "seed": seed,
            "conv_type": conv_type, "removed_or_permuted_feature": label,
            "columns_found": columns_found, "per_attack_recall": par,
            "delta_f1": metrics["f1"] - full_result["f1"],
            "delta_auprc": (metrics["auprc"] - full_result["auprc"]
                             if metrics.get("auprc") is not None and full_result.get("auprc") is not None
                             else None),
            "delta_fpr": metrics["fpr"] - full_result["fpr"],
            **metrics,
        }

    results = []
    results.append(_eval(gt.edge_attr, "none", "none", None))

    rng = np.random.default_rng(seed)
    test_idx_np = test_idx.numpy()
    perm_within_test = rng.permutation(len(test_idx_np))

    def _permute_columns(cols: List[int]) -> torch.Tensor:
        ea = gt.edge_attr.clone()
        test_block = ea[test_idx]
        for c in cols:
            test_block[:, c] = test_block[perm_within_test, c]
        ea[test_idx] = test_block
        return ea

    all_found_cols = []
    for feat in TOP_FEATURES:
        cols = find_feature_columns(gt.feature_names, feat)
        if not cols:
            results.append({
                "dataset": "scadanet", "split_protocol": "track_b", "seed": seed,
                "conv_type": conv_type, "removed_or_permuted_feature": feat,
                "columns_found": [], "skipped_reason": "no matching column in this dataset's "
                                                         "feature_names (exact or prefix match)",
            })
            continue
        all_found_cols.extend(cols)
        results.append(_eval(_permute_columns(cols), feat, feat, [gt.feature_names[i] for i in cols]))

    if all_found_cols:
        results.append(_eval(_permute_columns(sorted(set(all_found_cols))), "top-3 jointly",
                              "top-3 jointly", [gt.feature_names[i] for i in sorted(set(all_found_cols))]))

    all_cols = list(range(gt.edge_attr.shape[1]))
    results.append(_eval(_permute_columns(all_cols), "all content shuffled", "all content shuffled", None))

    for r in results:
        r["notes"] = {"split_meta": split_meta, "reference_condition_c_f1": full_result["f1"],
                       "reference_condition_c_model_hyperparams": full_result["best_hyperparams"]}

    return results


# ---------------------------------------------------------------------
# Result artifacts + CLI
# ---------------------------------------------------------------------

def _condition_row(r: Dict) -> Dict:
    return {
        "dataset": r["dataset"], "split_protocol": r["split_protocol"],
        "condition": r.get("condition"), "conv_type": r["notes"].get("conv_type"),
        "seed": r["seed"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
        "auprc": r.get("auprc"), "auroc": r.get("auroc"), "fpr": r["fpr"],
        "delta_f1_vs_full": r.get("delta_f1_vs_full"),
        "delta_auprc_vs_full": r.get("delta_auprc_vs_full"),
        "delta_fpr_vs_full": r.get("delta_fpr_vs_full"),
        "n_trainable_params": r.get("n_trainable_params"),
        "train_runtime_s": r.get("runtime"), "threshold": r.get("threshold"),
    }


def _append_condition_to_csv(r: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([_condition_row(r)])
    header = not out_path.exists()
    row_df.to_csv(out_path, mode="a", header=header, index=False)


def _write_one_condition(r: Dict, json_subdir: Path, csv_path: Path,
                          rewiring_metadata_path: Optional[Path] = None) -> None:
    json_subdir.mkdir(parents=True, exist_ok=True)
    fname = f"{r['dataset']}_{r['split_protocol']}_{r['condition']}_seed{r['seed']}.json"
    write_result(dict(r), str(json_subdir / fname))
    _append_condition_to_csv(r, csv_path)
    if rewiring_metadata_path and "rewiring_metadata" in r.get("notes", {}):
        _append_rewiring_metadata(r, rewiring_metadata_path)
    print(f"[ablation_runner] checkpointed condition {r['condition']} (seed {r['seed']}) "
          f"-> {json_subdir / fname} and {csv_path}")


def _append_rewiring_metadata(r: Dict, path: Path) -> None:
    """Issue #4 item 6: save rewiring metadata (node/edge counts, degree
    summary, connected components, seed) for every D/E condition run, so
    a bad Condition D/E result is checkable against "did the rewiring
    actually preserve graph size" rather than taken on faith."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "dataset": r["dataset"], "split_protocol": r["split_protocol"],
        "condition": r["condition"], "seed": r["seed"],
        "conv_type": r["notes"].get("conv_type"),
        **r["notes"]["rewiring_metadata"],
    }
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2, default=str))


def _feature_counterfactuals_to_csv(results: List[Dict], out_path: Path) -> None:
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"], "split_protocol": r["split_protocol"], "seed": r["seed"],
            "conv_type": r["conv_type"], "removed_or_permuted_feature": r["removed_or_permuted_feature"],
            "columns_found": str(r.get("columns_found")),
        }
        if "skipped_reason" in r:
            row["skipped_reason"] = r["skipped_reason"]
        else:
            row.update({"precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                        "auprc": r.get("auprc"), "fpr": r["fpr"],
                        "delta_f1": r["delta_f1"], "delta_auprc": r.get("delta_auprc"),
                        "delta_fpr": r["delta_fpr"]})
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_path.exists():
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[ablation_runner] wrote {out_path} ({len(df)} rows total)")


def main():
    parser = argparse.ArgumentParser(description="Issue #4 ablation experiments.")
    parser.add_argument("--experiment", required=True,
                         choices=["c1_trackb", "c1_temporal", "c2_features"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv-type", default="graphsage", choices=["graphsage", "gcn", "gat"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--conditions", default=None,
                         help="Comma-separated subset of A,B,C,D,E,F to run this call "
                              "(e.g. --conditions A,B,C). Default: all six. Results are "
                              "checkpointed after each condition, so a long run can be "
                              "split/resumed across cells the same way as Issue #3's "
                              "--families flag.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[ablation_runner] using device: {device}")
    conditions = args.conditions.split(",") if args.conditions else None

    kwargs = {"seed": args.seed, "conv_type": args.conv_type, "device": device,
              "log_every": args.log_every}
    if args.data_path:
        kwargs["csv_path"] = args.data_path
    if args.epochs:
        kwargs["epochs"] = args.epochs

    if args.experiment == "c1_trackb":
        csv_path = RESULTS_DIR / "scadanet_trackb_topology_ablation.csv"
        json_subdir = RESULTS_DIR / "c1_trackb"
        rewiring_path = RESULTS_DIR / "rewiring_metadata.json"
        kwargs["conditions"] = conditions
        kwargs["on_result"] = lambda r: _write_one_condition(r, json_subdir, csv_path, rewiring_path)
        run_condition_matrix_trackb(**kwargs)
    elif args.experiment == "c1_temporal":
        csv_path = RESULTS_DIR / "scadanet_temporal_topology_ablation.csv"
        json_subdir = RESULTS_DIR / "c1_temporal"
        rewiring_path = RESULTS_DIR / "rewiring_metadata.json"
        kwargs["conditions"] = conditions
        kwargs["on_result"] = lambda r: _write_one_condition(r, json_subdir, csv_path, rewiring_path)
        run_condition_matrix_temporal(**kwargs)
    else:
        results = run_feature_counterfactuals(**kwargs)
        _feature_counterfactuals_to_csv(results, RESULTS_DIR / "feature_counterfactuals.csv")
        json_subdir = RESULTS_DIR / "c2_features"
        json_subdir.mkdir(parents=True, exist_ok=True)
        for r in results:
# Inference-only perturbations do not have the fields of a training run.
            if "skipped_reason" not in r:
                fname = f"feature_{r['removed_or_permuted_feature'].replace(' ', '_')}_seed{r['seed']}.json"
                (json_subdir / fname).write_text(json.dumps(r, indent=2, default=str))

    print(f"[ablation_runner] experiment {args.experiment} complete.")


if __name__ == "__main__":
    main()
