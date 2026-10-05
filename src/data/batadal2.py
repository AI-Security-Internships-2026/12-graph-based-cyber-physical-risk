"""BATADAL 2.0 (Erba et al., RICSS 2024; Zenodo 13692004) loading and cyber-physical graphs.

The release simulates C-Town with DHALSIM: 52 + 1 normal runs and 33 attack runs, 2,880
steps of 5 minutes each. Every run folder holds `scada_values.csv` (the 41 tags the SCADA
sees, falsified during concealment attacks), `ground_truth.csv` (the true physical state
plus one 0/1 column per attack) and `configuration/` (config.yaml with the PLC layout and
the attacks, map.inp with the EPANET control rules).

Unlike BATADAL v1, the PLC layout is known, so the graph can be the plant's actual
cyber-physical structure instead of a correlation estimate:

* "plc": components wired to the same PLC are connected, and each tank connects to the
  actuators its level switches (the [CONTROLS] section of map.inp, e.g. T1 -> PU1/PU2).
  Those control loops cross PLC boundaries, i.e. they run over the network.
* "correlation": BATADAL v1's Pearson graph, fitted on training-normal rows only.
* "none": no edges (the model sees each node alone), the topology ablation.
"""
import glob
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

from src.data.batadal import build_correlation_graph

WINDOW_SIZE = 36  # steps of 5 minutes, 3 hours
STRIDE = 12       # 1 hour
_CONTROL = re.compile(r"^\s*LINK\s+(\S+)\s+\S+\s+IF\s+NODE\s+(\S+)", re.IGNORECASE)


def discover_runs(root: str) -> pd.DataFrame:
    """One row per run folder: path, kind (normal/attack), attack signature and flags."""
    rows = []
    patterns = [("normal", "normal_operating_conditions_3_0/output/batch_*"),
                ("normal", "normal_operating_conditions_one_week/output_*"),
                ("attack", "evasion_data/attack_output_*")]
    for kind, pattern in patterns:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            if not os.path.exists(os.path.join(path, "scada_values.csv")):
                continue
            row = {"run": os.path.relpath(path, root), "path": path, "kind": kind,
                   "signature": os.path.relpath(path, root), "targets": "", "concealment": False}
            if kind == "attack":
                attacks = _read_config(path).get("attacks") or []
                attacks = attacks if isinstance(attacks, list) else attacks.get("network_attacks", [])
                # Runs repeated with the same attacks (e.g. 01/02) share a signature, so a
                # grouped split never puts near-duplicates on both sides.
                row["signature"] = "|".join(sorted(f"{a.get('name')}:{a.get('type')}:{a.get('target')}:"
                                                   f"{(a.get('trigger') or {}).get('start')}-"
                                                   f"{(a.get('trigger') or {}).get('end')}" for a in attacks))
                row["targets"] = ",".join(sorted({str(a.get("target")) for a in attacks if a.get("target")}))
                row["concealment"] = (any("conceal" in str(a.get("type", "")) for a in attacks)
                                      or os.path.exists(os.path.join(path, "input_concealment_values.csv")))
            rows.append(row)
    return pd.DataFrame(rows)


def load(root: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """(runs, frames) for the window runner."""
    runs = discover_runs(root)
    return runs, {r.run: load_run(r.path) for r in runs.itertuples()}


def _read_config(run_path: str) -> dict:
    with open(os.path.join(run_path, "configuration", "config.yaml")) as f:
        return yaml.safe_load(f)


def load_run(run_path: str) -> pd.DataFrame:
    """SCADA tags of one run plus ATT_FLAG (1 if any attack column of the ground truth is set)."""
    scada = pd.read_csv(os.path.join(run_path, "scada_values.csv"))
    # A few runs log some iterations twice seconds apart (keep the later SCADA poll), and
    # attack_output_13 has two rows without an iteration number.
    scada = (scada.dropna(subset=["iteration"]).sort_values("timestamp", kind="stable")
                  .drop_duplicates("iteration", keep="last").sort_values("iteration"))
    truth = pd.read_csv(os.path.join(run_path, "ground_truth.csv"),
                        usecols=lambda c: c == "iteration" or "attack" in c.lower())
    label_cols = [c for c in truth.columns if c != "iteration"]
    truth["ATT_FLAG"] = truth[label_cols].max(axis=1).astype(int) if label_cols else 0
    df = scada.merge(truth[["iteration", "ATT_FLAG"]], on="iteration", how="left", validate="one_to_one")
    if df["ATT_FLAG"].isna().any():
        raise ValueError(f"{run_path}: SCADA iterations missing from the ground truth")
    df = df.drop(columns=["iteration", "timestamp"]).astype({"ATT_FLAG": int})
    # attack_output_14 misses 10 polls; hold the last reading, as a SCADA historian would.
    df = df.ffill().bfill()
    if df.isna().any().any():
        raise ValueError(f"{run_path}: missing SCADA values remain after forward fill")
    return df


def plc_layout(run_path: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """(component -> PLC, control edges tank -> actuator) from config.yaml and map.inp."""
    owner = {}
    for plc in _read_config(run_path).get("plcs", []):
        for tag in (plc.get("sensors") or []) + (plc.get("actuators") or []):
            owner[tag] = plc["name"]
    controls, in_controls = [], False
    with open(os.path.join(run_path, "configuration", "map.inp")) as f:
        for line in f:
            if line.strip().startswith("["):
                in_controls = line.strip().upper() == "[CONTROLS]"
                continue
            match = _CONTROL.match(line) if in_controls else None
            if match:
                actuator, tank = match.groups()
                controls.append((tank, actuator))
    return owner, sorted(set(controls))


def plc_graph(components: Sequence[str], owner: Dict[str, str],
              controls: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    present = set(components)
    by_plc: Dict[str, List[str]] = {}
    for c in components:
        by_plc.setdefault(owner.get(c, "unknown"), []).append(c)
    edges = {(m[i], m[j]) for m in by_plc.values() for i in range(len(m)) for j in range(i + 1, len(m))}
    edges |= {(t, a) for t, a in controls if t in present and a in present}
    return sorted(edges)


def component_type(tag: str) -> str:
    if re.fullmatch(r"T\d+", tag):
        return "Tank"
    if re.fullmatch(r"(PU|V)\d+F", tag):
        return "Flow"
    if re.fullmatch(r"(PU|V)\d+", tag):
        return "Actuator_Status"
    if re.fullmatch(r"J\d+", tag):
        return "Pressure"
    return "Unknown"


def build_dataset(root: str, runs: pd.DataFrame, fit_runs: Sequence[str], topology: str = "plc",
                  window_size: int = WINDOW_SIZE, stride: int = STRIDE,
                  frames: Dict[str, pd.DataFrame] = None, seed: int = 0) -> Tuple[List[Data], Dict]:
    """Window graphs for every run; windows never cross runs. Normalisation, z-score
    baseline and the correlation topology use only the normal rows of `fit_runs`."""
    frames = frames or {r.run: load_run(r.path) for r in runs.itertuples()}
    components = sorted(c for c in next(iter(frames.values())).columns if c != "ATT_FLAG")
    fit = pd.concat([frames[r] for r in fit_runs], ignore_index=True)
    normal = fit.loc[fit["ATT_FLAG"] == 0, components]
    types = {c: component_type(c) for c in components}

    # Normal batches ship without config.yaml; the plant is the same in every run.
    owner, controls = plc_layout(runs.loc[runs.kind == "attack"].iloc[0].path)
    if topology == "plc":
        pairs = plc_graph(components, owner, controls)
    elif topology == "correlation":
        sample = normal.sample(min(len(normal), 50_000), random_state=seed)
        pairs = list(build_correlation_graph(sample, components, types).edges())
    elif topology == "none":
        pairs = []
    else:
        raise ValueError("topology must be plc, correlation or none")
    index = {c: i for i, c in enumerate(components)}
    edges = [(index[u], index[v]) for u, v in pairs]
    edges += [(v, u) for u, v in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)

    type_codes = {t: i for i, t in enumerate(sorted(set(types.values())))}
    plc_codes = {p: i for i, p in enumerate(sorted(set(owner.values())))}
    static = np.array([[type_codes[types[c]], plc_codes.get(owner.get(c), -1)] for c in components], float)
    graphs, info = window_graphs(runs, frames, components, fit, normal, static, edge_index, window_size, stride)
    info.update({"n_edges": len(pairs), "topology": topology})
    return graphs, info


def window_graphs(runs: pd.DataFrame, frames: Dict[str, pd.DataFrame], components: Sequence[str],
                  fit: pd.DataFrame, normal: pd.DataFrame, static: np.ndarray, edge_index: torch.Tensor,
                  window_size: int, stride: int) -> Tuple[List[Data], Dict]:
    """Sliding-window graphs within each run (windows never cross runs).

    Node features: window mean and std scaled by the training range, the window mean's
    z-score against the training-normal baseline (clipped to +-5), then the `static`
    per-node columns. A window is an attack window if any row in it is."""
    vmin = fit[components].min().to_numpy(np.float64)
    span = fit[components].max().to_numpy(np.float64) - vmin + 1e-9
    base_mean = normal.mean().to_numpy(np.float64)
    base_std = normal.std().replace(0, 1e-9).fillna(1).to_numpy(np.float64)
    graphs, meta = [], []
    for r in runs.itertuples():
        values = frames[r.run][components].to_numpy(np.float64)
        attack = frames[r.run]["ATT_FLAG"].to_numpy(np.int64)
        starts = np.arange(0, len(values) - window_size + 1, stride)
        ends = starts + window_size
        zero = np.zeros((1, values.shape[1]))
        c1 = np.vstack([zero, np.cumsum(values, 0)])
        c2 = np.vstack([zero, np.cumsum(values ** 2, 0)])
        ca = np.concatenate([[0], np.cumsum(attack)])
        w_sum = c1[ends] - c1[starts]
        w_mean = w_sum / window_size
        w_std = np.sqrt(np.clip((c2[ends] - c2[starts] - w_sum * w_mean) / (window_size - 1), 0, None))
        labels = ((ca[ends] - ca[starts]) > 0).astype(int)
        feats = np.stack([(w_mean - vmin) / span, w_std / span,
                          np.clip((w_mean - base_mean) / base_std, -5, 5)], axis=-1)
        feats = np.concatenate([feats, np.broadcast_to(static, feats.shape[:2] + (static.shape[1],))], axis=-1)
        x = torch.tensor(feats, dtype=torch.float32)
        for i in range(len(starts)):
            graphs.append(Data(x=x[i], edge_index=edge_index, y=torch.tensor([int(labels[i])])))
            meta.append((r.run, int(starts[i])))
    info = {"components": list(components),
            "window_runs": np.array([m[0] for m in meta]), "window_starts": np.array([m[1] for m in meta])}
    return graphs, info


def grouped_folds(runs: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per run. Attack runs are grouped by signature and normal runs spread
    separately, so every fold holds both, and repeated scenarios never straddle folds."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(runs), -1)
    for kind in ("attack", "normal"):
        groups = np.array(sorted(runs.loc[runs.kind == kind, "signature"].unique()), dtype=object)
        rng.shuffle(groups)
        for i, g in enumerate(groups):
            fold[(runs.kind == kind).to_numpy() & (runs.signature == g).to_numpy()] = i % n_folds
    return fold
