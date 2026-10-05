"""WaDi.A2 (iTrust water distribution testbed, 19 Nov 2019 release) as grouped segments.

Files (iTrust release): WADI_14days_new.csv (14 days of normal operation, 1 Hz, unstable
periods already removed by iTrust) and WADI_attackdataLABLE.csv (2 days with 15 attacks,
label column "Attack LABLE (1:No Attack, -1:Attack)").

The normal period holds no attacks, so supervised detection needs attack data in training.
Both files are cut into segments that the cross-validation keeps whole:
  * normal: one segment per calendar day;
  * attack period: one segment per attack event, cut midway between consecutive events,
    so every attack-period row (normal or attack) belongs to exactly one event's segment.
This mirrors BATADAL 2.0's scenario-held-out protocol: no attack event, and no day, appears
on both sides of a split.

Topologies: "stage" (tags of the same process stage, i.e. the prefix 1/2/2A/2B/3 that maps
to the stage's PLC, are connected), "correlation" (BATADAL v1's Pearson graph fitted on
training-normal rows), "none".
"""
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.batadal import build_correlation_graph
from src.data.batadal2 import grouped_folds, window_graphs  # noqa: F401  (grouped_folds re-exported)

WINDOW_SIZE = 120  # rows at 1 Hz: 2 minutes
STRIDE = 60
NORMAL_FILE = "WADI_14days_new.csv"
ATTACK_FILE = "WADI_attackdataLABLE.csv"
LABEL = "Attack LABLE (1:No Attack, -1:Attack)"
META = ["Row", "Date", "Time"]
TOPOLOGIES = {"graphsage_stage": "stage", "graphsage_correlation": "correlation",
              "graphsage_none": "none", "mlp": "stage"}  # the MLP ignores edges


def _read(path: str, header: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=header, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(subset=["Date"]).reset_index(drop=True)  # the attack file ends with empty rows


def _components(normal: pd.DataFrame, attack: pd.DataFrame) -> List[str]:
    """Sensor/actuator columns present in both files and not empty in either.

    iTrust's A2 files leave a few tags (e.g. 2_LS_001_AL, 2_P_001_STATUS) entirely empty."""
    shared = [c for c in normal.columns if c in attack.columns and c not in META + [LABEL]]
    return [c for c in shared if normal[c].notna().any() and attack[c].notna().any()]


def _attack_events(flag: np.ndarray) -> List[Tuple[int, int]]:
    """(start, end) index pairs of contiguous attack runs, end exclusive."""
    edges = np.diff(np.concatenate([[0], flag, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_frames(root: str, runs: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """Segment frames keyed by run name: component columns plus ATT_FLAG (0/1)."""
    normal = _read(os.path.join(root, NORMAL_FILE), header=0)
    attack = _read(os.path.join(root, ATTACK_FILE), header=1)
    comps = _components(normal, attack)
    frames = {}

    def clean(df, flag):
        out = df[comps].apply(pd.to_numeric, errors="coerce").ffill().bfill().astype(np.float32)
        out["ATT_FLAG"] = flag
        if out.isna().any().any():
            raise ValueError("WaDi: missing values remain after forward fill")
        return out.reset_index(drop=True)

    for day, part in normal.groupby(normal["Date"].astype(str).str.strip(), sort=False):
        frames[f"normal/{pd.to_datetime(day, dayfirst=False).date()}"] = clean(part, 0)

    flag = (pd.to_numeric(attack[LABEL]) == -1).astype(int).to_numpy()
    events = _attack_events(flag)
    cuts = [0] + [(events[i][1] + events[i + 1][0]) // 2 for i in range(len(events) - 1)] + [len(attack)]
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        frames[f"attack/event_{i + 1:02d}"] = clean(attack.iloc[a:b], flag[a:b])
    return frames


def discover_runs(root: str, frames: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    frames = frames or load_frames(root)
    rows = []
    for name, f in frames.items():
        kind = name.split("/")[0]
        rows.append({"run": name, "path": root, "kind": kind, "signature": name,
                     "targets": "", "concealment": False, "n_rows": len(f), "attack_rows": int(f.ATT_FLAG.sum())})
    return pd.DataFrame(rows)


def load(root: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """(runs, frames) for the window runner."""
    frames = load_frames(root)
    return discover_runs(root, frames), frames


def stage(tag: str) -> str:
    """Process stage from the tag prefix: 1, 2, 2A, 2B or 3."""
    m = re.match(r"^(\d[AB]?)_", tag)
    return m.group(1) if m else "other"


def component_type(tag: str) -> str:
    m = re.match(r"^\d[AB]?_([A-Z]+)_", tag)
    return m.group(1) if m else "other"


def stage_graph(components: Sequence[str]) -> List[Tuple[str, str]]:
    by_stage: Dict[str, List[str]] = {}
    for c in components:
        by_stage.setdefault(stage(c), []).append(c)
    return [(m[i], m[j]) for m in by_stage.values() for i in range(len(m)) for j in range(i + 1, len(m))]


def build_dataset(root: str, runs: pd.DataFrame, fit_runs: Sequence[str], topology: str = "stage",
                  window_size: int = WINDOW_SIZE, stride: int = STRIDE,
                  frames: Dict[str, pd.DataFrame] = None, seed: int = 0) -> Tuple[List[Data], Dict]:
    """Window graphs for every segment; statistics and the correlation graph use only the
    normal rows of `fit_runs`."""
    frames = frames or load_frames(root)
    components = sorted(c for c in next(iter(frames.values())).columns if c != "ATT_FLAG")
    fit = pd.concat([frames[r] for r in fit_runs], ignore_index=True)
    normal = fit.loc[fit["ATT_FLAG"] == 0, components]
    if topology == "stage":
        pairs = stage_graph(components)
    elif topology == "correlation":
        sample = normal.sample(min(len(normal), 50_000), random_state=seed)
        pairs = list(build_correlation_graph(sample, components, {c: component_type(c) for c in components}).edges())
    elif topology == "none":
        pairs = []
    else:
        raise ValueError("topology must be stage, correlation or none")
    index = {c: i for i, c in enumerate(components)}
    edges = [(index[u], index[v]) for u, v in pairs]
    edges += [(v, u) for u, v in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    types = sorted({component_type(c) for c in components})
    stages = sorted({stage(c) for c in components})
    static = np.array([[types.index(component_type(c)), stages.index(stage(c))] for c in components], float)
    graphs, info = window_graphs(runs, frames, components, fit, normal, static, edge_index, window_size, stride)
    info.update({"n_edges": len(pairs), "topology": topology})
    return graphs, info
