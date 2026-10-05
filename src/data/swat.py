"""SWaT (iTrust Secure Water Treatment, A1 Dec 2015) loading and sensor-graph windows.

Mirrors `src.data.batadal`: sensors/actuators are graph nodes, sliding windows are
classification samples, and every statistic (correlation topology, normalization,
baseline z-scores) is fitted on training rows only.

Two sources are supported:

* official iTrust release - `load_swat_official(normal_path, attack_path)`, taking the
  normal-week file (use version 1, which drops the 30-minute tank-draining period) and the
  attack-period file, as .csv or .xlsx.
* the unofficial Kaggle copy (vishala28/swat-dataset-secure-water-treatment-system) -
  `load_swat_kaggle_merged(path)`. Its merged.csv stacks normal v0, normal v1 (the same week
  again, ~495k byte-identical rows) and the attack period, out of time order. The loader
  rebuilds the official v1 + attack layout (944,919 rows). Its normal week also lacks seven
  components, so the result has 44 of the 51. Use this copy for development only; report
  results from the official release.

Every loader returns one chronological frame with columns DATETIME, the 51 sensor and
actuator columns, ATT_FLAG (0/1) and PHASE ("normal_week" / "attack_period").
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.batadal import build_correlation_graph
from src.utils.reproducibility import replayable_split

WINDOW_SIZE = 120  # rows; SWaT samples at 1 Hz, so 2 minutes
STRIDE = 30        # rows
DRAIN_SECONDS = 30 * 60  # tank-draining period at the start of normal v0
META_COLS = ["DATETIME", "ATT_FLAG", "PHASE"]
A1_ATTACK_PERIOD_START = pd.Timestamp("2015-12-28 10:00:00")  # first row of the A1 attack file

# Name prefix -> component type. Longest prefixes first so DPIT is not read as PIT.
_TYPE_PREFIXES = [("DPIT", "Diff_Pressure"), ("FIT", "Flow"), ("LIT", "Level"),
                  ("AIT", "Analyzer"), ("PIT", "Pressure"), ("MV", "Motorized_Valve"),
                  ("UV", "UV_Dechlorinator"), ("P", "Pump")]


def get_sensor_type(col: str) -> str:
    """Classify a SWaT column (e.g. FIT101, MV201, P602) by its tag prefix."""
    for prefix, kind in _TYPE_PREFIXES:
        if re.fullmatch(prefix + r"\d+", col):
            return kind
    return "Unknown"


def get_stage(col: str) -> int:
    """Process stage P1-P6, the first digit of the tag number (FIT101 -> 1, P602 -> 6)."""
    match = re.search(r"(\d)\d\d$", col)
    return int(match.group(1)) if match else 0


def _standardize(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Strip headers, parse timestamps, map the label to ATT_FLAG and tag the phase."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns={"Timestamp": "DATETIME"})
    df["DATETIME"] = pd.to_datetime(df["DATETIME"].astype(str).str.strip(), dayfirst=True, format="mixed")
    # The official attack file spells some labels "A ttack"; remove all whitespace.
    label = df.pop("Normal/Attack").astype(str).str.replace(r"\s+", "", regex=True)
    unknown = set(label.unique()) - {"Normal", "Attack"}
    if unknown:
        raise ValueError(f"Unexpected SWaT labels: {sorted(unknown)}")
    df["ATT_FLAG"] = (label == "Attack").astype(int)
    df["PHASE"] = phase
    return df


def _read_any(path: str) -> pd.DataFrame:
    """Read .csv/.xlsx; the official .xlsx files have a title row above the header."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, header=None)
        header_row = next(i for i in range(min(5, len(raw)))
                          if raw.iloc[i].astype(str).str.strip().eq("Normal/Attack").any())
        df = raw.iloc[header_row + 1:].reset_index(drop=True)
        df.columns = raw.iloc[header_row].astype(str).str.strip()
        return df
    return pd.read_csv(path, low_memory=False)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("DATETIME", kind="stable").reset_index(drop=True)
    if df["DATETIME"].duplicated().any():
        raise ValueError("SWaT frame still has duplicate timestamps after cleaning")
    sensor_cols = [c for c in df.columns if c not in META_COLS]
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="raise").astype(np.float32)
    missing = df[sensor_cols].isna().sum()
    if missing.any():
        # A NaN would poison every later window through the prefix sums; fail loudly instead.
        raise ValueError(f"SWaT has missing sensor values: {missing[missing > 0].to_dict()}")
    return df[["DATETIME"] + sensor_cols + ["ATT_FLAG", "PHASE"]]


def load_swat_official(normal_path: str, attack_path: str,
                       drop_drain_period: bool = True) -> pd.DataFrame:
    """Official iTrust A1 files. Pass normal v1; with v0, `drop_drain_period` removes the
    first 30 minutes so the result matches v1."""
    normal = _standardize(_read_any(normal_path), "normal_week")
    attack = _standardize(_read_any(attack_path), "attack_period")
    if drop_drain_period:
        normal = normal[normal["DATETIME"] >= normal["DATETIME"].min() + pd.Timedelta(seconds=DRAIN_SECONDS)]
    return _finalize(pd.concat([normal, attack], ignore_index=True))


def load_swat_kaggle_merged(path: str, drop_columns_missing_in_normal: bool = True) -> pd.DataFrame:
    """Kaggle merged.csv -> official layout (normal v1 + attack period).

    Rows repeated across normal v0/v1 are dropped, as is the drain period present only in
    v0. Rows from the first timestamp of the official attack file on are tagged
    attack_period. Components with no normal-week values in this copy are dropped (listed in
    `df.attrs["dropped_columns"]`), leaving 44 of the 51."""
    df = _standardize(pd.read_csv(path, low_memory=False), "normal_week")
    # merged.csv stores normal v0 before v1, so keeping the last copy keeps v1.
    df = df.drop_duplicates(subset="DATETIME", keep="last")
    start = df["DATETIME"].min()
    df = df[df["DATETIME"] >= start + pd.Timedelta(seconds=DRAIN_SECONDS)]
    df.loc[df["DATETIME"] >= A1_ATTACK_PERIOD_START, "PHASE"] = "attack_period"
    if df.loc[df["PHASE"] == "normal_week", "ATT_FLAG"].any():
        raise ValueError("Attack rows found before the attack period; check the source file")
    dropped = []
    if drop_columns_missing_in_normal:
        # Both normal-week copies in the Kaggle file leave seven components (MV101, AIT201,
        # MV201, P201, P202, P204, MV303) empty; the official files have them.
        normal = df.loc[df["PHASE"] == "normal_week"]
        dropped = [c for c in df.columns if c not in META_COLS and normal[c].isna().all()]
        df = df.drop(columns=dropped)
    df = _finalize(df)
    df.attrs["dropped_columns"] = dropped
    return df


def sensor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def build_process_graph(sensor_cols: Sequence[str]) -> List[Tuple[str, str]]:
    """Design-prior topology: components in the same stage are connected, and each stage
    connects to the next (P1 -> P2 -> ... -> P6) through all component pairs across the two
    stages. Approximates the plant's water flow without the P&ID; refine with the official
    P&ID once available."""
    by_stage: Dict[int, List[str]] = {}
    for col in sensor_cols:
        by_stage.setdefault(get_stage(col), []).append(col)
    edges = []
    stages = sorted(s for s in by_stage if s > 0)
    for s in stages:
        members = by_stage[s]
        edges += [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
    for a, b in zip(stages, stages[1:]):
        edges += [(u, v) for u in by_stage[a] for v in by_stage[b]]
    return edges


def build_windowed_graphs(df: pd.DataFrame, window_size: int = WINDOW_SIZE, stride: int = STRIDE,
                          fit_rows=None, topology: str = "correlation",
                          correlation_threshold: float = 0.7,
                          correlation_sample: Optional[int] = 50_000,
                          seed: int = 0) -> Tuple[List[Data], np.ndarray, Dict]:
    """Sliding windows over the sensor columns with a fixed topology.

    Node features follow BATADAL: per-window mean and std scaled by the training range,
    the window mean's z-score against the training-normal baseline (clipped to +-5), and
    the component-type code; plus the stage number. `topology` is "correlation" (BATADAL's
    Pearson graph fitted on training-normal rows, at most `correlation_sample` of them) or
    "process" (`build_process_graph`). A window is an attack window if any row in it is an
    attack row.
    """
    sensors = sorted(sensor_columns(df))
    index = {s: i for i, s in enumerate(sensors)}
    types = {s: get_sensor_type(s) for s in sensors}
    type_codes = {t: i for i, t in enumerate(sorted(set(types.values())))}

    fit_df = df if fit_rows is None else df.iloc[np.asarray(fit_rows, dtype=int)]
    normal = fit_df.loc[fit_df["ATT_FLAG"] == 0, sensors]
    if len(normal) < 2:
        raise ValueError("SWaT needs at least two normal training rows for its baseline")

    if topology == "correlation":
        sample = normal
        if correlation_sample and len(normal) > correlation_sample:
            sample = normal.sample(correlation_sample, random_state=seed).sort_index()
        graph = build_correlation_graph(sample, sensors, types, threshold=correlation_threshold)
        pairs = list(graph.edges())
    elif topology == "process":
        pairs = build_process_graph(sensors)
    else:
        raise ValueError("topology must be 'correlation' or 'process'")
    edges = [(index[u], index[v]) for u, v in pairs]
    edges += [(v, u) for u, v in edges]
    edge_index = (torch.tensor(edges, dtype=torch.long).T if edges
                  else torch.empty((2, 0), dtype=torch.long))

    vmin = fit_df[sensors].min().to_numpy(np.float64)
    span = fit_df[sensors].max().to_numpy(np.float64) - vmin + 1e-9
    base_mean = normal.mean().to_numpy(np.float64)
    base_std = normal.std().replace(0, 1e-9).fillna(1).to_numpy(np.float64)
    static = np.array([[type_codes[types[s]], get_stage(s)] for s in sensors], dtype=np.float64)

    values = df[sensors].to_numpy(np.float64)
    attack = df["ATT_FLAG"].to_numpy(np.int64)
    starts = np.arange(0, len(df) - window_size + 1, stride)
    # Prefix sums give every window's mean, std and attack count without a Python loop.
    zero = np.zeros((1, values.shape[1]))
    c1 = np.vstack([zero, np.cumsum(values, axis=0)])
    c2 = np.vstack([zero, np.cumsum(values ** 2, axis=0)])
    ca = np.concatenate([[0], np.cumsum(attack)])
    ends = starts + window_size
    w_sum = c1[ends] - c1[starts]
    w_mean = w_sum / window_size
    w_var = (c2[ends] - c2[starts] - w_sum * w_mean) / (window_size - 1)
    w_std = np.sqrt(np.clip(w_var, 0, None))
    labels = ((ca[ends] - ca[starts]) > 0).astype(int)

    feats = np.stack([(w_mean - vmin) / span, w_std / span,
                      np.clip((w_mean - base_mean) / base_std, -5, 5)], axis=-1)
    feats = np.concatenate([feats, np.broadcast_to(static, feats.shape[:2] + (2,))], axis=-1)
    feats = torch.tensor(feats, dtype=torch.float32)

    graphs = [Data(x=feats[i], edge_index=edge_index, y=torch.tensor([int(labels[i])]))
              for i in range(len(starts))]
    phase = df["PHASE"].to_numpy()
    extra = {"starts": starts.tolist(), "sensor_list": sensors, "window_size": window_size,
             "stride": stride, "topology": topology, "n_edges": len(pairs),
             "window_phase": [phase[s] if phase[s] == phase[s + window_size - 1] else "boundary"
                              for s in starts]}
    return graphs, labels, extra


def standard_protocol_rows(df: pd.DataFrame, val_frac: float = 0.1) -> Dict[str, np.ndarray]:
    """The standard SWaT protocol: fit on the normal week, test on the attack period. The
    last `val_frac` of the normal week is held out as a normal-only validation tail."""
    normal_idx = np.flatnonzero(df["PHASE"].to_numpy() == "normal_week")
    attack_idx = np.flatnonzero(df["PHASE"].to_numpy() == "attack_period")
    cut = int(len(normal_idx) * (1 - val_frac))
    return {"fit": normal_idx[:cut], "val": normal_idx[cut:], "test": attack_idx}


def temporal_split_windows(window_graphs: List[Data], split_frac: float = 0.7
                           ) -> Tuple[List[Data], List[Data]]:
    """Chronological split on windows, as in `src.data.batadal`. Windows overlap by
    `window_size - stride` rows, so neighbours across the boundary share raw rows."""
    train_idx, test_idx = reference_window_positions(len(window_graphs), split_frac)
    return [window_graphs[i] for i in train_idx], [window_graphs[i] for i in test_idx]


@replayable_split
def reference_window_positions(n_windows, split_frac):
    split_idx = int(n_windows * split_frac)
    return np.arange(split_idx), np.arange(split_idx, n_windows)
