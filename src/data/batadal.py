"""BATADAL loading, correlation-graph topology, sliding-window graph construction.

Data source: BATADAL_dataset04.csv, downloaded manually from
http://www.batadal.net/data.html and kept in a local data folder (this is
the "not on Kaggle" dataset — pass its path explicitly, there is no
kagglehub fallback for it).
"""
from typing import Dict, List, Tuple
from src.utils.reproducibility import replayable_split

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch_geometric.data import Data

WINDOW_SIZE = 24   # hours
STRIDE = 6         # hours
CORRELATION_THRESHOLD = 0.7


def load_batadal_df(csv_path: str = "BATADAL_dataset04.csv") -> pd.DataFrame:
    """Load, strip columns, fix ATT_FLAG (-999 -> 0)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["ATT_FLAG"] = df["ATT_FLAG"].replace(-999, 0)
    return df


def get_sensor_type(col: str) -> str:
    """Classify a BATADAL sensor column by its name prefix."""
    if col.startswith("L_T"):
        return "Tank"
    if col.startswith("F_PU"):
        return "Pump_Flow"
    if col.startswith("F_V"):
        return "Valve_Flow"
    if col.startswith("P_J"):
        return "Pressure"
    if col.startswith("S_PU"):
        return "Pump_Status"
    if col.startswith("S_V"):
        return "Valve_Status"
    return "Unknown"


def build_correlation_graph(data: pd.DataFrame, sensor_cols: List[str],
                             sensor_type_map: Dict[str, str],
                             threshold: float = CORRELATION_THRESHOLD) -> nx.Graph:
    """Pearson correlation graph over sensor columns, edge kept if |corr| >= threshold and
    p < 0.05.
    """
    G = nx.Graph()
    for col in sensor_cols:
        G.add_node(col, sensor_type=sensor_type_map[col])

    cols = data.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_i, col_j = cols[i], cols[j]
            if data[col_i].std() == 0 or data[col_j].std() == 0:
                continue
            corr, pval = pearsonr(data[col_i].fillna(0), data[col_j].fillna(0))
            if abs(corr) >= threshold and pval < 0.05:
                G.add_edge(col_i, col_j, weight=round(abs(corr), 3), correlation=round(corr, 3))
    return G


def build_windowed_graphs(df: pd.DataFrame, window_size: int = WINDOW_SIZE,
                           stride: int = STRIDE, fit_rows=None) -> Tuple[List[Data], np.ndarray, Dict]:
    """Sliding 24h/6h-stride windows over the sensor columns. Topology is FIXED from the
    normal-period correlation graph (24 points/window is too few to re-estimate
    correlation reliably per window); node features vary per window: mean, std, and
    z-score deviation from each sensor's own normal-period baseline.

    Returns (window_graphs, window_labels, extra) where extra holds `starts` and
    `sensor_list`.
    """
    sensor_cols = [c for c in df.columns if c not in ["DATETIME", "ATT_FLAG"]]
    sensor_type_map = {col: get_sensor_type(col) for col in sensor_cols}

    fit_df = df if fit_rows is None else df.iloc[np.asarray(fit_rows, dtype=int)]
    df_normal = fit_df[fit_df["ATT_FLAG"] == 0][sensor_cols]
    if len(df_normal) < 2:
        raise ValueError("BATADAL needs at least two normal training rows for its correlation baseline")
    G_normal = build_correlation_graph(df_normal, sensor_cols, sensor_type_map)

    sensor_list = sorted(sensor_cols)
    sensor_index = {s: i for i, s in enumerate(sensor_list)}
    type_codes = {t: i for i, t in enumerate(sorted(set(sensor_type_map.values())))}

    fixed_edges = [(sensor_index[u], sensor_index[v]) for u, v in G_normal.edges()
                   if u in sensor_index and v in sensor_index]
    fixed_edges = fixed_edges + [(v, u) for u, v in fixed_edges]
    edge_index = (torch.tensor(fixed_edges, dtype=torch.long).T
                  if fixed_edges else torch.empty((2, 0), dtype=torch.long))

    baseline_mean = df_normal[sensor_list].mean()
    baseline_std = df_normal[sensor_list].std().replace(0, 1e-9).fillna(1)

    is_attack = (df["ATT_FLAG"] == 1).astype(int).to_numpy()
    n_rows = len(df)
    starts = list(range(0, n_rows - window_size + 1, stride))

    window_graphs, window_labels = [], []
    for s in starts:
        window = df[sensor_list].iloc[s:s + window_size]
        label = int(is_attack[s:s + window_size].sum() > 0)

        w_mean = window.mean()
        w_std = window.std().fillna(0)
        zscore = ((w_mean - baseline_mean) / baseline_std).clip(-5, 5)

        feats = []
        for sensor in sensor_list:
            vmin, vmax = fit_df[sensor].min(), fit_df[sensor].max()
            mean_norm = (w_mean[sensor] - vmin) / (vmax - vmin + 1e-9)
            std_norm = w_std[sensor] / (vmax - vmin + 1e-9)
            feats.append([mean_norm, std_norm, zscore[sensor], type_codes[sensor_type_map[sensor]]])

        x = torch.tensor(np.array(feats, dtype=np.float32))
        g = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        window_graphs.append(g)
        window_labels.append(label)

    window_labels = np.array(window_labels)
    extra = {"starts": starts, "sensor_list": sensor_list, "window_size": window_size, "stride": stride}
    return window_graphs, window_labels, extra


def temporal_split_windows(window_graphs: List[Data], split_frac: float = 0.7
                            ) -> Tuple[List[Data], List[Data]]:
    """Chronological 70/30 split on windows (already time-ordered, no reshuffle). Note:
    with WINDOW_SIZE=24 and STRIDE=6, windows near the boundary share up to 18h of raw
    rows across train/test, so this boundary is softer than SCADANet's. Use
    `batadal_purged_split` for a leakage-free split.
    """
    train_idx,test_idx=reference_window_positions(len(window_graphs),split_frac)
    return [window_graphs[i] for i in train_idx],[window_graphs[i] for i in test_idx]


@replayable_split
def reference_window_positions(n_windows,split_frac):
    split_idx=int(n_windows*split_frac)
    return np.arange(split_idx),np.arange(split_idx,n_windows)
