"""
SCADANet loading, feature construction, temporal windowing.

Every function body here is moved (not rewritten) from the notebook cells
named in each docstring. Column lists, thresholds, and constants are
copied exactly — if you change them, change them here AND note it in
docs/weekly-progress.md, since they define what "reproduces the paper"
means.
"""
from src.utils.reproducibility import replayable_split
from dataclasses import dataclass
from typing import Dict, List, Tuple

import kagglehub
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

SRC_IP_COL = "Source"
DST_IP_COL = "Destination"
LABEL_COL = "Attack_Type"

# Real per-flow signal -> the feature set
NUMERIC_FEATURE_COLS = [
    "ip_ttl", "frame_len", "ip_len", "Tcp_window_size", "Udp_length",
    "Tcp_len", "Tcp_hdr_len", "Http_content_length", "Http_time",
    "Frame_time_delta",
]
CATEGORICAL_FEATURE_COLS = [
    "Protocol", "Http_request_method", "Http_response_code",
    "Tcp_flags_reset", "http_accept", "Icmp_type", "Icmp_code",
    "Http_user_agent",
    "Tls_record",
    "tls_handshake",
    "Http_request_uri_query",
    "Time_response",
    "Tcp_analysis_flags",
]


def load_scadanet_df(csv_path: str = None) -> pd.DataFrame:
    """If csv_path is None, downloads via kagglehub exactly as the notebook did."""
    if csv_path is None:
        path = kagglehub.dataset_download("ealgul/scada-dataset-v01")
        import glob
        csv_files = glob.glob(f"{path}/*.csv")
        csv_path = csv_files[0]

    scada_df = pd.read_csv(csv_path)
    scada_df.columns = scada_df.columns.str.strip()

    scada_df["is_attack"] = (
        ~scada_df[LABEL_COL].astype(str).str.strip().str.lower().str.contains("normal")
    ).astype(int)

    return scada_df


@dataclass
class GraphTensors:
    x: torch.Tensor
    edge_index: torch.Tensor          # fixed IP-pair topology
    query_edges: torch.Tensor         # per-flow (src, dst) pairs, with repeats
    edge_attr: torch.Tensor
    y: torch.Tensor
    ip_index: Dict[str, int]
    all_ips: List[str]
    data: Data                        # PyG Data(x, edge_index) for convenience
    feature_names: List[str] = None   # column names for edge_attr, in order


def _normalize_mixed_type_categorical(series: pd.Series) -> pd.Series:
    """Collapse numeric-like values written in different forms (200.0 / '200' / '200.0')
    into one consistent string before one-hot.
    """
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    is_numeric_like = numeric.notna()
    normalized = series.astype(str)
    normalized[is_numeric_like] = numeric[is_numeric_like].astype(int).astype(str)
    normalized[series.isna()] = "missing"
    return normalized


def _build_edge_attr(df: pd.DataFrame, numeric_cols: List[str],
                      categorical_cols: List[str], fit_idx=None) -> Tuple[torch.Tensor, List[str]]:
    """Median-impute + z-score numeric, one-hot categorical."""
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    if fit_idx is not None:
        return _fit_transform_edge_attr(df, numeric_cols, categorical_cols, fit_idx)

    num_block = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    num_block = num_block.fillna(num_block.median(numeric_only=True))
    num_arr = num_block.to_numpy(dtype=np.float32)
    mu, sigma = num_arr.mean(axis=0, keepdims=True), num_arr.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    num_arr = (num_arr - mu) / sigma

    cat_block = df[categorical_cols].apply(_normalize_mixed_type_categorical)
    cat_dummies = pd.get_dummies(cat_block, columns=categorical_cols, dummy_na=False)
    cat_arr = cat_dummies.to_numpy(dtype=np.float32)

    edge_attr = np.concatenate([num_arr, cat_arr], axis=1)
    feature_names = numeric_cols + list(cat_dummies.columns)
    return torch.tensor(edge_attr, dtype=torch.float32), feature_names


def build_graph_tensors(df: pd.DataFrame, fit_idx=None) -> GraphTensors:
    """Build the IP-pair topology, query edges and edge attributes.

    Node features are a constant, non-identifying vector; raw flow-count degree is
    deliberately excluded to avoid shortcut learning.
    """
    all_ips = sorted(set(df[SRC_IP_COL].astype(str)) | set(df[DST_IP_COL].astype(str)))
    ip_index = {ip: i for i, ip in enumerate(all_ips)}

    x = torch.ones((len(all_ips), 4), dtype=torch.float32)

    unique_pairs = df[[SRC_IP_COL, DST_IP_COL]].astype(str).drop_duplicates()
    topo_src = unique_pairs[SRC_IP_COL].map(ip_index).to_numpy()
    topo_dst = unique_pairs[DST_IP_COL].map(ip_index).to_numpy()
    edge_index = torch.tensor(np.stack([topo_src, topo_dst]), dtype=torch.long)

    q_src = torch.tensor(df[SRC_IP_COL].astype(str).map(ip_index).to_numpy(), dtype=torch.long)
    q_dst = torch.tensor(df[DST_IP_COL].astype(str).map(ip_index).to_numpy(), dtype=torch.long)
    query_edges = torch.stack([q_src, q_dst])
    y = torch.tensor(df["is_attack"].to_numpy(), dtype=torch.long)

    edge_attr, feature_names = _build_edge_attr(df, NUMERIC_FEATURE_COLS, CATEGORICAL_FEATURE_COLS, fit_idx=fit_idx)

    data = Data(x=x, edge_index=edge_index)
    return GraphTensors(x=x, edge_index=edge_index, query_edges=query_edges,
                         edge_attr=edge_attr, y=y, ip_index=ip_index,
                         all_ips=all_ips, data=data, feature_names=feature_names)


def add_temporal_windows(df: pd.DataFrame, ip_index: Dict[str, int],
                          n_windows: int = 20) -> Tuple[pd.DataFrame, List[Dict]]:
    """Assign equal-count time windows and build cumulative topology snapshots.

    Returns (df sorted+windowed, list of snapshot dicts with cumulative 'edge_index' per
    window). snapshot[w] is the topology visible using only windows 0..w, so no test-
    period edges leak into the training-period topology.

    Ties on duplicate `Time` values are broken with a secondary sort key derived from
    each row's own content (a hash over several columns), so the order does not depend
    on the CSV row order or on how the file was downloaded.
    """
    tie_break_cols = [c for c in [
        SRC_IP_COL, DST_IP_COL, "Protocol", "Length", "frame_len", "ip_ttl",
        "Tcp_seq", "Tcp_ack", "Udp_srcport", "Udp_dstport", LABEL_COL,
    ] if c in df.columns]
    content_key = pd.util.hash_pandas_object(
        df[tie_break_cols].astype(str), index=False
    )
    df_keyed = df.assign(_content_tiebreak=content_key.to_numpy())

    df_t = (
        df_keyed.sort_values(["Time", "_content_tiebreak"], kind="stable")
        .drop(columns="_content_tiebreak")
        .reset_index()
        .rename(columns={"index": "orig_idx"})
    )
    df_t["window"] = (np.arange(len(df_t)) * n_windows) // len(df_t)

    snapshots = []
    seen_pairs = set()
    for w in range(n_windows):
        w_df = df_t[df_t["window"] == w]
        pairs = w_df[[SRC_IP_COL, DST_IP_COL]].astype(str).drop_duplicates()
        pairs = pairs[pairs[SRC_IP_COL].isin(ip_index) & pairs[DST_IP_COL].isin(ip_index)]
        seen_pairs |= set(zip(pairs[SRC_IP_COL], pairs[DST_IP_COL]))

        if seen_pairs:
            src = [ip_index[a] for a, b in sorted(seen_pairs)]
            dst = [ip_index[b] for a, b in sorted(seen_pairs)]
            edge_index_w = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        else:
            edge_index_w = torch.zeros((2, 0), dtype=torch.long)

        snapshots.append({"window": w, "n_edges_seen": len(seen_pairs), "edge_index": edge_index_w})

    return df_t, snapshots


@replayable_split
def temporal_split_positions(df_t: pd.DataFrame, train_fraction: float = 0.7
                              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """SPLIT_WINDOW boundary and train/test row positions (positions into df_t, which is
    already sorted+windowed — use these to index query_edges/edge_attr/y built on the
    SAME row order).
    """
    n_windows = int(df_t["window"].max()) + 1
    split_window = int(n_windows * train_fraction)
    train_mask = (df_t["window"] < split_window).to_numpy()
    test_mask = (df_t["window"] >= split_window).to_numpy()
    return np.nonzero(train_mask)[0], np.nonzero(test_mask)[0], split_window


def _fit_transform_edge_attr(df, numeric_cols, categorical_cols, fit_idx):
    idx = np.asarray(fit_idx, dtype=int)
    if not len(idx): raise ValueError("Feature fit requires training rows")
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf],np.nan)
    med = numeric.iloc[idx].median().fillna(0)
    numeric = numeric.fillna(med)
    mu = numeric.iloc[idx].mean(); sd = numeric.iloc[idx].std(ddof=0).replace(0,1).fillna(1)
    num = ((numeric-mu)/sd).to_numpy(dtype=np.float32)
    cats = df[categorical_cols].apply(_normalize_mixed_type_categorical)
    train_cats = pd.get_dummies(cats.iloc[idx], columns=categorical_cols, dtype=float)
    cat = pd.get_dummies(cats, columns=categorical_cols, dtype=float).reindex(columns=train_cats.columns,fill_value=0)
    return torch.tensor(np.concatenate([num,cat.to_numpy(dtype=np.float32)],axis=1),dtype=torch.float32), numeric_cols+list(cat.columns)
