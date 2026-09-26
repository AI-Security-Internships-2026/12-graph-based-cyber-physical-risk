"""
SCADANet loading, feature construction, temporal windowing.

Every function body here is moved (not rewritten) from the notebook cells
named in each docstring. Column lists, thresholds, and constants are
copied exactly — if you change them, change them here AND note it in
docs/weekly-progress.md, since they define what "reproduces the paper"
means.
"""
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

# Real per-flow signal -> the feature set (Week 7 Part G.2, Cell 116)
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
    """Week 6 Part G data-loading cell (Cell 106) + label cell (Cell 107).
    If csv_path is None, downloads via kagglehub exactly as the notebook did.
    """
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
                                       # (added for Issue #4's feature-permutation
                                       # counterfactuals; previously computed by
                                       # _build_edge_attr and silently discarded)


def _normalize_mixed_type_categorical(series: pd.Series) -> pd.Series:
    """Cell 117 — collapse numeric-like values written in different forms
    (200.0 / '200' / '200.0') into one consistent string before one-hot."""
    numeric = pd.to_numeric(series, errors="coerce")
    is_numeric_like = numeric.notna()
    normalized = series.astype(str)
    normalized[is_numeric_like] = numeric[is_numeric_like].astype(int).astype(str)
    normalized[series.isna()] = "missing"
    return normalized


def _build_edge_attr(df: pd.DataFrame, numeric_cols: List[str],
                      categorical_cols: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """Cell 117 — median-impute + z-score numeric, one-hot categorical."""
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

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


def build_graph_tensors(df: pd.DataFrame) -> GraphTensors:
    """Cell 109 (topology + query edges) + Cell 117 (edge_attr).
    Constant, non-identifying node feature vector — see Cell 109 comment:
    raw flow-count degree is deliberately excluded (shortcut-learning risk)."""
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

    edge_attr, feature_names = _build_edge_attr(df, NUMERIC_FEATURE_COLS, CATEGORICAL_FEATURE_COLS)

    data = Data(x=x, edge_index=edge_index)
    return GraphTensors(x=x, edge_index=edge_index, query_edges=query_edges,
                         edge_attr=edge_attr, y=y, ip_index=ip_index,
                         all_ips=all_ips, data=data, feature_names=feature_names)


def add_temporal_windows(df: pd.DataFrame, ip_index: Dict[str, int],
                          n_windows: int = 20) -> Tuple[pd.DataFrame, List[Dict]]:
    """Cell 156 (equal-COUNT windowing) + Cell 157 (streaming/cumulative
    topology snapshots). Returns (df sorted+windowed, list of snapshot dicts
    with cumulative 'edge_index' per window) — snapshot[w] is the topology
    visible using only windows 0..w, so no test-period edges leak into the
    training-period topology.

    Tie-breaking on duplicate `Time` values: the original notebook cell
    used an unspecified sort (quicksort), whose tie order isn't guaranteed
    identical across pandas versions. An earlier fix used `kind="stable"`,
    which fixes tie order *relative to input row order* — but that still
    depends on what order kagglehub's download/cache handed the CSV rows
    in, which isn't guaranteed identical across separate downloads. A
    ~3–10 flow boundary discrepancy (out of ~160k test flows) was observed
    across two separate Colab runs of otherwise identical code, consistent
    with this: the fix below removes the dependency on file/download order
    entirely by using a secondary sort key derived from each row's own
    content (a hash over several columns unlikely to collide for two
    genuinely different flows), so tie-breaking is now a function of what
    the data IS, not what order it arrived in.
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
            src = [ip_index[a] for a, b in seen_pairs]
            dst = [ip_index[b] for a, b in seen_pairs]
            edge_index_w = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        else:
            edge_index_w = torch.zeros((2, 0), dtype=torch.long)

        snapshots.append({"window": w, "n_edges_seen": len(seen_pairs), "edge_index": edge_index_w})

    return df_t, snapshots


def temporal_split_positions(df_t: pd.DataFrame, train_fraction: float = 0.7
                              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Cell 159 — SPLIT_WINDOW boundary and train/test row positions
    (positions into df_t, which is already sorted+windowed — use these to
    index query_edges/edge_attr/y built on the SAME row order)."""
    n_windows = int(df_t["window"].max()) + 1
    split_window = int(n_windows * train_fraction)
    train_mask = (df_t["window"] < split_window).to_numpy()
    test_mask = (df_t["window"] >= split_window).to_numpy()
    return np.nonzero(train_mask)[0], np.nonzero(test_mask)[0], split_window
