"""
Issue #6 — leakage-safe host-relative feature normalization for SCADANet,
plus the host-level false-positive diagnostics the issue's section 5
requires.

Every function here works on a *(dataframe, idx)* pair where `idx` is a
set of row POSITIONS into that exact dataframe (not global row labels) —
this is the same convention `src/data/scadanet.py` and
`src/experiments/temporal_runner.py` already use (`df.iloc[pos]`), so the
same functions apply unchanged whether the caller is operating on the
Track B split's `df` (Issue #1's row order) or the strict-temporal
protocol's `df_t` (Issue #5's chronologically re-sorted frame) — pass
whichever frame the `idx` values were drawn from.

Leakage-safety contract: `fit_host_stats()` / `fit_host_flag_rate()` are
the ONLY functions here that estimate anything from data, and they must
be called with train-only (or train-minus-validation) positions. Every
other function in this module (`host_normalized_matrix`,
`host_flag_rate_feature`, `build_host_normalized_edge_attr`) only
*applies* an already-fit `host_stats`/`rate_map` dict — it performs no
fitting itself, so calling it on test-set `idx` cannot leak test
statistics into the normalization, by construction rather than by
convention.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Continuous features implicated in the false-positive analysis.
HOST_NORM_NUMERIC_COLS: List[str] = ["frame_len", "ip_len", "Tcp_window_size"]

# Hosts with fewer training rows use global training statistics.
MIN_HOST_TRAIN_SAMPLES = 20

# Hosts identified in the Track B false-positive analysis.
KNOWN_PROBLEMATIC_HOSTS: List[str] = ["192.168.119.140", "192.168.119.141"]


def _to_positions(idx) -> np.ndarray:
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy()
    return np.asarray(idx)


# ---------------------------------------------------------------------
# Fitting (TRAIN-ONLY — the only functions in this module that estimate
# anything from data)
# ---------------------------------------------------------------------

def fit_host_stats(
    df: pd.DataFrame, train_idx, src_col: str,
    numeric_cols: List[str] = HOST_NORM_NUMERIC_COLS,
    min_samples: int = MIN_HOST_TRAIN_SAMPLES,
) -> Dict[str, Dict[str, Optional[Dict[str, float]]]]:
    """Per-host mean/std/median/IQR for each of `numeric_cols`, computed
    from `df.iloc[train_idx]` ONLY. Returns
    ``{"__global__": {col: stats}, host: {col: stats_or_None}}`` — a host
    entry is `None` for a given column when that host has fewer than
    `min_samples` train rows for it (too few to estimate reliably), which
    `host_normalized_matrix` reads as "use the global fallback for this
    host+col" rather than an error."""
    train_df = df.iloc[_to_positions(train_idx)]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    def _stats(vals: pd.Series) -> Dict[str, float]:
        vals = pd.to_numeric(vals, errors="coerce")
        std = float(vals.std())
        iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
        return {
            "mean": float(vals.mean()), "std": std if std > 1e-9 else 1.0,
            "median": float(vals.median()), "iqr": iqr if iqr > 1e-9 else 1.0,
            "n_train": int(vals.notna().sum()),
        }

    stats: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {
        "__global__": {col: _stats(train_df[col]) for col in numeric_cols}
    }
    for host, g in train_df.groupby(train_df[src_col].astype(str)):
        entry = {}
        for col in numeric_cols:
            n = int(pd.to_numeric(g[col], errors="coerce").notna().sum())
            entry[col] = _stats(g[col]) if n >= min_samples else None
        stats[str(host)] = entry
    return stats


def fit_host_flag_rate(
    df: pd.DataFrame, train_idx, src_col: str, flag_col: str, positive_value,
    min_samples: int = MIN_HOST_TRAIN_SAMPLES,
) -> Optional[Dict[str, Optional[float]]]:
    """Per-host P(flag_col == positive_value), TRAIN-ONLY (Issue #6
    section 2's optional categorical treatment — e.g.
    P(Tcp_flags_reset='Set' | host, training period)). Returns None if
    `flag_col` isn't in `df` (not every dataset build has every column),
    matching this repo's "applicable=False" convention elsewhere rather
    than raising."""
    if flag_col not in df.columns:
        return None
    train_df = df.iloc[_to_positions(train_idx)]
    is_pos = train_df[flag_col].astype(str).str.strip().eq(str(positive_value))
    hosts = train_df[src_col].astype(str)

    rates: Dict[str, Optional[float]] = {"__global__": float(is_pos.mean())}
    counts = is_pos.groupby(hosts).count()
    means = is_pos.groupby(hosts).mean()
    for host in counts.index:
        rates[str(host)] = float(means[host]) if counts[host] >= min_samples else None
    return rates


# ---------------------------------------------------------------------
# Applying an already-fit host_stats / rate_map (safe to call on ANY
# idx, including test — no fitting happens here)
# ---------------------------------------------------------------------

def host_normalized_matrix(
    df: pd.DataFrame, idx, src_col: str, host_stats: Dict, numeric_cols: List[str],
    method: str = "zscore",
) -> np.ndarray:
    """Apply a `host_stats` dict (from `fit_host_stats`, fit on TRAIN
    positions) to `df.iloc[idx]` — `idx` may be train, val, or test
    positions; this function never re-estimates anything, it only looks
    up each row's host's center/scale (or the `__global__` fallback for
    an unseen or too-small host) and rescales. `method`: "zscore" uses
    mean/std, "robust" uses median/IQR (Issue #6's optional robust
    variant for heavily skewed features)."""
    if method not in ("zscore", "robust"):
        raise ValueError(f"method must be 'zscore' or 'robust', got {method!r}")
    center_key, scale_key = ("mean", "std") if method == "zscore" else ("median", "iqr")

    sub = df.iloc[_to_positions(idx)]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    hosts = sub[src_col].astype(str)
    global_stats = host_stats["__global__"]

    out = np.zeros((len(sub), len(numeric_cols)), dtype=np.float32)
    for j, col in enumerate(numeric_cols):
        vals = pd.to_numeric(sub[col], errors="coerce")
        unique_hosts = hosts.unique()
        center_map, scale_map = {}, {}
        for h in unique_hosts:
            entry = host_stats.get(h, {}).get(col) if h in host_stats else None
            if entry is None:
                entry = global_stats[col]
            center_map[h] = entry[center_key]
            scale_map[h] = entry[scale_key]
        centers = hosts.map(center_map).to_numpy(dtype=np.float64)
        scales = hosts.map(scale_map).to_numpy(dtype=np.float64)
        raw = vals.to_numpy(dtype=np.float64)
        out[:, j] = np.where(np.isnan(raw), 0.0, (raw - centers) / scales).astype(np.float32)
    return out


def host_flag_rate_feature(df: pd.DataFrame, idx, src_col: str, rate_map: Dict) -> np.ndarray:
    """Apply a `rate_map` from `fit_host_flag_rate` (TRAIN-fit) to
    `df.iloc[idx]` — same safe-to-call-on-test contract as
    `host_normalized_matrix`."""
    sub = df.iloc[_to_positions(idx)]
    hosts = sub[src_col].astype(str)
    global_rate = rate_map["__global__"]
    return hosts.map(lambda h: rate_map.get(h) if rate_map.get(h) is not None else global_rate
                      ).to_numpy(dtype=np.float32)


def build_host_normalized_edge_attr(
    df: pd.DataFrame, edge_attr: torch.Tensor, feature_names: List[str], train_idx, src_col: str,
    numeric_cols: List[str] = HOST_NORM_NUMERIC_COLS,
    flag_col: str = "Tcp_flags_reset", flag_positive: str = "Set",
    method: str = "zscore",
) -> Tuple[torch.Tensor, Dict, Optional[Dict], List[str]]:
    """Starts from `edge_attr`/`feature_names` — Issue #1's
    `build_graph_tensors` output, i.e. every OTHER numeric feature's
    global z-score and every categorical one-hot column is reused
    UNCHANGED, not recomputed — and:

      1. Replaces each column in `numeric_cols` (matched by name in
         `feature_names`) with its host-relative score, fit on
         `train_idx` only via `fit_host_stats`.
      2. Appends one new column — host-relative
         P(flag_col == flag_positive | host, train period) — per Issue
         #6 section 2's "consider host-relative frequency features"
         (explicitly NOT z-scoring the categorical one-hot itself).

    Returns (new_edge_attr, host_stats, flag_rate_map_or_None,
    new_feature_names) — the caller needs `host_stats`/`flag_rate_map`
    to apply the SAME fitted statistics to a differently-shaped
    `edge_attr` later isn't needed here (this function already applies
    to every row of `edge_attr`, train+test alike, using train-only fitted
    stats — see module docstring), but they're returned anyway for the
    result JSON / reproducibility record.
    """
    numeric_cols = [c for c in numeric_cols if c in feature_names and c in df.columns]
    host_stats = fit_host_stats(df, train_idx, src_col, numeric_cols)

    new_edge_attr = edge_attr.clone()
    all_idx = np.arange(edge_attr.shape[0])
    if numeric_cols:
        host_vals = host_normalized_matrix(df, all_idx, src_col, host_stats, numeric_cols, method=method)
        for j, col in enumerate(numeric_cols):
            col_j = feature_names.index(col)
            new_edge_attr[:, col_j] = torch.tensor(host_vals[:, j], dtype=torch.float32)

    flag_rate_map = fit_host_flag_rate(df, train_idx, src_col, flag_col, flag_positive)
    new_feature_names = list(feature_names)
    if flag_rate_map is not None:
        flag_feat = host_flag_rate_feature(df, all_idx, src_col, flag_rate_map)
        new_edge_attr = torch.cat(
            [new_edge_attr, torch.tensor(flag_feat, dtype=torch.float32).unsqueeze(1)], dim=1
        )
        new_feature_names.append(f"host_rate_{flag_col}_{flag_positive}")

    return new_edge_attr, host_stats, flag_rate_map, new_feature_names


# ---------------------------------------------------------------------
# Host-level diagnostics (Issue #6 section 5)
# ---------------------------------------------------------------------

def false_positives_by_host(
    df: pd.DataFrame, idx, src_col: str, dst_col: str, pred, true,
) -> Tuple[pd.Series, pd.Series]:
    """FP counts by source host and by destination host, over the given
    eval rows — descending pd.Series indexed by IP string."""
    sub = df.iloc[_to_positions(idx)]
    pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    true_np = true.detach().cpu().numpy() if torch.is_tensor(true) else np.asarray(true)
    is_fp = (pred_np == 1) & (true_np == 0)
    fp_by_src = sub.loc[sub.index[is_fp], src_col].astype(str).value_counts()
    fp_by_dst = sub.loc[sub.index[is_fp], dst_col].astype(str).value_counts()
    return fp_by_src, fp_by_dst


def top_k_fp_share(fp_by_host: pd.Series, ks: Tuple[int, ...] = (1, 2, 5)) -> Dict[str, Optional[float]]:
    """Share of all false positives caused by the top-k FP-generating
    hosts (Issue #6 section 5: "share of all false positives caused by
    top 1/top 2/top 5 hosts")."""
    total = int(fp_by_host.sum())
    if total == 0:
        return {f"top_{k}_fp_share": None for k in ks}
    sorted_counts = fp_by_host.sort_values(ascending=False)
    return {f"top_{k}_fp_share": float(sorted_counts.iloc[:k].sum() / total) for k in ks}
