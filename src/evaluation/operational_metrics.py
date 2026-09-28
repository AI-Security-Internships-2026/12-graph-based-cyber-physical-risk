"""The three operational metrics that need a real notion of TIME, not just per-flow labels:
false alarms per time unit, attack/event recall, and time-to-detect. `compute_metrics`
(precision/recall/F1/AUPRC/FPR) and `measure_inference_latency` /
`count_trainable_params` already exist in `src/evaluation/metrics.py` and are reused as-
is for the rest of Table S3 — nothing here duplicates those.

IMPORTANT — read before trusting any number this module produces:

1. `false_alarms_per_unit_time` assumes SCADANet's `Time` column (used already for
chronological ordering in `add_temporal_windows`) is in SECONDS. That's a reasonable
guess for a pcap-derived capture, and is the only unit `Frame_time_delta` existing
alongside it is consistent with — but it has NOT been confirmed against the real
dataset, because this module was written without real data or a working torch/PyG
environment. If `Time` turns out to be in a different unit, every "false alarms per
hour" number this produces is wrong by a constant factor until `seconds_per_time_unit`
is corrected. 2. SCADANet/BATADAL have no explicit "attack event ID" field — an attack
is a sequence of individually-labeled flows, not a pre-grouped incident.
`group_into_events` below turns that into events using a time-gap heuristic (a new event
starts when the gap since the last same-attack-type flow exceeds `gap_seconds`) — this
is a REASONABLE, commonly-used operational definition, but it is this session's own
definition, not a field the dataset provides.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def false_alarms_per_unit_time(
    pred, true, time_seconds, seconds_per_time_unit: float = 3600.0,
) -> Dict[str, Optional[float]]:
    """False positives per `seconds_per_time_unit` (default: per hour), over the time span
    actually covered by the given eval rows.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    time_seconds = np.asarray(time_seconds, dtype=float)
    if len(np.unique(time_seconds)) < 2:
        return {"false_alarms_per_time_unit": None,
                "reason": "time column has fewer than 2 distinct values in this eval set"}
    span_seconds = float(time_seconds.max() - time_seconds.min())
    if span_seconds <= 0:
        return {"false_alarms_per_time_unit": None,
                "reason": "non-positive time span in this eval set"}
    n_false_alarms = int(((pred == 1) & (true == 0)).sum())
    span_units = span_seconds / seconds_per_time_unit
    return {"false_alarms_per_time_unit": float(n_false_alarms / span_units) if span_units > 0 else None,
            "n_false_alarms": n_false_alarms, "span_time_units": float(span_units),
            "seconds_per_time_unit_assumed": seconds_per_time_unit}


def group_into_events(
    attack_type, time_seconds, is_attack, gap_seconds: float = 60.0,
) -> pd.DataFrame:
    """Groups consecutive (by time order) same-attack-type positive rows
    into "events" separated by gaps larger than `gap_seconds`. Returns a
    DataFrame with one row per flow: `event_id` (NaN for normal/benign
    rows — only attack rows are grouped) plus the original index
    position, so callers can join event_id back onto pred/true arrays.
    `gap_seconds` is this session's own operational choice (see module
    docstring point 2) — 60s is a starting default, not a dataset-given
    value; override it and re-check Table S3/Figure S3 if a different
    value changes the story."""
    df = pd.DataFrame({
        "pos": np.arange(len(attack_type)),
        "attack_type": np.asarray(attack_type),
        "time": np.asarray(time_seconds, dtype=float),
        "is_attack": np.asarray(is_attack).astype(bool),
    })
    df["event_id"] = np.nan
    next_event_id = 0
    for atype, g in df[df["is_attack"]].groupby("attack_type"):
        g = g.sort_values("time")
        gaps = g["time"].diff().fillna(np.inf)
        new_event = gaps > gap_seconds
        event_ids = new_event.cumsum().to_numpy() + next_event_id
        df.loc[g.index, "event_id"] = event_ids
        next_event_id = int(event_ids.max()) + 1
    return df


def event_level_recall(events_df: pd.DataFrame, pred) -> Dict[str, Optional[float]]:
    """Fraction of attack EVENTS (from `group_into_events`) with at
    least one flow correctly predicted positive — a coarser, arguably
    more operationally meaningful recall than per-flow recall, since a
    SOC analyst only needs ONE alert per incident to notice it."""
    pred = np.asarray(pred)
    events_df = events_df.copy()
    events_df["pred"] = pred
    attack_events = events_df.dropna(subset=["event_id"])
    if attack_events.empty:
        return {"event_recall": None, "n_events": 0,
                "reason": "no attack rows to group into events in this eval set"}
    detected = attack_events.groupby("event_id")["pred"].max()
    return {"event_recall": float((detected == 1).mean()), "n_events": int(detected.shape[0]),
            "n_events_detected": int((detected == 1).sum())}


def time_to_detect(events_df: pd.DataFrame, pred) -> Dict[str, Optional[float]]:
    """For each DETECTED event (>=1 true positive in it), the time
    between the event's first flow and its first correctly-predicted
    flow. Mean/median over detected events, in the same time units as
    `time_seconds` was given in (seconds, per this module's SCADANet
    `Time`-column assumption — see module docstring point 1)."""
    pred = np.asarray(pred)
    events_df = events_df.copy()
    events_df["pred"] = pred
    attack_events = events_df.dropna(subset=["event_id"])
    if attack_events.empty:
        return {"time_to_detect_mean": None, "time_to_detect_median": None,
                "reason": "no attack rows to group into events in this eval set"}

    ttds: List[float] = []
    for event_id, g in attack_events.groupby("event_id"):
        g = g.sort_values("time")
        hits = g[g["pred"] == 1]
        if hits.empty:
            continue  # event not detected at all — excluded from time-to-detect by definition
        ttds.append(float(hits["time"].iloc[0] - g["time"].iloc[0]))

    if not ttds:
        return {"time_to_detect_mean": None, "time_to_detect_median": None,
                "reason": "no events were detected (>=1 true positive) in this eval set"}
    return {"time_to_detect_mean": float(np.mean(ttds)), "time_to_detect_median": float(np.median(ttds)),
            "n_detected_events_timed": len(ttds)}
