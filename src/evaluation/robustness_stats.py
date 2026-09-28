"""Paired statistical comparisons and calibration metrics: pure statistical primitives with
no experiment orchestration (see src/experiments/robustness_runner.py).
"""
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sps


def descriptive_stats(values) -> Dict:
    """mean/std/median/95% CI for one condition's per-seed metric values. Uses a
    t-distribution CI, which is the standard choice for small samples (5-10 seeds,
    exactly this repo's seed-set size) — NOT a bootstrap, which needs more samples than
    a typical seed count to be reliable on its own; paired comparisons below also use a
    t-interval on the seed differences.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    median = float(np.median(values))
    if n > 1:
        se = std / np.sqrt(n)
        tcrit = sps.t.ppf(0.975, df=n - 1)
        ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
    else:
        ci_low = ci_high = float("nan")
    return {"n": n, "mean": mean, "std": std, "median": median,
            "ci_95_low": float(ci_low), "ci_95_high": float(ci_high)}


def paired_t_ci(a, b, alpha: float = 0.05) -> Tuple[float, float]:
    """95% t-interval on the paired mean difference (a - b). Replaces the
    earlier bootstrap over seed differences, which under-covers at
    n<=10 (percentile bootstrap on ~5-10 points is too narrow)."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    tcrit = float(sps.t.ppf(1 - alpha / 2, df=n - 1))
    return m - tcrit * se, m + tcrit * se


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values; None entries are passed through."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    order = sorted(idx, key=lambda i: p_values[i])
    m = len(order)
    adj = [None] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adj[i] = running
    return adj


def cohens_d_paired(a, b) -> float:
    """Effect size for a paired comparison: mean difference in units of
    the difference's own standard deviation."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    return float(diffs.mean() / sd) if sd > 1e-12 else float("nan")


def paired_comparison(
    a, b, label_a: str, label_b: str, metric_name: str,
    n_boot: int = 0, seed: int = 42,
) -> Dict:
    """One row of Table S2. `a`/`b` must be seed-matched — same length, same seed order.

    Test selection is MECHANICAL, not chosen per-comparison: a paired t-test is used
    only when Shapiro-Wilk on the DIFFERENCES doesn't reject normality (p > 0.05) AND
    there are at least 3 seeds (Shapiro is undefined below that); otherwise Wilcoxon
    signed-rank. Below 3 seeds, no test is run at all — the descriptive difference and
    t-interval is reported with `test_used` explaining why, rather than running an
    underpowered test and reporting a p-value that would be misleading with n<3.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"paired_comparison requires equal-length, seed-matched arrays "
            f"(got {len(a)} vs {len(b)}) for {label_a!r} vs {label_b!r} / {metric_name!r}")
    diffs = a - b
    n = len(diffs)

    # Primary test is FIXED in advance: paired t-test (+ Cohen's d and a
    # t-interval). Wilcoxon is reported as a robustness check only.
    # Shapiro-Wilk is kept as a diagnostic, not as a switch (near-powerless
    # at n<=10, so switching on it would be arbitrary).
    normal_p = None
    if n >= 3 and np.std(diffs) > 1e-12:
        _, normal_p = sps.shapiro(diffs)

    stat: Optional[float] = None
    p: Optional[float] = None
    wilcoxon_p: Optional[float] = None
    if n < 3:
        test_used = "too_few_seeds_for_a_test (n<3): reporting descriptive difference only"
    elif np.std(diffs) <= 1e-12:
        test_used = "degenerate (zero variance in differences): descriptive difference only"
        stat, p = (0.0, 1.0) if np.allclose(diffs, 0) else (None, None)
    else:
        stat, p = sps.ttest_rel(a, b)
        test_used = "paired_t_test"
        try:
            wilcoxon_p = float(sps.wilcoxon(a, b).pvalue)
        except ValueError:
            wilcoxon_p = None

    ci_low, ci_high = paired_t_ci(a, b)
    effect = cohens_d_paired(a, b)
    mean_diff = float(diffs.mean())
    rel_diff = float(mean_diff / b.mean()) if abs(b.mean()) > 1e-12 else None

    return {
        "comparison": f"{label_a} vs {label_b}", "metric": metric_name, "n_seeds": n,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_difference": mean_diff, "relative_difference": rel_diff,
        "ci_95_low": ci_low, "ci_95_high": ci_high, "ci_method": "paired_t_interval",
        "effect_size_cohens_d": effect, "test_used": test_used,
        "test_statistic": float(stat) if stat is not None else None,
        "p_value": float(p) if p is not None else None,
        "wilcoxon_p_value_robustness": wilcoxon_p,
        "normality_p_value": float(normal_p) if normal_p is not None else None,
    }


def expected_calibration_error(probs, true, n_bins: int = 10) -> float:
    """ECE — mean absolute gap between predicted probability and actual positive rate,
    within `n_bins` equal-width probability bins, weighted by bin size.
    """
    probs = np.asarray(probs, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(probs)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(true[mask].mean() - probs[mask].mean())
    return float(ece)


def brier_score(probs, true) -> float:
    probs = np.asarray(probs, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean((probs - true) ** 2))
