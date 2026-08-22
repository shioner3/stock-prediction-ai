"""Score Validation: does a higher Total Score correspond to a better
Forward Return? This module only computes descriptive bucket statistics
and a monotonicity indicator - it never concludes Score is valid or
invalid. That judgment (with proper OOS separation, bootstrap CIs, and
permutation tests) is Phase 6's job.

Score Validation JOINS Score Records with Forward Targets
(targets/forward_returns.py) - a join that is only possible because both
already exist independently; Score itself was computed with zero
knowledge of Forward Targets (see tests/test_target_leakage.py and
tests/test_scoring_no_lookahead.py's Test C).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import BacktestMetrics, compute_metrics

FIXED_BUCKET_EDGES = [0, 20, 40, 60, 80, 101]
FIXED_BUCKET_LABELS = ["0-19", "20-39", "40-59", "60-79", "80-100"]
QUANTILE_BUCKET_LABELS = [f"Q{i + 1}" for i in range(5)]


def assign_fixed_buckets(total_score: pd.Series) -> pd.Series:
    """Fixed-width buckets over the SCORE'S OWN 0-100 scale (not the
    observed distribution) - see assign_quantile_buckets() for the
    distribution-aware alternative, provided because Score's actual
    distribution may be far from uniform across [0,100].
    """
    return pd.cut(
        total_score, bins=FIXED_BUCKET_EDGES, labels=FIXED_BUCKET_LABELS,
        right=False, include_lowest=True,
    )


def assign_quantile_buckets(total_score: pd.Series, n_buckets: int = 5) -> pd.Series:
    """Buckets of (approximately) equal population size, labelled Q1
    (lowest scores) .. Qn (highest). Falls back to fewer buckets if the
    Score distribution has too many repeated values for `n_buckets`
    distinct quantile edges (pandas' duplicates="drop").
    """
    labels = [f"Q{i + 1}" for i in range(n_buckets)]
    try:
        return pd.qcut(total_score, q=n_buckets, labels=labels, duplicates="drop")
    except ValueError:
        # Fewer unique values than n_buckets - qcut cannot produce even
        # one bucket boundary; every row falls in a single bucket.
        return pd.Series(["Q1"] * len(total_score), index=total_score.index)


def compute_bucket_metrics(
    scored: pd.DataFrame, bucket_col: str, return_col: str
) -> dict[str, BacktestMetrics]:
    """One BacktestMetrics per bucket value present in scored[bucket_col]
    (n/win_rate/average_return/median_return/profit_factor/expectancy -
    reused from backtest/metrics.py rather than re-implemented here).
    Rows where return_col is NaN are dropped before grouping (a Forward
    Return can be NaN near the end of a ticker's history, same as any
    other Feature's warmup-shaped NaN).
    """
    valid = scored.dropna(subset=[return_col])
    if valid.empty:
        return {}
    result = {}
    for bucket, group in valid.groupby(bucket_col, observed=True):
        renamed = group.rename(columns={return_col: "return"})
        result[str(bucket)] = compute_metrics(renamed)
    return result


def check_monotonicity(
    bucket_metrics: dict[str, BacktestMetrics], bucket_order: list[str]
) -> bool | None:
    """True if average_return is non-decreasing from the lowest to the
    highest Score bucket in bucket_order. None if fewer than 2 buckets
    have any trades to compare (too little data to judge either way).
    Non-decreasing (not strictly increasing), since exact ties are
    plausible with limited data and a strict bar would be needlessly
    harsh.
    """
    values: list[float] = []
    for b in bucket_order:
        avg = bucket_metrics[b].average_return if b in bucket_metrics else None
        if avg is not None:
            values.append(avg)
    if len(values) < 2:
        return None
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def monotonicity_correlation(
    bucket_metrics: dict[str, BacktestMetrics], bucket_order: list[str]
) -> float | None:
    """Pearson correlation between a bucket's ordinal rank (0, 1, 2, ...
    in bucket_order) and its average_return - a simple, continuous
    complement to the boolean check_monotonicity(). +1 = perfectly
    monotonic increasing, -1 = perfectly monotonic decreasing, near 0 =
    no relationship. Deliberately NOT called "Spearman": it correlates
    bucket rank against the raw average_return values, not against a
    second rank - a true Spearman test belongs in Phase 6's statistical
    validation, not here.
    """
    ranks: list[int] = []
    returns: list[float] = []
    for i, b in enumerate(bucket_order):
        avg = bucket_metrics[b].average_return if b in bucket_metrics else None
        if avg is not None:
            ranks.append(i)
            returns.append(avg)
    if len(ranks) < 2:
        return None
    if np.std(returns) == 0:
        return None  # every bucket has the identical average return - correlation is undefined
    return float(np.corrcoef(ranks, returns)[0, 1])


def bucket_spread(
    bucket_metrics: dict[str, BacktestMetrics], high_label: str, low_label: str
) -> float | None:
    """Phase 6.5 section 21: average_return(high_label) - average_return
    (low_label) - e.g. Q5-Q1 spread. None if either bucket has no trades.
    """
    high = bucket_metrics.get(high_label)
    low = bucket_metrics.get(low_label)
    if high is None or low is None or high.average_return is None or low.average_return is None:
        return None
    return high.average_return - low.average_return


def bucket_ratio(
    bucket_metrics: dict[str, BacktestMetrics], high_label: str, low_label: str
) -> float | None:
    """average_return(high_label) / average_return(low_label) - e.g.
    Q5/Q1 ratio. None if either bucket has no trades, or if low_label's
    average return is exactly 0 (undefined ratio).
    """
    high = bucket_metrics.get(high_label)
    low = bucket_metrics.get(low_label)
    if high is None or low is None or high.average_return is None or low.average_return is None:
        return None
    if low.average_return == 0:
        return None
    return high.average_return / low.average_return


def extract_bucket_returns(
    scored: pd.DataFrame, bucket_col: str, return_col: str, bucket_label: str
) -> np.ndarray:
    """Raw (NaN-dropped) return values for one bucket - the input a
    bootstrap CI needs (compute_bucket_metrics() only returns the
    already-summarized BacktestMetrics, not the underlying array).
    """
    valid = scored.dropna(subset=[return_col])
    subset = valid[valid[bucket_col].astype(str) == bucket_label]
    return subset[return_col].to_numpy(dtype=float)
