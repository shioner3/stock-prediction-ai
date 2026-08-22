"""Phase 9 section 10: Contiguous Block Bootstrap.

A second, stronger correction beyond Day Cluster Bootstrap
(backtest/day_cluster_bootstrap.py): resamples CONTIGUOUS runs of
block_length_days consecutive trading days (not single days, and not
individual trades) with replacement. This additionally captures
time-adjacent autocorrelation - e.g. a rebound that plays out over
several consecutive days, or a signal that keeps firing on the same
tickers for a few days running - which day-level resampling alone can
still miss if nearby days are themselves correlated.

block_length_days is a FIXED, pre-committed parameter
(config/phase9_settings.yaml, spec section 10: "block lengthを結果に
合わせて変更してはいけない") - never tuned to produce a more favorable CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd

from backtest.day_cluster_bootstrap import STATISTIC_NAMES, compute_stat
from config.loader import BlockBootstrapConfig


@dataclass(frozen=True)
class BlockBootstrapResult:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    block_length_days: int
    n_days: int
    n_trades: int


def block_bootstrap(
    trades: pd.DataFrame,
    metric_name: str,
    config: BlockBootstrapConfig,
    date_col: str = "signal_date",
) -> BlockBootstrapResult:
    if metric_name not in STATISTIC_NAMES:
        raise ValueError(f"metric_name={metric_name!r} must be one of {STATISTIC_NAMES}")

    valid = trades.dropna(subset=["return", date_col])
    n_trades = len(valid)
    if n_trades == 0:
        return BlockBootstrapResult(
            metric_name, float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, config.block_length_days, 0, 0,
        )

    returns_by_day: dict[date_type, np.ndarray] = {
        day: group["return"].to_numpy(dtype=float) for day, group in valid.groupby(date_col)
    }
    unique_days = np.array(sorted(returns_by_day.keys()), dtype=object)
    n_days = len(unique_days)
    # Cannot exceed the number of distinct trading days available.
    block_length = min(config.block_length_days, n_days)
    n_blocks = max(1, math.ceil(n_days / block_length))
    max_start = n_days - block_length  # inclusive upper bound for a block's start index

    point_estimate = compute_stat(valid["return"].to_numpy(dtype=float), metric_name)

    rng = np.random.default_rng(config.seed)
    resample_stats = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        pieces = [
            returns_by_day[d] for s in starts for d in unique_days[s : s + block_length]
        ]
        combined = np.concatenate(pieces) if pieces else np.array([], dtype=float)
        resample_stats[i] = compute_stat(combined, metric_name)

    finite_stats = resample_stats[np.isfinite(resample_stats)]
    alpha = 1.0 - config.confidence_level
    if len(finite_stats) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite_stats, alpha / 2))
        ci_high = float(np.quantile(finite_stats, 1 - alpha / 2))

    return BlockBootstrapResult(
        metric_name=metric_name,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence_level,
        block_length_days=block_length,
        n_days=n_days,
        n_trades=n_trades,
    )
