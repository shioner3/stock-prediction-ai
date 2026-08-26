"""Q5-Q1 spread bootstrap CI (spec section 16): trade-level, day-cluster,
and block bootstrap for the DIFFERENCE of Q5's and Q1's mean Forward
Return.

Trade-level reuses backtest.bootstrap.bootstrap_diff_ci() directly
(unmodified V1 code - Phase 6.5's own "Q5-Q1 bootstrap CI" primitive,
literally built for this exact purpose already).

Day-cluster and block bootstrap have no existing two-sample-difference
equivalent in V1 (backtest/day_cluster_bootstrap.py and backtest/
block_bootstrap.py each resample ONE group's days to CI a single-group
statistic) - spec section 16 explicitly asks for the SPREAD's own CI
under day/block resampling, so this module adds that as NEW code,
mirroring V1's two modules' exact resampling algorithm (draw day
indices from a seeded Generator, either individually or as contiguous
blocks) but drawing the SAME resampled day sequence for BOTH the Q5
and Q1 groups per resample - this is deliberate, not an approximation:
Q1 and Q5 rows on the same calendar day are correlated (a market-wide
move on that day affects both buckets simultaneously), so pairing the
day draw across groups is the more correct design, not merely a
convenience. Only days present in BOTH groups are eligible for
resampling (spec's "同日" pairing has no meaning on a day where one
bucket has zero rows - vanishingly rare at Full Universe scale but
handled explicitly rather than silently).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd

from config.loader import BlockBootstrapConfig, DayClusterBootstrapConfig


@dataclass(frozen=True)
class SpreadBootstrapResult:
    method: str  # "day_cluster" or "block"
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    n_days: int
    n_high_obs: int
    n_low_obs: int
    block_length_days: int | None


def _returns_by_day(df: pd.DataFrame, date_col: str) -> dict[date_type, np.ndarray]:
    valid = df.dropna(subset=["return", date_col])
    return {day: g["return"].to_numpy(dtype=float) for day, g in valid.groupby(date_col)}


def day_cluster_spread_bootstrap(
    high: pd.DataFrame, low: pd.DataFrame, config: DayClusterBootstrapConfig,
    date_col: str = "date",
) -> SpreadBootstrapResult:
    high_by_day = _returns_by_day(high, date_col)
    low_by_day = _returns_by_day(low, date_col)
    shared_days = np.array(sorted(set(high_by_day) & set(low_by_day)), dtype=object)
    n_days = len(shared_days)

    if n_days == 0:
        return SpreadBootstrapResult(
            "day_cluster", float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, 0,
            sum(len(v) for v in high_by_day.values()), sum(len(v) for v in low_by_day.values()),
            None,
        )

    point_estimate = float(
        np.concatenate([high_by_day[d] for d in shared_days]).mean()
        - np.concatenate([low_by_day[d] for d in shared_days]).mean()
    )

    rng = np.random.default_rng(config.seed)
    day_pick_indices = rng.integers(0, n_days, size=(config.n_resamples, n_days))

    diffs = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        chosen = shared_days[day_pick_indices[i]]
        high_mean = np.concatenate([high_by_day[d] for d in chosen]).mean()
        low_mean = np.concatenate([low_by_day[d] for d in chosen]).mean()
        diffs[i] = high_mean - low_mean

    alpha = 1.0 - config.confidence_level
    finite = diffs[np.isfinite(diffs)]
    ci_low, ci_high = (
        (float(np.quantile(finite, alpha / 2)), float(np.quantile(finite, 1 - alpha / 2)))
        if len(finite) else (float("nan"), float("nan"))
    )

    return SpreadBootstrapResult(
        method="day_cluster", point_estimate=point_estimate, ci_low=ci_low, ci_high=ci_high,
        n_resamples=config.n_resamples, confidence_level=config.confidence_level, n_days=n_days,
        n_high_obs=sum(len(v) for v in high_by_day.values()),
        n_low_obs=sum(len(v) for v in low_by_day.values()), block_length_days=None,
    )


def block_spread_bootstrap(
    high: pd.DataFrame, low: pd.DataFrame, config: BlockBootstrapConfig, date_col: str = "date",
) -> SpreadBootstrapResult:
    high_by_day = _returns_by_day(high, date_col)
    low_by_day = _returns_by_day(low, date_col)
    shared_days = np.array(sorted(set(high_by_day) & set(low_by_day)), dtype=object)
    n_days = len(shared_days)

    if n_days == 0:
        return SpreadBootstrapResult(
            "block", float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, 0,
            sum(len(v) for v in high_by_day.values()), sum(len(v) for v in low_by_day.values()),
            config.block_length_days,
        )

    point_estimate = float(
        np.concatenate([high_by_day[d] for d in shared_days]).mean()
        - np.concatenate([low_by_day[d] for d in shared_days]).mean()
    )

    block_length = min(config.block_length_days, n_days)
    n_blocks = max(1, math.ceil(n_days / block_length))
    max_start = n_days - block_length

    rng = np.random.default_rng(config.seed)
    diffs = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        chosen = [d for s in starts for d in shared_days[s : s + block_length]]
        high_mean = np.concatenate([high_by_day[d] for d in chosen]).mean()
        low_mean = np.concatenate([low_by_day[d] for d in chosen]).mean()
        diffs[i] = high_mean - low_mean

    alpha = 1.0 - config.confidence_level
    finite = diffs[np.isfinite(diffs)]
    ci_low, ci_high = (
        (float(np.quantile(finite, alpha / 2)), float(np.quantile(finite, 1 - alpha / 2)))
        if len(finite) else (float("nan"), float("nan"))
    )

    return SpreadBootstrapResult(
        method="block", point_estimate=point_estimate, ci_low=ci_low, ci_high=ci_high,
        n_resamples=config.n_resamples, confidence_level=config.confidence_level, n_days=n_days,
        n_high_obs=sum(len(v) for v in high_by_day.values()),
        n_low_obs=sum(len(v) for v in low_by_day.values()), block_length_days=block_length,
    )
