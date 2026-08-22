"""Bootstrap confidence intervals (Phase 6 section 11-12) for trade-
return statistics, via resampling WITH replacement from the OBSERVED
trade set. This is NOT a null-hypothesis test (see backtest/permutation.py
for that) - it only quantifies how much the point estimate itself might
vary given sampling noise in the trades actually observed.

Deterministic given (data, config.seed): same input array + same seed
always produces the exact same CI (tests/test_backtest_bootstrap.py
verifies this directly). n_resamples/seed/confidence_level are fixed in
config and must never be changed to chase a more favorable interval
(Phase 6 section 12).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.loader import BootstrapConfig

STATISTIC_NAMES = ("mean_return", "profit_factor", "expectancy")

# A single (n_resamples, n) float64 resample matrix must stay well under
# available RAM - a Signal that triggers very frequently (e.g. tens of
# thousands of rows across the Full Universe) combined with
# n_resamples=10_000 can otherwise demand tens of GB in one allocation
# (see backtest/bootstrap.py's Phase 11 fix - long_momentum_continuation's
# forward-horizon population hit a 35GB single-array OOM before this
# chunking was added). Bounding chunk_elements keeps peak memory to a few
# hundred MB regardless of n, at the cost of a Python-level loop over
# chunks - negligible next to the OOM it prevents. Chosen so every
# existing (data, config) combination in this codebase still fits in ONE
# chunk (chunk_size == n_resamples), which keeps results BIT-IDENTICAL to
# the pre-chunking implementation for every case tests/backtest already
# cover - chunking a numpy Generator's sequential integer stream only
# changes results when it actually splits work across multiple calls.
_CHUNK_ELEMENT_BUDGET = 50_000_000


def _resample_stat_chunked(
    values: npt.NDArray[np.float64],
    n_resamples: int,
    rng: np.random.Generator,
    stat_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    n = len(values)
    chunk_size = max(1, min(n_resamples, _CHUNK_ELEMENT_BUDGET // max(n, 1)))
    stats = np.empty(n_resamples, dtype=float)
    start = 0
    while start < n_resamples:
        batch = min(chunk_size, n_resamples - start)
        resample_indices = rng.integers(0, n, size=(batch, n))
        stats[start : start + batch] = stat_fn(values[resample_indices])
        start += batch
    return stats


@dataclass(frozen=True)
class BootstrapResult:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    n_observations: int


def _mean_stat(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return samples.mean(axis=1)


def _profit_factor_stat(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    wins = np.where(samples > 0, samples, 0.0).sum(axis=1)
    losses = np.where(samples < 0, samples, 0.0).sum(axis=1)  # <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            losses < 0, wins / np.abs(losses), np.where(wins > 0, np.inf, np.nan)
        )
    return result


def _single_profit_factor(values: npt.NDArray[np.float64]) -> float:
    wins = values[values > 0].sum()
    losses = values[values < 0].sum()
    if losses < 0:
        return float(wins / abs(losses))
    return float("inf") if wins > 0 else float("nan")


def bootstrap_ci(
    returns: npt.ArrayLike, metric_name: str, config: BootstrapConfig
) -> BootstrapResult:
    if metric_name not in STATISTIC_NAMES:
        raise ValueError(f"metric_name={metric_name!r} must be one of {STATISTIC_NAMES}")

    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return BootstrapResult(
            metric_name, float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, 0,
        )

    if metric_name == "profit_factor":
        point_estimate = _single_profit_factor(values)
    else:
        point_estimate = float(values.mean())

    rng = np.random.default_rng(config.seed)
    stat_fn = _profit_factor_stat if metric_name == "profit_factor" else _mean_stat
    resample_stats = _resample_stat_chunked(values, config.n_resamples, rng, stat_fn)

    finite_stats = resample_stats[np.isfinite(resample_stats)]
    alpha = 1.0 - config.confidence_level
    if len(finite_stats) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite_stats, alpha / 2))
        ci_high = float(np.quantile(finite_stats, 1 - alpha / 2))

    return BootstrapResult(
        metric_name=metric_name,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence_level,
        n_observations=n,
    )


@dataclass(frozen=True)
class BootstrapDiffResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    n_observations_high: int
    n_observations_low: int


def bootstrap_diff_ci(
    high_values: npt.ArrayLike, low_values: npt.ArrayLike, config: BootstrapConfig
) -> BootstrapDiffResult:
    """Two-sample bootstrap CI for the DIFFERENCE OF MEANS (high - low) -
    Phase 6.5 section 21's "Q5-Q1 bootstrap CI" (the spread's own
    confidence interval, not two separate single-sample CIs). Each
    sample is resampled with replacement INDEPENDENTLY of the other (the
    standard two-sample bootstrap), from the same seeded Generator so the
    whole computation stays deterministic given (data, config.seed).
    """
    high = np.asarray(high_values, dtype=float)
    high = high[~np.isnan(high)]
    low = np.asarray(low_values, dtype=float)
    low = low[~np.isnan(low)]
    n_high, n_low = len(high), len(low)

    if n_high == 0 or n_low == 0:
        return BootstrapDiffResult(
            float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, n_high, n_low,
        )

    point_estimate = float(high.mean() - low.mean())

    rng = np.random.default_rng(config.seed)
    # Draws ALL of the high resample means first (chunked), THEN all of
    # the low resample means (chunked) - matching the pre-chunking
    # implementation's two-phase draw order exactly, so this stays
    # bit-identical to it for ANY chunk size (unlike interleaving high
    # and low draws chunk-by-chunk, which would change which underlying
    # random draws land in which sample - see _resample_stat_chunked's
    # docstring for why sequential, non-interleaved chunking preserves
    # the exact draw sequence).
    high_means = _resample_stat_chunked(high, config.n_resamples, rng, _mean_stat)
    low_means = _resample_stat_chunked(low, config.n_resamples, rng, _mean_stat)
    diffs = high_means - low_means

    alpha = 1.0 - config.confidence_level
    ci_low = float(np.quantile(diffs, alpha / 2))
    ci_high = float(np.quantile(diffs, 1 - alpha / 2))

    return BootstrapDiffResult(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence_level,
        n_observations_high=n_high,
        n_observations_low=n_low,
    )
