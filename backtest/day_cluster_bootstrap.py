"""Phase 9 section 9: Trading Day Cluster Bootstrap.

backtest/bootstrap.py's existing bootstrap_ci() resamples individual
TRADES, implicitly treating every trade as an independent draw. That
assumption breaks down badly when many different tickers' trades are
driven by the SAME market day - exactly what Phase 8 found for
long_oversold_rebound (53% of its BEAR-regime trades came from a single
9-trading-day event). Day Cluster Bootstrap resamples TRADING DAYS with
replacement instead: every trade that occurred on a chosen day is always
kept together, so a day that happens to be a driver-event day is either
included as a whole unit or not at all in a given resample - it can
never be "diluted" by mixing only some of its trades with an independent
draw from elsewhere.

Added as a NEW module rather than changing backtest/bootstrap.py - the
existing trade-level bootstrap remains exactly as Phase 6-8 used it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd

from config.loader import DayClusterBootstrapConfig

STATISTIC_NAMES = ("mean_return", "profit_factor", "expectancy", "cumulative_return")


@dataclass(frozen=True)
class DayClusterBootstrapResult:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    n_days: int
    n_trades: int


def compute_stat(values: np.ndarray, metric_name: str) -> float:
    if len(values) == 0:
        return float("nan")
    if metric_name in ("mean_return", "expectancy"):
        return float(values.mean())
    if metric_name == "cumulative_return":
        return float(values.sum())
    if metric_name == "profit_factor":
        wins = values[values > 0].sum()
        losses = values[values < 0].sum()
        if losses < 0:
            return float(wins / abs(losses))
        return float("inf") if wins > 0 else float("nan")
    raise ValueError(f"metric_name={metric_name!r} must be one of {STATISTIC_NAMES}")


def day_cluster_bootstrap(
    trades: pd.DataFrame,
    metric_name: str,
    config: DayClusterBootstrapConfig,
    date_col: str = "signal_date",
) -> DayClusterBootstrapResult:
    if metric_name not in STATISTIC_NAMES:
        raise ValueError(f"metric_name={metric_name!r} must be one of {STATISTIC_NAMES}")

    valid = trades.dropna(subset=["return", date_col])
    n_trades = len(valid)
    if n_trades == 0:
        return DayClusterBootstrapResult(
            metric_name, float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, 0, 0,
        )

    returns_by_day: dict[date_type, np.ndarray] = {
        day: group["return"].to_numpy(dtype=float) for day, group in valid.groupby(date_col)
    }
    unique_days = np.array(sorted(returns_by_day.keys()), dtype=object)
    n_days = len(unique_days)

    point_estimate = compute_stat(valid["return"].to_numpy(dtype=float), metric_name)

    rng = np.random.default_rng(config.seed)
    day_pick_indices = rng.integers(0, n_days, size=(config.n_resamples, n_days))

    resample_stats = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        chosen_days = unique_days[day_pick_indices[i]]
        combined = np.concatenate([returns_by_day[d] for d in chosen_days])
        resample_stats[i] = compute_stat(combined, metric_name)

    finite_stats = resample_stats[np.isfinite(resample_stats)]
    alpha = 1.0 - config.confidence_level
    if len(finite_stats) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite_stats, alpha / 2))
        ci_high = float(np.quantile(finite_stats, 1 - alpha / 2))

    return DayClusterBootstrapResult(
        metric_name=metric_name,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence_level,
        n_days=n_days,
        n_trades=n_trades,
    )
