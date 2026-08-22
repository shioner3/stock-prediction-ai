"""Phase 14 section 15: Ticker Cluster Bootstrap.

backtest/day_cluster_bootstrap.py resamples TRADING DAYS with replacement,
protecting against a single market-day event (e.g. 2024-08) dominating
the trade-level bootstrap. It does NOT protect against the mirror-image
pseudo-replication risk: a single TICKER that happens to trigger
long_oversold_rebound repeatedly could dominate a resample even when its
individual trades land on different days, understating the true
uncertainty if that ticker's trades are correlated with each other
(same company, same sector exposure, same idiosyncratic volatility
regime) in a way independent trades would not be. Ticker Cluster
Bootstrap resamples TICKERS with replacement instead: every trade from a
chosen ticker is always kept together, so a single repeat-offender
ticker is either included as a whole unit or not at all in a given
resample.

Mirrors day_cluster_bootstrap.py's exact resampling algorithm and
structure (same compute_stat()/STATISTIC_NAMES, same CI methodology),
imported directly rather than duplicated - only the grouping key
(ticker column instead of date column) differs.

Added as a NEW module rather than changing backtest/day_cluster_bootstrap.py
or backtest/bootstrap.py - both existing bootstraps remain exactly as
Phase 6-9 used them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.day_cluster_bootstrap import STATISTIC_NAMES, compute_stat
from config.loader import TickerClusterBootstrapConfig

__all__ = ["STATISTIC_NAMES", "TickerClusterBootstrapResult", "ticker_cluster_bootstrap"]


@dataclass(frozen=True)
class TickerClusterBootstrapResult:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    n_tickers: int
    n_trades: int


def ticker_cluster_bootstrap(
    trades: pd.DataFrame,
    metric_name: str,
    config: TickerClusterBootstrapConfig,
    ticker_col: str = "ticker",
) -> TickerClusterBootstrapResult:
    if metric_name not in STATISTIC_NAMES:
        raise ValueError(f"metric_name={metric_name!r} must be one of {STATISTIC_NAMES}")

    valid = trades.dropna(subset=["return", ticker_col])
    n_trades = len(valid)
    if n_trades == 0:
        return TickerClusterBootstrapResult(
            metric_name, float("nan"), float("nan"), float("nan"),
            config.n_resamples, config.confidence_level, 0, 0,
        )

    returns_by_ticker: dict[str, np.ndarray] = {
        ticker: group["return"].to_numpy(dtype=float)
        for ticker, group in valid.groupby(ticker_col)
    }
    unique_tickers = np.array(sorted(returns_by_ticker.keys()), dtype=object)
    n_tickers = len(unique_tickers)

    point_estimate = compute_stat(valid["return"].to_numpy(dtype=float), metric_name)

    rng = np.random.default_rng(config.seed)
    ticker_pick_indices = rng.integers(0, n_tickers, size=(config.n_resamples, n_tickers))

    resample_stats = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        chosen_tickers = unique_tickers[ticker_pick_indices[i]]
        combined = np.concatenate([returns_by_ticker[t] for t in chosen_tickers])
        resample_stats[i] = compute_stat(combined, metric_name)

    finite_stats = resample_stats[np.isfinite(resample_stats)]
    alpha = 1.0 - config.confidence_level
    if len(finite_stats) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite_stats, alpha / 2))
        ci_high = float(np.quantile(finite_stats, 1 - alpha / 2))

    return TickerClusterBootstrapResult(
        metric_name=metric_name,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence_level,
        n_tickers=n_tickers,
        n_trades=n_trades,
    )
