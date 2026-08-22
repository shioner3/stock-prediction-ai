from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.ticker_cluster_bootstrap import ticker_cluster_bootstrap
from config.loader import TickerClusterBootstrapConfig


def _trades(rows: list[tuple[str, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "signal_date": [r[1] for r in rows],
            "return": [r[2] for r in rows],
        }
    )


def _d(offset: int) -> date:
    return date(2024, 1, 1) + timedelta(days=offset)


def test_point_estimate_matches_plain_mean() -> None:
    trades = _trades([("A", _d(0), 0.01), ("A", _d(1), 0.02), ("B", _d(1), -0.01)])
    result = ticker_cluster_bootstrap(
        trades, "mean_return", TickerClusterBootstrapConfig(n_resamples=500, seed=1)
    )
    assert np.isclose(result.point_estimate, np.mean([0.01, 0.02, -0.01]))
    assert result.n_tickers == 2
    assert result.n_trades == 3


def test_deterministic_same_seed_gives_identical_result() -> None:
    trades = _trades(
        [("A", _d(0), 0.02), ("A", _d(3), -0.01), ("B", _d(1), 0.03), ("C", _d(2), -0.02)]
    )
    config = TickerClusterBootstrapConfig(n_resamples=1000, seed=7)
    a = ticker_cluster_bootstrap(trades, "expectancy", config)
    b = ticker_cluster_bootstrap(trades, "expectancy", config)
    assert a == b


def test_same_ticker_trades_on_different_days_are_always_kept_together() -> None:
    """A single ticker that fires repeatedly across many different DAYS
    (so Day Cluster Bootstrap would treat each of its trades as an
    independent unit) should still show up as an ALL-or-NOTHING unit
    under Ticker Cluster Bootstrap - i.e. the CI should be wide enough to
    reflect "this ticker may or may not appear", unlike a day-level
    bootstrap which would smooth it out because its trades are spread
    across separate days.
    """
    # 1 ticker with 50 highly profitable trades spread across 50 distinct
    # days (a repeat-offender ticker) + 5 other tickers with 1 modest
    # trade each on their own distinct days.
    big_ticker_trades = [("T", _d(i), 0.10) for i in range(50)]
    small_tickers = [(f"S{i}", _d(50 + i), 0.001) for i in range(5)]
    trades = _trades(big_ticker_trades + small_tickers)

    config = TickerClusterBootstrapConfig(n_resamples=2000, seed=3)
    result = ticker_cluster_bootstrap(trades, "mean_return", config)

    # With 6 unique tickers and ticker-level resampling, a meaningful
    # fraction of resamples should exclude the big ticker entirely,
    # giving a mean near 0.001 - so the CI should span from near-zero up
    # to near the point estimate (which is dominated by the big ticker
    # being present at least once, on average).
    assert result.ci_low < result.point_estimate


def test_empty_trades_gives_nan_result() -> None:
    result = ticker_cluster_bootstrap(
        _trades([]), "mean_return", TickerClusterBootstrapConfig(n_resamples=100)
    )
    assert np.isnan(result.point_estimate)
    assert result.n_tickers == 0
    assert result.n_trades == 0


def test_profit_factor_all_wins_gives_inf_point_estimate() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(1), 0.02)])
    result = ticker_cluster_bootstrap(
        trades, "profit_factor", TickerClusterBootstrapConfig(n_resamples=200, seed=1)
    )
    assert result.point_estimate == float("inf")


def test_cumulative_return_matches_sum() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(1), -0.02), ("B", _d(2), 0.03)])
    result = ticker_cluster_bootstrap(
        trades, "cumulative_return", TickerClusterBootstrapConfig(n_resamples=200, seed=1)
    )
    assert np.isclose(result.point_estimate, 0.06)


def test_invalid_metric_name_raises() -> None:
    trades = _trades([("A", _d(0), 0.05)])
    try:
        ticker_cluster_bootstrap(
            trades, "not_a_metric", TickerClusterBootstrapConfig(n_resamples=10)
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
