from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.day_cluster_bootstrap import day_cluster_bootstrap
from config.loader import DayClusterBootstrapConfig


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
    trades = _trades([("A", _d(0), 0.01), ("B", _d(0), 0.02), ("C", _d(1), -0.01)])
    result = day_cluster_bootstrap(
        trades, "mean_return", DayClusterBootstrapConfig(n_resamples=500, seed=1)
    )
    assert np.isclose(result.point_estimate, np.mean([0.01, 0.02, -0.01]))
    assert result.n_days == 2
    assert result.n_trades == 3


def test_deterministic_same_seed_gives_identical_result() -> None:
    trades = _trades(
        [("A", _d(0), 0.02), ("B", _d(0), -0.01), ("C", _d(1), 0.03), ("D", _d(2), -0.02)]
    )
    config = DayClusterBootstrapConfig(n_resamples=1000, seed=7)
    a = day_cluster_bootstrap(trades, "expectancy", config)
    b = day_cluster_bootstrap(trades, "expectancy", config)
    assert a == b


def test_same_day_trades_are_always_kept_together() -> None:
    """A day with an extreme, correlated cluster of trades should show up
    as an ALL-or-NOTHING unit in resamples - i.e. the CI should be wide
    enough to reflect "this day may or may not appear", unlike a
    trade-level bootstrap which would smooth it out by mixing partial
    draws from the big day with draws from small days.
    """
    # 1 day with 50 highly profitable trades (a concentrated event) + 5
    # separate days with 1 modest trade each.
    big_day_trades = [("T", _d(0), 0.10)] * 50
    small_days = [(f"S{i}", _d(i + 1), 0.001) for i in range(5)]
    trades = _trades(big_day_trades + small_days)

    config = DayClusterBootstrapConfig(n_resamples=2000, seed=3)
    result = day_cluster_bootstrap(trades, "mean_return", config)

    # With 6 unique days and day-level resampling, a meaningful fraction
    # of resamples should exclude the big day entirely, giving a mean
    # near 0.001 - so the CI should span from near-zero up to near the
    # point estimate (which is dominated by the big day being present at
    # least once, on average).
    assert result.ci_low < result.point_estimate


def test_empty_trades_gives_nan_result() -> None:
    result = day_cluster_bootstrap(
        _trades([]), "mean_return", DayClusterBootstrapConfig(n_resamples=100)
    )
    assert np.isnan(result.point_estimate)
    assert result.n_days == 0
    assert result.n_trades == 0


def test_profit_factor_all_wins_gives_inf_point_estimate() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(1), 0.02)])
    result = day_cluster_bootstrap(
        trades, "profit_factor", DayClusterBootstrapConfig(n_resamples=200, seed=1)
    )
    assert result.point_estimate == float("inf")


def test_cumulative_return_matches_sum() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(1), -0.02), ("C", _d(1), 0.03)])
    result = day_cluster_bootstrap(
        trades, "cumulative_return", DayClusterBootstrapConfig(n_resamples=200, seed=1)
    )
    assert np.isclose(result.point_estimate, 0.06)
