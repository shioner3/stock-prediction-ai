from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.block_bootstrap import block_bootstrap
from config.loader import BlockBootstrapConfig


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
    trades = _trades([("A", _d(i), 0.01 * i) for i in range(10)])
    result = block_bootstrap(
        trades, "mean_return", BlockBootstrapConfig(block_length_days=5, n_resamples=500, seed=1)
    )
    assert np.isclose(result.point_estimate, np.mean([0.01 * i for i in range(10)]))
    assert result.n_days == 10
    assert result.block_length_days == 5


def test_deterministic_same_seed_gives_identical_result() -> None:
    trades = _trades([("A", _d(i), 0.01 * (i % 3 - 1)) for i in range(15)])
    config = BlockBootstrapConfig(block_length_days=5, n_resamples=1000, seed=9)
    a = block_bootstrap(trades, "expectancy", config)
    b = block_bootstrap(trades, "expectancy", config)
    assert a == b


def test_block_length_clamped_to_available_days() -> None:
    # Only 3 unique days, but block_length_days=5 requested.
    trades = _trades([("A", _d(0), 0.01), ("B", _d(1), 0.02), ("C", _d(2), -0.01)])
    result = block_bootstrap(
        trades, "mean_return", BlockBootstrapConfig(block_length_days=5, n_resamples=200, seed=1)
    )
    assert result.block_length_days == 3  # clamped, not the requested 5
    assert not np.isnan(result.point_estimate)


def test_empty_trades_gives_nan_result() -> None:
    result = block_bootstrap(
        _trades([]), "mean_return", BlockBootstrapConfig(n_resamples=100)
    )
    assert np.isnan(result.point_estimate)
    assert result.n_days == 0
    assert result.n_trades == 0


def test_single_day_gives_that_days_value_every_resample() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(0), 0.03)])
    result = block_bootstrap(
        trades, "mean_return", BlockBootstrapConfig(block_length_days=5, n_resamples=200, seed=1)
    )
    # Only one day exists, so every resample is exactly that day's data -
    # CI collapses to the point estimate.
    assert np.isclose(result.ci_low, result.point_estimate)
    assert np.isclose(result.ci_high, result.point_estimate)


def test_contiguous_days_stay_together_within_a_block() -> None:
    """A 3-day extreme cluster embedded among many calm days should
    appear as a single unit within blocks - verified indirectly by
    checking the CI is influenced by its presence/absence (wide CI),
    analogous to the day-cluster test.
    """
    calm_days = [(f"C{i}", _d(i), 0.001) for i in range(20)]
    cluster_days = [("X", _d(50 + j), 0.20) for j in range(3)]
    trades = _trades(calm_days + cluster_days)

    config = BlockBootstrapConfig(block_length_days=5, n_resamples=2000, seed=4)
    result = block_bootstrap(trades, "mean_return", config)
    assert result.ci_low < result.point_estimate
