from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.loader import BlockBootstrapConfig, DayClusterBootstrapConfig
from v2.validation.spread_bootstrap import block_spread_bootstrap, day_cluster_spread_bootstrap


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def _grouped(dates: list[date], per_day: int, mean: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(d, v) for d in dates for v in rng.normal(mean, 0.01, per_day)]
    return pd.DataFrame({"date": [r[0] for r in rows], "return": [r[1] for r in rows]})


def test_day_cluster_spread_detects_real_difference() -> None:
    dates = _dates(40)
    high = _grouped(dates, 5, mean=0.03, seed=1)
    low = _grouped(dates, 5, mean=0.0, seed=2)
    result = day_cluster_spread_bootstrap(
        high, low, DayClusterBootstrapConfig(n_resamples=500, seed=7)
    )
    assert result.point_estimate > 0.02
    assert result.ci_low > 0  # CI should exclude zero given a real ~3% gap
    assert result.n_days == 40


def test_block_spread_detects_real_difference() -> None:
    dates = _dates(40)
    high = _grouped(dates, 5, mean=0.03, seed=1)
    low = _grouped(dates, 5, mean=0.0, seed=2)
    result = block_spread_bootstrap(
        high, low, BlockBootstrapConfig(n_resamples=500, block_length_days=5, seed=8)
    )
    assert result.point_estimate > 0.02
    assert result.ci_low > 0
    assert result.block_length_days == 5


def test_day_cluster_spread_no_real_difference_ci_spans_zero() -> None:
    dates = _dates(40)
    high = _grouped(dates, 5, mean=0.0, seed=1)
    low = _grouped(dates, 5, mean=0.0, seed=2)
    result = day_cluster_spread_bootstrap(
        high, low, DayClusterBootstrapConfig(n_resamples=1000, seed=9)
    )
    assert result.ci_low < 0 < result.ci_high


def test_deterministic_same_seed_gives_identical_result() -> None:
    dates = _dates(20)
    high = _grouped(dates, 3, mean=0.02, seed=1)
    low = _grouped(dates, 3, mean=0.0, seed=2)
    config = DayClusterBootstrapConfig(n_resamples=300, seed=11)
    a = day_cluster_spread_bootstrap(high, low, config)
    b = day_cluster_spread_bootstrap(high, low, config)
    assert a == b


def test_only_shared_days_are_used() -> None:
    """A day present in only one of the two groups must not be treated
    as a shared resampling unit.
    """
    high = pd.DataFrame({"date": [date(2024, 1, 1), date(2024, 1, 2)], "return": [0.01, 0.02]})
    low = pd.DataFrame({"date": [date(2024, 1, 1)], "return": [0.0]})
    result = day_cluster_spread_bootstrap(high, low, DayClusterBootstrapConfig(n_resamples=100))
    assert result.n_days == 1


def test_empty_shared_days_gives_nan_result() -> None:
    high = pd.DataFrame({"date": [date(2024, 1, 1)], "return": [0.01]})
    low = pd.DataFrame({"date": [date(2024, 1, 2)], "return": [0.0]})
    result = day_cluster_spread_bootstrap(high, low, DayClusterBootstrapConfig(n_resamples=100))
    assert result.n_days == 0
    assert np.isnan(result.point_estimate)


def test_block_length_capped_at_available_days() -> None:
    dates = _dates(3)
    high = _grouped(dates, 2, mean=0.01, seed=1)
    low = _grouped(dates, 2, mean=0.0, seed=2)
    result = block_spread_bootstrap(
        high, low, BlockBootstrapConfig(n_resamples=50, block_length_days=10)
    )
    assert result.block_length_days == 3
