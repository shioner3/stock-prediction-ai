from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.costs import apply_cost
from config.loader import TransactionCostTierConfig


def test_zero_cost_leaves_returns_unchanged() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    tier = TransactionCostTierConfig(name="zero", round_trip_bps=0.0)
    result = apply_cost(returns, tier)
    pd.testing.assert_series_equal(result, returns)


def test_cost_subtracts_bps_as_a_fraction() -> None:
    returns = pd.Series([0.01, -0.02])
    tier = TransactionCostTierConfig(name="base", round_trip_bps=30.0)
    result = apply_cost(returns, tier)
    assert np.isclose(result.iloc[0], 0.01 - 0.003)
    assert np.isclose(result.iloc[1], -0.02 - 0.003)


def test_higher_cost_tier_always_reduces_return_more() -> None:
    returns = pd.Series([0.05] * 3)
    low = apply_cost(returns, TransactionCostTierConfig(name="low", round_trip_bps=10.0))
    high = apply_cost(returns, TransactionCostTierConfig(name="high", round_trip_bps=80.0))
    assert (high < low).all()
