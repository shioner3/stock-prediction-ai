from __future__ import annotations

import pandas as pd

from v2.validation.cost import COST_TIERS, compute_cost_sensitivity


def test_four_tiers_present() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    results = compute_cost_sensitivity(returns)
    assert [r.tier_name for r in results] == ["zero", "low", "base", "high"]
    assert [r.round_trip_bps for r in results] == [0.0, 10.0, 30.0, 80.0]


def test_zero_tier_matches_gross() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    results = compute_cost_sensitivity(returns)
    zero = next(r for r in results if r.tier_name == "zero")
    assert abs(zero.net_stats.mean_return - zero.gross_stats.mean_return) < 1e-12


def test_higher_cost_tier_reduces_net_return_more() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    results = compute_cost_sensitivity(returns)
    by_tier = {r.tier_name: r.net_stats.mean_return for r in results}
    assert by_tier["zero"] > by_tier["low"] > by_tier["base"] > by_tier["high"]


def test_cost_can_flip_positive_edge_negative() -> None:
    returns = pd.Series([0.002, 0.001, -0.001, 0.0015])  # tiny mean edge
    results = compute_cost_sensitivity(returns)
    high = next(r for r in results if r.tier_name == "high")
    assert high.net_stats.mean_return < 0


def test_default_tiers_are_the_module_constant() -> None:
    returns = pd.Series([0.01])
    results = compute_cost_sensitivity(returns, tiers=COST_TIERS)
    assert len(results) == 4
