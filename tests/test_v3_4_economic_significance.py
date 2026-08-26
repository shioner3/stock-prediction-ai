"""v3/robustness/economic_significance.py - Max Losing Streak and
Annualized Return, the 2 genuinely new metrics this module adds.
"""

from __future__ import annotations

import numpy as np

from v3.robustness.economic_significance import annualized_return, max_losing_streak


def test_max_losing_streak_finds_longest_run() -> None:
    returns = np.array([0.01, -0.01, -0.02, -0.01, 0.03, -0.01, -0.01, -0.01, -0.01, 0.02])
    assert max_losing_streak(returns) == 4


def test_max_losing_streak_zero_when_no_losses() -> None:
    returns = np.array([0.01, 0.02, 0.0, 0.03])
    assert max_losing_streak(returns) == 0


def test_max_losing_streak_all_losses() -> None:
    returns = np.array([-0.01, -0.02, -0.01])
    assert max_losing_streak(returns) == 3


def test_annualized_return_compounds_correctly() -> None:
    # 1% every 5 trading days, 252 trading days/year -> 50.4 periods/year
    result = annualized_return(0.01, window_days=5)
    expected = (1.01) ** (252 / 5) - 1
    assert abs(result - expected) < 1e-12


def test_annualized_return_zero_mean_is_zero() -> None:
    assert abs(annualized_return(0.0, window_days=5)) < 1e-12
