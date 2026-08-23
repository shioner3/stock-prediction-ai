"""Gini coefficient / contribution ranking (v3/robustness/gini.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from v3.robustness.gini import compute_contribution_ranking, gini_coefficient, lorenz_curve


def test_gini_perfect_equality_is_zero() -> None:
    values = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    assert gini_coefficient(values) == 0.0


def test_gini_max_concentration_approaches_one() -> None:
    values = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
    g = gini_coefficient(values)
    assert g is not None
    assert g > 0.7  # heavily concentrated, well above a moderate-inequality value


def test_gini_handles_negative_values_via_shift() -> None:
    # Shifting every value by a constant should not change which
    # days/tickers contribute more than others - Gini computed on a
    # negative-containing series should equal Gini on the shifted
    # non-negative version.
    values = np.array([-5.0, -5.0, -5.0, -5.0, 15.0])
    shifted = values - values.min()
    assert gini_coefficient(values) == gini_coefficient(shifted)


def test_gini_empty_returns_none() -> None:
    assert gini_coefficient(np.array([])) is None


def test_lorenz_curve_starts_and_ends_at_boundary() -> None:
    pop, val = lorenz_curve(np.array([1.0, 2.0, 3.0, 4.0]))
    assert pop[0] == 0.0 and val[0] == 0.0
    assert pop[-1] == 1.0
    assert abs(val[-1] - 1.0) < 1e-9


def test_compute_contribution_ranking_sorted_descending() -> None:
    df = pd.DataFrame({
        "ticker": ["A", "A", "B", "C", "C", "C"],
        "actual": [0.01, 0.02, 0.10, -0.01, -0.01, -0.01],
    })
    ranking = compute_contribution_ranking(df, "ticker", "actual")
    assert list(ranking.index) == ["B", "A", "C"]
    assert ranking.loc["B"] == 0.10
    assert abs(ranking.loc["A"] - 0.03) < 1e-9
    assert abs(ranking.loc["C"] - (-0.03)) < 1e-9
