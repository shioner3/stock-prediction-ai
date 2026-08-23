from __future__ import annotations

import numpy as np
import pytest

from v2.validation.monotonicity import compute_monotonicity, kendall_tau_b, spearman_on_points


def test_perfectly_monotonic_buckets() -> None:
    r = compute_monotonicity({"Q1": 0.0, "Q2": 0.01, "Q3": 0.02, "Q4": 0.03, "Q5": 0.04}, 5)
    assert r.is_monotonic_nondecreasing is True
    assert r.spearman == pytest.approx(1.0)
    assert r.kendall == 1.0
    assert abs(r.q5_q1_spread - 0.04) < 1e-12
    assert r.non_monotonic_pattern is None


def test_reversed_buckets_flagged_q1_highest() -> None:
    r = compute_monotonicity({"Q1": 0.05, "Q2": 0.03, "Q3": 0.02, "Q4": 0.01, "Q5": 0.0}, 5)
    assert r.is_monotonic_nondecreasing is False
    assert r.non_monotonic_pattern == "Q1が最も高い(逆行)"
    assert r.spearman == pytest.approx(-1.0)


def test_middle_peak_flagged() -> None:
    r = compute_monotonicity({"Q1": 0.0, "Q2": 0.01, "Q3": 0.05, "Q4": 0.01, "Q5": 0.02}, 5)
    assert r.is_monotonic_nondecreasing is False
    assert "Q3" in r.non_monotonic_pattern


def test_missing_bucket_returns_none_result() -> None:
    r = compute_monotonicity({"Q1": 0.0, "Q2": 0.01, "Q3": None, "Q4": 0.03, "Q5": 0.04}, 5)
    assert r.is_monotonic_nondecreasing is None
    assert r.spearman is None
    assert r.kendall is None


def test_kendall_tau_b_perfect_agreement() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kendall_tau_b(x, y) == 1.0


def test_kendall_tau_b_perfect_disagreement() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert kendall_tau_b(x, y) == -1.0


def test_kendall_tau_b_handles_ties() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 3.0, 4.0, 5.0])
    tau = kendall_tau_b(x, y)
    assert tau is not None
    assert 0.9 < tau < 1.0


def test_spearman_on_points_constant_y_gives_none() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 1.0])
    assert spearman_on_points(x, y) is None
