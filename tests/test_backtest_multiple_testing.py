from __future__ import annotations

import math

from backtest.multiple_testing import benjamini_hochberg_correction


def test_empty_input_gives_empty_result() -> None:
    assert benjamini_hochberg_correction({}) == {}


def test_nan_and_none_entries_are_dropped() -> None:
    result = benjamini_hochberg_correction({"a": 0.01, "b": float("nan"), "c": None})  # type: ignore[dict-item]
    assert set(result.keys()) == {"a"}
    assert result["a"].n_tests == 1


def test_single_p_value_adjusted_equals_raw() -> None:
    result = benjamini_hochberg_correction({"a": 0.03})
    assert result["a"].adjusted_p_value == 0.03
    assert result["a"].rank == 1
    assert result["a"].n_tests == 1


def test_hand_computed_bh_adjustment() -> None:
    # Classic textbook example: raw p-values 0.01, 0.02, 0.03, 0.04, 0.50
    # (n=5). BH q(i) = min_{j>=i}(p(j) * n / j):
    #   rank5: 0.50*5/5=0.50            -> q5=0.50
    #   rank4: 0.04*5/4=0.05  min(0.05,0.50)=0.05  -> q4=0.05
    #   rank3: 0.03*5/3=0.05  min(0.05,0.05)=0.05  -> q3=0.05
    #   rank2: 0.02*5/2=0.05  min(0.05,0.05)=0.05  -> q2=0.05
    #   rank1: 0.01*5/1=0.05  min(0.05,0.05)=0.05  -> q1=0.05
    p_values = {"s1": 0.01, "s2": 0.02, "s3": 0.03, "s4": 0.04, "s5": 0.50}
    result = benjamini_hochberg_correction(p_values)
    assert math.isclose(result["s1"].adjusted_p_value, 0.05)
    assert math.isclose(result["s2"].adjusted_p_value, 0.05)
    assert math.isclose(result["s3"].adjusted_p_value, 0.05)
    assert math.isclose(result["s4"].adjusted_p_value, 0.05)
    assert math.isclose(result["s5"].adjusted_p_value, 0.50)
    assert result["s1"].rank == 1
    assert result["s5"].rank == 5
    assert all(r.n_tests == 5 for r in result.values())


def test_adjusted_p_values_are_monotone_non_decreasing_by_rank() -> None:
    p_values = {f"s{i}": p for i, p in enumerate([0.001, 0.2, 0.05, 0.9, 0.3, 0.01])}
    result = benjamini_hochberg_correction(p_values)
    by_rank = sorted(result.values(), key=lambda r: r.rank)
    adjusted = [r.adjusted_p_value for r in by_rank]
    assert all(adjusted[i] <= adjusted[i + 1] for i in range(len(adjusted) - 1))


def test_adjusted_p_value_never_below_raw_p_value() -> None:
    p_values = {"a": 0.001, "b": 0.5, "c": 0.02}
    result = benjamini_hochberg_correction(p_values)
    for r in result.values():
        assert r.adjusted_p_value >= r.raw_p_value - 1e-12


def test_significant_flag_uses_alpha_threshold() -> None:
    p_values = {"a": 0.001, "b": 0.5}
    result = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert result["a"].significant is True
    assert result["b"].significant is False


def test_all_identical_p_values_all_adjusted_equal() -> None:
    p_values = {f"s{i}": 0.04 for i in range(5)}
    result = benjamini_hochberg_correction(p_values)
    adjusted_values = {r.adjusted_p_value for r in result.values()}
    assert len(adjusted_values) == 1
    assert math.isclose(next(iter(adjusted_values)), 0.04)
