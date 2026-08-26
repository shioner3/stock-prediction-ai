"""v3/residual/residual_strength.py."""

from __future__ import annotations

from v3.residual.residual_strength import residual_strength


def test_ratio_computed_normally() -> None:
    assert abs(residual_strength(0.002, 0.004) - 0.5) < 1e-9


def test_negative_ratio_when_signs_flip() -> None:
    assert residual_strength(-0.002, 0.004) == -0.5


def test_na_when_original_near_zero() -> None:
    assert residual_strength(0.001, 0.00001) is None


def test_na_when_either_input_is_none() -> None:
    assert residual_strength(None, 0.01) is None
    assert residual_strength(0.01, None) is None
