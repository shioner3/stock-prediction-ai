"""v3/robustness/decision_v3_4.py - the 4-way edge-source classification
tree, exercised through all 4 branches.
"""

from __future__ import annotations

from v3.robustness.decision_v3_4 import (
    EdgeClassification,
    EdgeClassificationInputs,
    classify_edge_source,
)


def test_no_robust_edge_when_original_spread_not_positive() -> None:
    inputs = EdgeClassificationInputs(
        orig_q5_q1_spread=-0.001, beta_adjusted_q5_q1_spread=0.01,
        topix_relative_q5_q1_spread=0.01, bear_excluded_q5_q1_spread=0.01,
        day_top20_excluded_q5_q1_spread=0.01,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.NO_ROBUST_EDGE


def test_stock_selection_edge_when_everything_survives() -> None:
    inputs = EdgeClassificationInputs(
        orig_q5_q1_spread=0.01, beta_adjusted_q5_q1_spread=0.008,
        topix_relative_q5_q1_spread=0.006, bear_excluded_q5_q1_spread=0.004,
        day_top20_excluded_q5_q1_spread=0.002,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.STOCK_SELECTION_EDGE
    assert result.orig_positive and result.beta_survives and result.bear_excl_survives


def test_market_timing_edge_when_beta_and_bear_both_vanish() -> None:
    inputs = EdgeClassificationInputs(
        orig_q5_q1_spread=0.01, beta_adjusted_q5_q1_spread=-0.001,
        topix_relative_q5_q1_spread=0.002, bear_excluded_q5_q1_spread=-0.002,
        day_top20_excluded_q5_q1_spread=-0.001,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.MARKET_TIMING_EDGE
    assert not result.beta_survives and not result.bear_excl_survives


def test_mixed_when_signals_disagree() -> None:
    inputs = EdgeClassificationInputs(
        orig_q5_q1_spread=0.01, beta_adjusted_q5_q1_spread=0.005,  # survives
        topix_relative_q5_q1_spread=0.003,  # survives
        bear_excluded_q5_q1_spread=-0.001,  # does NOT survive
        day_top20_excluded_q5_q1_spread=0.001,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.MIXED


def test_none_values_treated_as_not_surviving() -> None:
    inputs = EdgeClassificationInputs(
        orig_q5_q1_spread=0.01, beta_adjusted_q5_q1_spread=None,
        topix_relative_q5_q1_spread=None, bear_excluded_q5_q1_spread=None,
        day_top20_excluded_q5_q1_spread=None,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.MARKET_TIMING_EDGE
