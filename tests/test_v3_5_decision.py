"""v3/residual/decision_v3_5.py - the 4-way edge classification tree,
exercised through all 4 branches.
"""

from __future__ import annotations

from v3.residual.decision_v3_5 import (
    EdgeClassification,
    EdgeClassificationInputs,
    classify_edge_source,
)

_FULLY_ROBUST_C = {
    "beta_residual_q5_q1": 0.01,
    "beta_residual_bear_excluded_q5_q1": 0.005,
    "beta_residual_top5_expectancy": 0.02,
    "beta_residual_top10_expectancy": 0.015,
    "beta_residual_top20_expectancy": 0.01,
    "beta_residual_day_cluster_ci_low": 0.001,
    "beta_residual_block_ci_low": 0.001,
    "beta_residual_permutation_p": 0.01,
    "beta_residual_fdr_significant": True,
}


def test_stock_selection_edge_when_everything_robust() -> None:
    inputs = EdgeClassificationInputs(
        raw_q5_q1=0.02, topix_relative_q5_q1=0.008, **_FULLY_ROBUST_C,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.STOCK_SELECTION_EDGE


def test_market_timing_edge_when_both_residual_targets_negative_but_raw_positive() -> None:
    inputs = EdgeClassificationInputs(
        raw_q5_q1=0.02, topix_relative_q5_q1=-0.01,
        beta_residual_q5_q1=-0.008, beta_residual_bear_excluded_q5_q1=-0.005,
        beta_residual_top5_expectancy=-0.01, beta_residual_top10_expectancy=-0.005,
        beta_residual_top20_expectancy=-0.003, beta_residual_day_cluster_ci_low=-0.001,
        beta_residual_block_ci_low=-0.002, beta_residual_permutation_p=0.9,
        beta_residual_fdr_significant=False,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.MARKET_TIMING_EDGE


def test_no_robust_edge_when_nothing_positive() -> None:
    inputs = EdgeClassificationInputs(
        raw_q5_q1=-0.001, topix_relative_q5_q1=-0.01,
        beta_residual_q5_q1=-0.008, beta_residual_bear_excluded_q5_q1=-0.005,
        beta_residual_top5_expectancy=-0.01, beta_residual_top10_expectancy=-0.005,
        beta_residual_top20_expectancy=-0.003, beta_residual_day_cluster_ci_low=-0.001,
        beta_residual_block_ci_low=-0.002, beta_residual_permutation_p=0.9,
        beta_residual_fdr_significant=False,
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.NO_ROBUST_EDGE


def test_mixed_edge_when_positive_but_not_fully_robust() -> None:
    # Target C's spread is positive but fails the FDR/permutation gate.
    inputs = EdgeClassificationInputs(
        raw_q5_q1=0.02, topix_relative_q5_q1=0.005,
        beta_residual_q5_q1=0.01, beta_residual_bear_excluded_q5_q1=0.005,
        beta_residual_top5_expectancy=0.02, beta_residual_top10_expectancy=0.015,
        beta_residual_top20_expectancy=0.01, beta_residual_day_cluster_ci_low=0.001,
        beta_residual_block_ci_low=0.001, beta_residual_permutation_p=0.5,  # fails
        beta_residual_fdr_significant=False,  # fails
    )
    result = classify_edge_source(inputs)
    assert result.classification == EdgeClassification.MIXED_EDGE


def test_mixed_edge_when_only_one_of_a_b_positive() -> None:
    inputs = EdgeClassificationInputs(
        raw_q5_q1=0.02, topix_relative_q5_q1=0.005, **_FULLY_ROBUST_C,
    )
    # topix_relative positive, beta_residual fully robust -> STOCK_SELECTION_EDGE branch;
    # flip topix_relative to negative to force MIXED via "only B positive".
    inputs_mixed = EdgeClassificationInputs(
        raw_q5_q1=0.02, topix_relative_q5_q1=-0.001, **_FULLY_ROBUST_C,
    )
    result_stock = classify_edge_source(inputs)
    result_mixed = classify_edge_source(inputs_mixed)
    assert result_stock.classification == EdgeClassification.STOCK_SELECTION_EDGE
    assert result_mixed.classification == EdgeClassification.MIXED_EDGE
