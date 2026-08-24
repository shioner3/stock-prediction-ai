"""Spec sections 18/31: mechanical 4-way Edge Classification. Directly
implements spec section 18's 8 named criteria (A-H, all evaluated
against Target C = Beta-adjusted Residual, the most market-neutral of
the 3 new Target definitions) plus Target B (TOPIX-relative)'s own
Q5-Q1 sign as a second, independent "does a residual-trained ranking
show ANY edge" check - never redefined after seeing results (spec's own
explicit prohibition, section 18: "ただし、この条件を後から変更しては
いけない").

  STOCK_SELECTION_EDGE: Target B positive AND Target C passes ALL of
    criteria A-H (positive spread, survives BEAR exclusion, positive
    Top-N expectancy, both Bootstrap CIs > 0, Permutation significant,
    FDR-significant).
  MARKET_TIMING_EDGE: neither Target B nor Target C shows a positive
    spread, but the ORIGINAL Raw Target (A) does - the edge exists in
    the raw ranking but does not survive retraining on either residual
    Target.
  NO_ROBUST_EDGE: neither Target B nor Target C is positive, AND the Raw
    Target itself is not positive either - no edge anywhere.
  MIXED_EDGE: everything else (e.g. one of B/C is positive but not both,
    or Target C is positive but fails part of the A-H robustness gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EdgeClassification(str, Enum):
    STOCK_SELECTION_EDGE = "STOCK_SELECTION_EDGE"
    MARKET_TIMING_EDGE = "MARKET_TIMING_EDGE"
    MIXED_EDGE = "MIXED_EDGE"
    NO_ROBUST_EDGE = "NO_ROBUST_EDGE"


@dataclass(frozen=True)
class EdgeClassificationInputs:
    raw_q5_q1: float | None  # Target A
    topix_relative_q5_q1: float | None  # Target B - criterion A
    beta_residual_q5_q1: float | None  # Target C - criterion B
    beta_residual_bear_excluded_q5_q1: float | None  # criterion C
    beta_residual_top5_expectancy: float | None  # criterion D
    beta_residual_top10_expectancy: float | None
    beta_residual_top20_expectancy: float | None
    beta_residual_day_cluster_ci_low: float | None  # criterion E
    beta_residual_block_ci_low: float | None  # criterion F
    beta_residual_permutation_p: float | None  # criterion G
    beta_residual_fdr_significant: bool | None  # criterion H
    significance_alpha: float = 0.05


@dataclass(frozen=True)
class EdgeClassificationResult:
    classification: EdgeClassification
    criterion_a_topix_relative_positive: bool
    criterion_b_beta_residual_positive: bool
    criterion_c_survives_bear_exclusion: bool
    criterion_d_topn_positive_expectancy: bool
    criterion_e_day_cluster_ci_positive: bool
    criterion_f_block_ci_positive: bool
    criterion_g_permutation_significant: bool
    criterion_h_fdr_significant: bool
    raw_positive: bool
    reasons: list[str]


def _positive(x: float | None) -> bool:
    return x is not None and x > 0


def classify_edge_source(inputs: EdgeClassificationInputs) -> EdgeClassificationResult:
    criterion_a = _positive(inputs.topix_relative_q5_q1)
    criterion_b = _positive(inputs.beta_residual_q5_q1)
    criterion_c = _positive(inputs.beta_residual_bear_excluded_q5_q1)
    criterion_d = (
        _positive(inputs.beta_residual_top5_expectancy)
        and _positive(inputs.beta_residual_top10_expectancy)
        and _positive(inputs.beta_residual_top20_expectancy)
    )
    criterion_e = _positive(inputs.beta_residual_day_cluster_ci_low)
    criterion_f = _positive(inputs.beta_residual_block_ci_low)
    criterion_g = (
        inputs.beta_residual_permutation_p is not None
        and inputs.beta_residual_permutation_p < inputs.significance_alpha
    )
    criterion_h = bool(inputs.beta_residual_fdr_significant)
    raw_positive = _positive(inputs.raw_q5_q1)

    target_c_fully_robust = (
        criterion_b and criterion_c and criterion_d and criterion_e and criterion_f
        and criterion_g and criterion_h
    )

    reasons: list[str] = []
    if criterion_a and target_c_fully_robust:
        classification = EdgeClassification.STOCK_SELECTION_EDGE
        reasons.append(
            "TOPIX-relative Target positive AND Beta-adjusted Residual Target passes all of "
            "criteria A-H"
        )
    elif not criterion_a and not criterion_b:
        if raw_positive:
            classification = EdgeClassification.MARKET_TIMING_EDGE
            reasons.append(
                "Raw Target positive but neither TOPIX-relative nor Beta-adjusted Residual "
                "Target shows a positive spread"
            )
        else:
            classification = EdgeClassification.NO_ROBUST_EDGE
            reasons.append(
                "Neither Raw, TOPIX-relative, nor Beta-adjusted Residual Target is positive"
            )
    else:
        classification = EdgeClassification.MIXED_EDGE
        reasons.append(
            f"partial evidence: criterion_a={criterion_a} criterion_b={criterion_b} "
            f"target_c_fully_robust={target_c_fully_robust}"
        )

    return EdgeClassificationResult(
        classification=classification,
        criterion_a_topix_relative_positive=criterion_a,
        criterion_b_beta_residual_positive=criterion_b,
        criterion_c_survives_bear_exclusion=criterion_c,
        criterion_d_topn_positive_expectancy=criterion_d,
        criterion_e_day_cluster_ci_positive=criterion_e,
        criterion_f_block_ci_positive=criterion_f,
        criterion_g_permutation_significant=criterion_g,
        criterion_h_fdr_significant=criterion_h,
        raw_positive=raw_positive,
        reasons=reasons,
    )
