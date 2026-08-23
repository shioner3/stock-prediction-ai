"""Spec section 18/19: reapplies V3-3's Decision Framework UNCHANGED (no
new pass/fail thresholds this Phase - spec section 18's explicit
instruction), then adds a SEPARATE, additional 4-way classification of
WHERE the edge comes from - spec section 19's "most important judgment".

Classification is a pre-specified, mechanical decision tree over 5
already-computed booleans (never chosen after looking at which rule gives
the most favorable label):

  orig_positive        - V3-3's own primary Q5-Q1 spread > 0
  beta_survives        - `market_decomposition.py`'s Beta-adjusted-return
                          variant (section 3.C) spread > 0, holding the
                          ORIGINAL ranking fixed
  topix_rel_survives   - the TOPIX-relative-return variant (section 3.B)
                          spread > 0
  bear_excl_survives   - `leave_one_out.py`'s "exclude BEAR regime"
                          spread > 0 (section 6/9)
  day_top20_survives   - `leave_one_out.py`'s "exclude the top 20
                          contributing days" spread > 0 (section 7)

  NO_ROBUST_EDGE      : not orig_positive
  STOCK_SELECTION_EDGE: orig_positive AND beta_survives AND
                        topix_rel_survives AND bear_excl_survives
  MARKET_TIMING_EDGE  : orig_positive AND NOT beta_survives AND
                        NOT bear_excl_survives
  MIXED               : orig_positive, everything else (the signals
                        disagree - some but not all market-component
                        removals survive)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EdgeClassification(str, Enum):
    STOCK_SELECTION_EDGE = "STOCK_SELECTION_EDGE"
    MARKET_TIMING_EDGE = "MARKET_TIMING_EDGE"
    MIXED = "MIXED"
    NO_ROBUST_EDGE = "NO_ROBUST_EDGE"


@dataclass(frozen=True)
class EdgeClassificationInputs:
    orig_q5_q1_spread: float | None
    beta_adjusted_q5_q1_spread: float | None
    topix_relative_q5_q1_spread: float | None
    bear_excluded_q5_q1_spread: float | None
    day_top20_excluded_q5_q1_spread: float | None


@dataclass(frozen=True)
class EdgeClassificationResult:
    classification: EdgeClassification
    orig_positive: bool
    beta_survives: bool
    topix_rel_survives: bool
    bear_excl_survives: bool
    day_top20_survives: bool
    reasons: list[str]


def _positive(x: float | None) -> bool:
    return x is not None and x > 0


def classify_edge_source(inputs: EdgeClassificationInputs) -> EdgeClassificationResult:
    orig_positive = _positive(inputs.orig_q5_q1_spread)
    beta_survives = _positive(inputs.beta_adjusted_q5_q1_spread)
    topix_rel_survives = _positive(inputs.topix_relative_q5_q1_spread)
    bear_excl_survives = _positive(inputs.bear_excluded_q5_q1_spread)
    day_top20_survives = _positive(inputs.day_top20_excluded_q5_q1_spread)

    reasons: list[str] = []
    if not orig_positive:
        classification = EdgeClassification.NO_ROBUST_EDGE
        reasons.append(f"original Q5-Q1 spread is not positive ({inputs.orig_q5_q1_spread})")
    elif beta_survives and topix_rel_survives and bear_excl_survives:
        classification = EdgeClassification.STOCK_SELECTION_EDGE
        reasons.append("spread survives Beta-adjustment, TOPIX-relative return, and BEAR exclusion")
    elif not beta_survives and not bear_excl_survives:
        classification = EdgeClassification.MARKET_TIMING_EDGE
        reasons.append("spread vanishes under BOTH Beta-adjustment and BEAR-regime exclusion")
    else:
        classification = EdgeClassification.MIXED
        reasons.append(
            f"signals disagree: beta_survives={beta_survives} "
            f"topix_rel_survives={topix_rel_survives} "
            f"bear_excl_survives={bear_excl_survives} day_top20_survives={day_top20_survives}"
        )

    return EdgeClassificationResult(
        classification=classification, orig_positive=orig_positive, beta_survives=beta_survives,
        topix_rel_survives=topix_rel_survives, bear_excl_survives=bear_excl_survives,
        day_top20_survives=day_top20_survives, reasons=reasons,
    )
