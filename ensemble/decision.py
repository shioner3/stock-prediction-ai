"""Phase 12 section 31/40: Ensemble Decision Framework.

Every threshold here is FIXED before running any Phase 12 analysis
against real data (spec section 31: "Decision criteriaはPhase開始前に
固定する... Phase 12の結果を見て基準を変更してはならない") and chosen to
mirror this project's EXISTING decision philosophy rather than invent
new ad-hoc numbers:

- min_sample=30 matches config/loader.py::MinSampleConfig.min_oos_trades'
  existing default (the same "at least 30 observations" bar every prior
  OOS Decision gate already uses) - see ensemble/combinations.py's
  MIN_COMBINATION_SAMPLE, which uses the same constant.
- permutation p < 0.05 matches the conventional two-sided significance
  level used everywhere else in this project (Phase 6+).
- "High cost tier PF > 1" matches Phase 8/9's own repeated framing
  ("特にHigh costでPF>1を維持できるかを確認する").
- min_frequency_pct=0.05 (occurs on >=5% of trading days, roughly at
  least one occurrence every ~4 weeks in a ~250-trading-day year) is a
  NEW threshold specific to Phase 12 (no prior Phase needed a frequency
  gate, since a single Signal firing rarely enough to fail this bar
  would already fail min_sample first at Full Universe scale) - chosen
  as a round, conservative "usable for a 5-position swing tool" bar,
  fixed here before any real run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MIN_SAMPLE = 30
MIN_FREQUENCY_PCT = 0.05
SIGNIFICANCE_ALPHA = 0.05


class EnsembleDecision(str, Enum):
    ROBUST_ENSEMBLE = "ROBUST_ENSEMBLE"
    REGIME_DEPENDENT_ENSEMBLE = "REGIME_DEPENDENT_ENSEMBLE"
    EVENT_DEPENDENT_ENSEMBLE = "EVENT_DEPENDENT_ENSEMBLE"
    FREQUENCY_TOO_LOW = "FREQUENCY_TOO_LOW"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class EnsembleDecisionInputs:
    n_sample: int
    pct_trading_days_with_occurrence: float
    expectancy_ci_low: float | None
    profit_factor_high_cost: float | None
    permutation_p_value: float | None
    # None = "not enough data to evaluate this axis at all" (kept
    # separate from "evaluated and found negative", per section 17/16 -
    # an axis Phase 12 could not test must never silently count as
    # "passed").
    positive_excluding_aug2024: bool | None
    positive_in_bull: bool | None
    positive_in_neutral: bool | None
    positive_in_bear: bool | None


def classify_ensemble(inputs: EnsembleDecisionInputs) -> EnsembleDecision:
    if inputs.n_sample < MIN_SAMPLE:
        return EnsembleDecision.INSUFFICIENT_EVIDENCE
    if inputs.pct_trading_days_with_occurrence < MIN_FREQUENCY_PCT:
        return EnsembleDecision.FREQUENCY_TOO_LOW

    core_positive = (
        inputs.expectancy_ci_low is not None
        and inputs.expectancy_ci_low > 0
        and inputs.profit_factor_high_cost is not None
        and inputs.profit_factor_high_cost > 1.0
        and inputs.permutation_p_value is not None
        and inputs.permutation_p_value < SIGNIFICANCE_ALPHA
    )
    if not core_positive:
        return EnsembleDecision.REJECT

    if inputs.positive_excluding_aug2024 is False:
        return EnsembleDecision.EVENT_DEPENDENT_ENSEMBLE

    regime_flags = [
        inputs.positive_in_bull, inputs.positive_in_neutral, inputs.positive_in_bear,
    ]
    known_regime_flags = [f for f in regime_flags if f is not None]
    if known_regime_flags and not all(known_regime_flags):
        return EnsembleDecision.REGIME_DEPENDENT_ENSEMBLE

    return EnsembleDecision.ROBUST_ENSEMBLE
