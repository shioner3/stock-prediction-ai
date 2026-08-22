"""Phase 14 section 23: Conditional Edge Decision Framework.

Classifies long_oversold_rebound's BEAR/large-drawdown conditional edge
(Phase 13's exploratory finding) against Phase 14's independent
robustness battery. Every threshold here is FIXED before running any
Phase 14 analysis against real data (spec section 3/28: pre-registered,
never adjusted after seeing results) and chosen to mirror this project's
EXISTING decision philosophy rather than invent new ad-hoc numbers:

- MIN_SAMPLE=30 matches config/loader.py::MinSampleConfig.min_oos_trades'
  existing default, reused verbatim by backtest/decision.py and
  ensemble/decision.py's own MIN_SAMPLE - the same "at least 30
  observations" bar every prior OOS Decision gate already uses.
- SIGNIFICANCE_ALPHA=0.05 matches the conventional two-sided
  significance level used everywhere else in this project (Phase 6+),
  applied here to the FDR-ADJUSTED (not raw) permutation p-value for the
  core condition, since Phase 14 tests many conditions simultaneously
  (spec section 20's multiple-testing correction).
- "High cost tier PF > 1" matches Phase 8/9's own repeated framing
  ("特にHigh costでPF>1を維持できるかを確認する").

The seven categories are NOT a flat enum - they read as an
increasingly-qualified spectrum from "core condition shows nothing"
(REJECT/INSUFFICIENT_EVIDENCE) up through "shows something but it's not
really about the Signal" (TIMING_DEPENDENT), "only about 1-2 specific
events" (EVENT_DEPENDENT), "regime-specific but not fully robust within
the regime" (REGIME_DEPENDENT), to "regime-specific AND robust within
the regime" (ROBUST_CONDITIONAL). SCORE_INDEPENDENT is a separate,
always-attachable SECONDARY tag (Score adding no discriminating power
within the core condition is orthogonal to all of the above, not a
rejection) - spec section 24's Q9 asks this exact question
independently of the primary classification.

classify_conditional_edge() evaluates checks in DECREASING severity so
the first disqualifying finding wins as Primary, with any remaining
non-primary findings attached as Secondary (spec section 23: "Primary/
Secondary分類可" - "REGIME_DEPENDENT (Primary) + EVENT_CONCENTRATED
(Secondary)" is the spec's own worked example).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

MIN_SAMPLE = 30
SIGNIFICANCE_ALPHA = 0.05


class ConditionalEdgeDecision(str, Enum):
    ROBUST_CONDITIONAL = "ROBUST_CONDITIONAL"
    REGIME_DEPENDENT = "REGIME_DEPENDENT"
    EVENT_DEPENDENT = "EVENT_DEPENDENT"
    TIMING_DEPENDENT = "TIMING_DEPENDENT"
    SCORE_INDEPENDENT = "SCORE_INDEPENDENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ConditionalEdgeDecisionInputs:
    # --- core condition (pre-registered primary bucket, e.g. BEAR x
    # TOPIX20d<=-10%) --------------------------------------------------
    n_sample: int
    expectancy_ci_low: float | None
    profit_factor_high_cost: float | None
    permutation_p_value_fdr_adjusted: float | None

    # --- event dependency: does excluding a single named event flip the
    # core condition's expectancy non-positive? None = axis not testable
    # (e.g. too few trades remain either way) -------------------------
    positive_excluding_aug2024: bool | None
    positive_excluding_apr2025: bool | None
    # True only if EVERY individually-excluded major BEAR episode still
    # leaves a positive expectancy (Leave-One-Episode-Out, spec section 13).
    positive_excluding_each_major_episode: bool | None
    # Same, for Leave-One-Year-Out (spec section 14).
    positive_excluding_each_year: bool | None

    # --- timing dependency: among the pre-registered offset sweep
    # (excluding offset=0, the real trigger), what fraction of shifted
    # placebo dates ALSO clear the same core_positive bar? A high
    # fraction means the effect is not specific to the Signal's exact
    # trigger day - it is just "any day in this regime works", which
    # undermines the Signal's own timing value. None = not testable. ---
    timing_placebo_positive_fraction: float | None

    # --- regime specificity: does the CONTROL bucket (non-BEAR / TOPIX
    # 20d > -5%) show a comparably positive edge? If so the effect is
    # not actually regime-specific. None = control bucket not testable
    # (e.g. zero trades). ------------------------------------------------
    control_bucket_also_positive: bool | None

    # --- score independence (secondary, always attachable) - True if
    # Score Q1-Q5 shows no meaningful monotonic/rank-correlation
    # advantage within the core condition (spec section 24 Q9). None =
    # not testable (e.g. too few Score-scored trades in the core bucket).
    score_adds_no_discriminating_power: bool | None

    # Fixed threshold (not tuned to the result): if >= this fraction of
    # non-zero offsets also clear core_positive, timing is NOT
    # Signal-specific.
    timing_placebo_max_positive_fraction: float = 0.5


@dataclass(frozen=True)
class ConditionalEdgeDecisionResult:
    primary: ConditionalEdgeDecision
    secondary: list[ConditionalEdgeDecision] = field(default_factory=list)


def _core_positive(inputs: ConditionalEdgeDecisionInputs) -> bool:
    return (
        inputs.expectancy_ci_low is not None
        and inputs.expectancy_ci_low > 0
        and inputs.profit_factor_high_cost is not None
        and inputs.profit_factor_high_cost > 1.0
        and inputs.permutation_p_value_fdr_adjusted is not None
        and inputs.permutation_p_value_fdr_adjusted < SIGNIFICANCE_ALPHA
    )


def classify_conditional_edge(
    inputs: ConditionalEdgeDecisionInputs,
) -> ConditionalEdgeDecisionResult:
    if inputs.n_sample < MIN_SAMPLE:
        return ConditionalEdgeDecisionResult(ConditionalEdgeDecision.INSUFFICIENT_EVIDENCE)
    if not _core_positive(inputs):
        return ConditionalEdgeDecisionResult(ConditionalEdgeDecision.REJECT)

    secondary: list[ConditionalEdgeDecision] = []
    if inputs.score_adds_no_discriminating_power is True:
        secondary.append(ConditionalEdgeDecision.SCORE_INDEPENDENT)

    event_dependent = (
        inputs.positive_excluding_aug2024 is False
        or inputs.positive_excluding_apr2025 is False
        or inputs.positive_excluding_each_major_episode is False
        or inputs.positive_excluding_each_year is False
    )
    if event_dependent:
        return ConditionalEdgeDecisionResult(ConditionalEdgeDecision.EVENT_DEPENDENT, secondary)

    if (
        inputs.timing_placebo_positive_fraction is not None
        and inputs.timing_placebo_positive_fraction >= inputs.timing_placebo_max_positive_fraction
    ):
        return ConditionalEdgeDecisionResult(ConditionalEdgeDecision.TIMING_DEPENDENT, secondary)

    # REGIME_DEPENDENT is the fallback below, not a separate branch here:
    # a control bucket that is ALSO positive fails the fully_robust gate
    # (via the "control_bucket_also_positive is False" clause) and falls
    # through to REGIME_DEPENDENT on its own - encoding it as a second,
    # independent check here would just duplicate that same outcome.
    fully_robust = (
        inputs.positive_excluding_aug2024 is True
        and inputs.positive_excluding_apr2025 is True
        and inputs.positive_excluding_each_major_episode is True
        and inputs.positive_excluding_each_year is True
        and inputs.control_bucket_also_positive is False
    )
    if fully_robust:
        return ConditionalEdgeDecisionResult(ConditionalEdgeDecision.ROBUST_CONDITIONAL, secondary)

    return ConditionalEdgeDecisionResult(ConditionalEdgeDecision.REGIME_DEPENDENT, secondary)
