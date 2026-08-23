"""Phase V2-2 Decision Framework (spec section 24): classifies whether
V2 Score carries reproducible predictive information, mechanically -
every threshold below is fixed BEFORE running the real analysis and
must never be adjusted after seeing results (spec section 24's own
"Decisionは機械的に行い、結果を見てthresholdを変更しない").

Four tiers, evaluated top-down so the FIRST criterion that fires wins
(mirrors the layered-condition style backtest/decision.py and
ensemble/decision.py already use in this project):

    ROBUST_CANDIDATE > CONDITIONAL_CANDIDATE > WEAK_EVIDENCE > REJECT

ROBUST_CANDIDATE requires ALL of: Q5-Q1 spread > 0, Block Bootstrap
spread CI lower bound > 0, permutation p sufficiently low (< alpha),
significant after FDR correction, direction reproduces across at least
half the tested Holding Periods, reproduces across at least half the
tested years, no extreme single-event dependency (excluding either
2024-08 or the single max-contribution day must not flip the spread
non-positive), reproduces in the Top-N candidate test, and survives at
least the "low" cost tier (spec's own list, section 24).

CONDITIONAL_CANDIDATE: the core spread is real (positive, FDR-significant)
but fails ROBUST on a SCOPE-LIMITING criterion only (regime-specific,
holding-period-specific, or market-segment-specific) - not on the
underlying event/reproducibility criteria that would instead route to
WEAK_EVIDENCE.

REJECT: any of the OUTRIGHT-NEGATIVE conditions spec section 24 lists
(spread <=0 persistently, bootstrap CI crosses zero, no permutation
significance, does not survive year-to-year, disappears after cost,
extreme event concentration, no real Score-Return relationship).

WEAK_EVIDENCE: everything else - some positive signal, but reproducibility
or statistical significance is insufficient to call it either REJECT or
CONDITIONAL_CANDIDATE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SIGNIFICANCE_ALPHA = 0.05
MIN_HOLDING_PERIOD_REPRODUCIBILITY = 0.5  # >= half of tested windows must agree in direction
MIN_YEAR_REPRODUCIBILITY = 0.5  # >= half of tested years must agree in direction


class V2Decision(str, Enum):
    ROBUST_CANDIDATE = "ROBUST_CANDIDATE"
    CONDITIONAL_CANDIDATE = "CONDITIONAL_CANDIDATE"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class V2DecisionInputs:
    q5_q1_spread: float | None
    block_bootstrap_ci_low: float | None
    permutation_p_value: float | None
    fdr_significant: bool | None
    # Fraction of tested Holding Periods whose OWN Q5-Q1 spread is > 0.
    holding_period_positive_fraction: float | None
    # Fraction of tested years whose OWN Q5-Q1 spread is > 0.
    year_positive_fraction: float | None
    # False if excluding 2024-08 OR the single max-contribution day flips
    # the spread <= 0; None if untestable.
    survives_event_exclusion: bool | None
    # True if the Top-N (spec's 5/10/20) Long Candidate Test's mean
    # return is positive and directionally consistent with the spread.
    topn_reproduces: bool | None
    # True if the spread stays positive at least through the "low"
    # (10bps) cost tier.
    survives_low_cost: bool | None
    # Regime/holding-period/segment SCOPE limitation flags - used only to
    # distinguish CONDITIONAL_CANDIDATE from ROBUST_CANDIDATE once the
    # core positive+significant gate has already passed.
    regime_dependent: bool | None
    holding_period_dependent: bool | None
    segment_dependent: bool | None


@dataclass(frozen=True)
class V2DecisionResult:
    decision: V2Decision
    reasons: list[str]


def _core_positive_and_significant(inputs: V2DecisionInputs) -> bool:
    return (
        inputs.q5_q1_spread is not None
        and inputs.q5_q1_spread > 0
        and inputs.block_bootstrap_ci_low is not None
        and inputs.block_bootstrap_ci_low > 0
        and inputs.permutation_p_value is not None
        and inputs.permutation_p_value < SIGNIFICANCE_ALPHA
        and inputs.fdr_significant is True
    )


def classify_v2_decision(inputs: V2DecisionInputs) -> V2DecisionResult:
    reasons: list[str] = []

    # --- REJECT: any outright-negative condition -----------------------
    if inputs.q5_q1_spread is None or inputs.q5_q1_spread <= 0:
        return V2DecisionResult(V2Decision.REJECT, ["Q5-Q1 spread <= 0 or unavailable"])
    if inputs.block_bootstrap_ci_low is not None and inputs.block_bootstrap_ci_low <= 0:
        reasons.append("Block Bootstrap CI crosses zero")
    if inputs.permutation_p_value is not None and inputs.permutation_p_value >= SIGNIFICANCE_ALPHA:
        reasons.append("Permutation test not significant")
    if inputs.fdr_significant is False:
        reasons.append("Not significant after FDR correction")
    if (
        inputs.year_positive_fraction is not None
        and inputs.year_positive_fraction < MIN_YEAR_REPRODUCIBILITY
    ):
        reasons.append("Does not reproduce year-to-year")
    if inputs.survives_low_cost is False:
        reasons.append("Disappears after transaction cost")
    if inputs.survives_event_exclusion is False:
        reasons.append("Extreme single-event/single-day concentration")
    if reasons:
        return V2DecisionResult(V2Decision.REJECT, reasons)

    # --- Core gate: positive, bootstrap-robust, permutation+FDR significant
    if not _core_positive_and_significant(inputs):
        return V2DecisionResult(
            V2Decision.WEAK_EVIDENCE,
            ["Positive spread but fails the core bootstrap/permutation/FDR significance gate"],
        )

    # --- Reproducibility across Holding Period / Top-N ------------------
    holding_ok = (
        inputs.holding_period_positive_fraction is not None
        and inputs.holding_period_positive_fraction >= MIN_HOLDING_PERIOD_REPRODUCIBILITY
    )
    if not holding_ok or inputs.topn_reproduces is not True:
        return V2DecisionResult(
            V2Decision.WEAK_EVIDENCE,
            ["Core gate passed but insufficient reproducibility across Holding Period/Top-N"],
        )

    # --- Scope-limiting conditions: CONDITIONAL_CANDIDATE vs ROBUST -----
    scope_flags = (
        inputs.regime_dependent, inputs.holding_period_dependent, inputs.segment_dependent,
    )
    scope_limited = any(flag is True for flag in scope_flags)
    if scope_limited:
        limits = [
            name
            for name, flag in (
                ("regime", inputs.regime_dependent),
                ("holding_period", inputs.holding_period_dependent),
                ("segment", inputs.segment_dependent),
            )
            if flag is True
        ]
        return V2DecisionResult(
            V2Decision.CONDITIONAL_CANDIDATE, [f"Scope-limited: {', '.join(limits)}-dependent"]
        )

    return V2DecisionResult(V2Decision.ROBUST_CANDIDATE, ["All robustness criteria satisfied"])
