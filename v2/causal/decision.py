"""Phase V2-3 Decision Framework (spec section 31): classifies the
ROBUSTNESS of the Q1 negative-predictive-signal phenomenon (Phase V2-2's
finding that the LOW-score bucket relatively outperforms the HIGH-score
bucket, i.e. Q5-Q1 spread < 0) - NEVER a Risk Filter adoption decision
(spec sections 32/41 explicitly forbid that here; classify only).

Sign convention note: "Q1 negative [predictive] signal" describes the
Score-Return relationship being INVERTED (negative), not that Q1's own
raw Forward Return is negative (Phase V2-2 found Q1's mean 5d return was
+0.388%, i.e. POSITIVE in absolute terms - see research/
phase_v2_2_report.md section 7). This module therefore operationalizes
the phenomenon via the (Q5-Q1) spread's SIGN (negative = phenomenon
present, matching V2-2's own established convention) plus Q1's absolute
deviation from the population mean (permutation test) as corroborating
evidence - documented here explicitly since spec section 20's literal
phrase "Q1のマイナスReturn" is otherwise ambiguous against V2-2's own
numbers.

Five tiers, evaluated top-down so the FIRST criterion that fires wins
(mirrors v2/validation/decision.py's own layered, pre-registered style):

    REJECT > EVENT_DEPENDENT_NEGATIVE > WEAK_EVIDENCE >
    CONDITIONAL_NEGATIVE > STRUCTURALLY_ROBUST_NEGATIVE

Every threshold is fixed BEFORE the real run and must never be adjusted
after seeing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SIGNIFICANCE_ALPHA = 0.05
MIN_HOLDING_PERIOD_REPRODUCIBILITY = 4 / 7  # >= 4 of 7 windows show a negative spread
MIN_YEAR_REPRODUCIBILITY = 3 / 5  # >= 3 of 5 years show a negative spread
MIN_REGIME_REPRODUCIBILITY = 2 / 3  # >= 2 of 3 regimes show a negative spread


class V2_3Decision(str, Enum):
    STRUCTURALLY_ROBUST_NEGATIVE = "STRUCTURALLY_ROBUST_NEGATIVE"
    CONDITIONAL_NEGATIVE = "CONDITIONAL_NEGATIVE"
    EVENT_DEPENDENT_NEGATIVE = "EVENT_DEPENDENT_NEGATIVE"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class V2_3DecisionInputs:
    primary_q5_q1_spread: float | None  # negative confirms the phenomenon
    day_cluster_spread_ci_high: float | None  # spread's own CI upper bound
    block_spread_ci_high: float | None
    q1_permutation_p_value: float | None  # Q1 vs population, primary window
    fdr_significant: bool | None
    holding_period_negative_fraction: float | None
    year_negative_fraction: float | None
    regime_negative_fraction: float | None
    # False if excluding 2024-08 OR the single max-contribution day flips
    # the spread >= 0 (or drives |spread| to near-zero); None if untestable.
    survives_event_exclusion: bool | None


@dataclass(frozen=True)
class V2_3DecisionResult:
    decision: V2_3Decision
    reasons: list[str] = field(default_factory=list)


def _core_negative_and_significant(inputs: V2_3DecisionInputs) -> bool:
    return (
        inputs.day_cluster_spread_ci_high is not None
        and inputs.day_cluster_spread_ci_high < 0
        and inputs.block_spread_ci_high is not None
        and inputs.block_spread_ci_high < 0
        and inputs.q1_permutation_p_value is not None
        and inputs.q1_permutation_p_value < SIGNIFICANCE_ALPHA
        and inputs.fdr_significant is True
    )


def classify_v2_3_decision(inputs: V2_3DecisionInputs) -> V2_3DecisionResult:
    # --- REJECT: no negative relationship to explain in the first place --
    if inputs.primary_q5_q1_spread is None or inputs.primary_q5_q1_spread >= 0:
        return V2_3DecisionResult(
            V2_3Decision.REJECT, ["Q5-Q1 spread >= 0 or unavailable at the primary window"]
        )

    # --- EVENT_DEPENDENT_NEGATIVE: the phenomenon is mostly one event ----
    if inputs.survives_event_exclusion is False:
        return V2_3DecisionResult(
            V2_3Decision.EVENT_DEPENDENT_NEGATIVE,
            ["Spread sign/magnitude does not survive excluding the 2024-08 event or the single "
             "max-contribution day"],
        )

    # --- Core gate: Day-Cluster + Block Bootstrap CI both exclude zero, --
    # --- permutation significant, FDR-significant ------------------------
    if not _core_negative_and_significant(inputs):
        return V2_3DecisionResult(
            V2_3Decision.WEAK_EVIDENCE,
            ["Negative spread present but fails the core Day-Cluster/Block Bootstrap + "
             "Permutation + FDR significance gate"],
        )

    # --- Reproducibility across Holding Period / Year / Regime -----------
    axes_ok = (
        inputs.holding_period_negative_fraction is not None
        and inputs.holding_period_negative_fraction >= MIN_HOLDING_PERIOD_REPRODUCIBILITY,
        inputs.year_negative_fraction is not None
        and inputs.year_negative_fraction >= MIN_YEAR_REPRODUCIBILITY,
        inputs.regime_negative_fraction is not None
        and inputs.regime_negative_fraction >= MIN_REGIME_REPRODUCIBILITY,
    )
    if all(axes_ok):
        return V2_3DecisionResult(
            V2_3Decision.STRUCTURALLY_ROBUST_NEGATIVE,
            ["Negative relationship reproduces across Holding Period/Year/Regime majorities, "
             "with significant Day-Cluster/Block Bootstrap + Permutation + FDR"],
        )

    return V2_3DecisionResult(
        V2_3Decision.CONDITIONAL_NEGATIVE,
        ["Core significance gate passed but reproducibility across Holding Period/Year/Regime "
         "is only partial"],
    )
