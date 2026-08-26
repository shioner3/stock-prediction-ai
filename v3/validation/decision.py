"""Phase V3-3 Decision Framework (spec section 26/27): a mechanical,
pre-registered classification of the Primary combination (Model A,
target_raw_5d) - every threshold fixed BEFORE the real run
(`v3/validation/wfo_config.py`), never adjusted after seeing results.

Four tiers, evaluated top-down so the FIRST criterion that fires wins
(mirrors `v2/validation/decision.py`'s / `v2/causal/decision.py`'s own
layered style):

    ACCEPT_CANDIDATE > WEAK_EVIDENCE > REJECT > INSUFFICIENT_EVIDENCE

Separately (spec section 27), four INDEPENDENT questions are answered,
never collapsed into one another:
    A. Does the model have statistically real predictive power (Rank IC)?
    B. Is it valid AS A RANKING (Q5-Q1 spread, robust)?
    C. Would Top-N selection have been viable (PF/Expectancy, beats
       Random/Momentum baselines)?
    D. Does it clearly exceed V1/V2 (the V2 Score benchmark)?
A "YES/YES/NO/..." combination is a valid, informative outcome - spec
section 27's own example ("予測力はあるが実際の銘柄選択には不十分").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from v3.validation.wfo_config import MIN_WINDOW_DIRECTION_AGREEMENT, SIGNIFICANCE_ALPHA


class V3_3Decision(str, Enum):
    ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class V3_3DecisionInputs:
    n_windows: int
    primary_q5_q1_spread: float | None
    rank_ic_mean: float | None
    window_direction_agreement: float | None  # fraction of windows with spread > 0
    day_cluster_ci_low: float | None
    block_ci_low: float | None
    q5_permutation_p: float | None
    fdr_significant: bool | None
    survives_event_exclusion: bool | None  # sign unchanged after excluding 2024-08
    top5_mean_return: float | None
    top10_mean_return: float | None
    top20_mean_return: float | None
    random_baseline_spread: float | None
    momentum_baseline_spread: float | None
    v2_score_baseline_spread: float | None


@dataclass(frozen=True)
class V3_3DecisionResult:
    decision: V3_3Decision
    reasons: list[str] = field(default_factory=list)
    question_a_predictive_power: bool | None = None
    question_b_ranking_valid: bool | None = None
    question_c_topn_viable: bool | None = None
    question_d_beats_v1_v2: bool | None = None


MIN_WINDOWS_REQUIRED = 3


def classify_v3_3_decision(inputs: V3_3DecisionInputs) -> V3_3DecisionResult:
    if inputs.n_windows < MIN_WINDOWS_REQUIRED:
        return V3_3DecisionResult(
            V3_3Decision.INSUFFICIENT_EVIDENCE,
            [f"only {inputs.n_windows} OOS windows available (< {MIN_WINDOWS_REQUIRED})"],
        )

    question_a = inputs.rank_ic_mean is not None and inputs.rank_ic_mean > 0
    question_b = (
        inputs.primary_q5_q1_spread is not None and inputs.primary_q5_q1_spread > 0
        and inputs.day_cluster_ci_low is not None and inputs.day_cluster_ci_low > 0
        and inputs.block_ci_low is not None and inputs.block_ci_low > 0
    )
    topn_positive = all(
        v is not None and v > 0
        for v in (inputs.top5_mean_return, inputs.top10_mean_return, inputs.top20_mean_return)
    )
    beats_random = (
        inputs.primary_q5_q1_spread is not None and inputs.random_baseline_spread is not None
        and inputs.primary_q5_q1_spread > inputs.random_baseline_spread
    )
    question_c = topn_positive and beats_random
    beats_momentum = (
        inputs.primary_q5_q1_spread is not None and inputs.momentum_baseline_spread is not None
        and inputs.primary_q5_q1_spread > inputs.momentum_baseline_spread
    )
    beats_v2 = (
        inputs.primary_q5_q1_spread is not None and inputs.v2_score_baseline_spread is not None
        and inputs.primary_q5_q1_spread > inputs.v2_score_baseline_spread
    )
    question_d = beats_momentum and beats_v2

    # --- REJECT: any outright-negative condition -------------------------
    reasons: list[str] = []
    if inputs.primary_q5_q1_spread is None or inputs.primary_q5_q1_spread <= 0:
        return V3_3DecisionResult(
            V3_3Decision.REJECT, ["Primary 5d Q5-Q1 spread <= 0 or unavailable"],
            question_a, question_b, question_c, question_d,
        )
    if not beats_random:
        reasons.append("Does not beat Random Baseline")
    if not (beats_momentum and beats_v2):
        reasons.append("Does not clearly beat Simple Baselines (Momentum/V2 Score)")
    if inputs.survives_event_exclusion is False:
        reasons.append("Does not survive 2024-08 event exclusion")
    if (
        inputs.window_direction_agreement is not None
        and inputs.window_direction_agreement < MIN_WINDOW_DIRECTION_AGREEMENT
    ):
        reasons.append("Direction does not reproduce across the majority of OOS windows")
    if reasons:
        return V3_3DecisionResult(
            V3_3Decision.REJECT, reasons, question_a, question_b, question_c, question_d
        )

    # --- Core gate: bootstrap-robust, permutation+FDR significant --------
    core_significant = (
        inputs.day_cluster_ci_low is not None and inputs.day_cluster_ci_low > 0
        and inputs.block_ci_low is not None and inputs.block_ci_low > 0
        and inputs.q5_permutation_p is not None and inputs.q5_permutation_p < SIGNIFICANCE_ALPHA
        and inputs.fdr_significant is True
    )
    if not core_significant or not topn_positive:
        return V3_3DecisionResult(
            V3_3Decision.WEAK_EVIDENCE,
            ["Positive spread and beats baselines, but fails the core Day-Cluster/Block "
             "Bootstrap + Permutation + FDR significance gate, or Top-N is not uniformly positive"],
            question_a, question_b, question_c, question_d,
        )

    return V3_3DecisionResult(
        V3_3Decision.ACCEPT_CANDIDATE,
        ["All pre-registered robustness, reproducibility, and baseline-comparison criteria "
         "satisfied"],
        question_a, question_b, question_c, question_d,
    )
