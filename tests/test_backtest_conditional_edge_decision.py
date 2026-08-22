from __future__ import annotations

from backtest.conditional_edge_decision import (
    ConditionalEdgeDecision,
    ConditionalEdgeDecisionInputs,
    classify_conditional_edge,
)


def _base_inputs(**overrides: object) -> ConditionalEdgeDecisionInputs:
    defaults: dict[str, object] = dict(
        n_sample=50,
        expectancy_ci_low=0.01,
        profit_factor_high_cost=1.5,
        permutation_p_value_fdr_adjusted=0.01,
        positive_excluding_aug2024=True,
        positive_excluding_apr2025=True,
        positive_excluding_each_major_episode=True,
        positive_excluding_each_year=True,
        timing_placebo_positive_fraction=0.1,
        control_bucket_also_positive=False,
        score_adds_no_discriminating_power=None,
    )
    defaults.update(overrides)
    return ConditionalEdgeDecisionInputs(**defaults)


def test_insufficient_evidence_below_min_sample() -> None:
    result = classify_conditional_edge(_base_inputs(n_sample=10))
    assert result.primary == ConditionalEdgeDecision.INSUFFICIENT_EVIDENCE


def test_reject_when_core_expectancy_ci_touches_zero() -> None:
    result = classify_conditional_edge(_base_inputs(expectancy_ci_low=-0.001))
    assert result.primary == ConditionalEdgeDecision.REJECT


def test_reject_when_high_cost_profit_factor_below_one() -> None:
    result = classify_conditional_edge(_base_inputs(profit_factor_high_cost=0.9))
    assert result.primary == ConditionalEdgeDecision.REJECT


def test_reject_when_permutation_p_not_significant() -> None:
    result = classify_conditional_edge(_base_inputs(permutation_p_value_fdr_adjusted=0.20))
    assert result.primary == ConditionalEdgeDecision.REJECT


def test_event_dependent_when_excluding_aug2024_flips_negative() -> None:
    result = classify_conditional_edge(_base_inputs(positive_excluding_aug2024=False))
    assert result.primary == ConditionalEdgeDecision.EVENT_DEPENDENT


def test_event_dependent_when_excluding_apr2025_flips_negative() -> None:
    result = classify_conditional_edge(_base_inputs(positive_excluding_apr2025=False))
    assert result.primary == ConditionalEdgeDecision.EVENT_DEPENDENT


def test_event_dependent_when_a_single_episode_removal_flips_negative() -> None:
    result = classify_conditional_edge(
        _base_inputs(positive_excluding_each_major_episode=False)
    )
    assert result.primary == ConditionalEdgeDecision.EVENT_DEPENDENT


def test_event_dependent_when_a_single_year_removal_flips_negative() -> None:
    result = classify_conditional_edge(_base_inputs(positive_excluding_each_year=False))
    assert result.primary == ConditionalEdgeDecision.EVENT_DEPENDENT


def test_timing_dependent_when_shifted_offsets_are_also_mostly_positive() -> None:
    result = classify_conditional_edge(
        _base_inputs(timing_placebo_positive_fraction=0.6)
    )
    assert result.primary == ConditionalEdgeDecision.TIMING_DEPENDENT


def test_event_dependent_takes_priority_over_timing_dependent() -> None:
    result = classify_conditional_edge(
        _base_inputs(positive_excluding_aug2024=False, timing_placebo_positive_fraction=0.9)
    )
    assert result.primary == ConditionalEdgeDecision.EVENT_DEPENDENT


def test_robust_conditional_when_everything_passes() -> None:
    result = classify_conditional_edge(_base_inputs())
    assert result.primary == ConditionalEdgeDecision.ROBUST_CONDITIONAL


def test_regime_dependent_when_control_bucket_also_positive() -> None:
    result = classify_conditional_edge(_base_inputs(control_bucket_also_positive=True))
    assert result.primary == ConditionalEdgeDecision.REGIME_DEPENDENT


def test_regime_dependent_when_robustness_axis_untested() -> None:
    result = classify_conditional_edge(
        _base_inputs(positive_excluding_each_major_episode=None)
    )
    assert result.primary == ConditionalEdgeDecision.REGIME_DEPENDENT


def test_score_independent_attaches_as_secondary_on_robust_conditional() -> None:
    result = classify_conditional_edge(
        _base_inputs(score_adds_no_discriminating_power=True)
    )
    assert result.primary == ConditionalEdgeDecision.ROBUST_CONDITIONAL
    assert ConditionalEdgeDecision.SCORE_INDEPENDENT in result.secondary


def test_score_independent_attaches_as_secondary_on_event_dependent() -> None:
    result = classify_conditional_edge(
        _base_inputs(
            positive_excluding_aug2024=False, score_adds_no_discriminating_power=True
        )
    )
    assert result.primary == ConditionalEdgeDecision.EVENT_DEPENDENT
    assert ConditionalEdgeDecision.SCORE_INDEPENDENT in result.secondary


def test_score_independent_absent_when_score_does_discriminate() -> None:
    result = classify_conditional_edge(
        _base_inputs(score_adds_no_discriminating_power=False)
    )
    assert result.secondary == []
