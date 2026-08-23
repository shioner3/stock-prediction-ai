from __future__ import annotations

from v3.validation.decision import V3_3Decision, V3_3DecisionInputs, classify_v3_3_decision


def _base_inputs(**overrides) -> V3_3DecisionInputs:
    defaults = dict(
        n_windows=6,
        primary_q5_q1_spread=0.002,
        rank_ic_mean=0.02,
        window_direction_agreement=0.8,
        day_cluster_ci_low=0.0005,
        block_ci_low=0.0002,
        q5_permutation_p=0.001,
        fdr_significant=True,
        survives_event_exclusion=True,
        top5_mean_return=0.001,
        top10_mean_return=0.0008,
        top20_mean_return=0.0005,
        random_baseline_spread=0.0001,
        momentum_baseline_spread=0.0005,
        v2_score_baseline_spread=0.0008,
    )
    defaults.update(overrides)
    return V3_3DecisionInputs(**defaults)


def test_insufficient_evidence_when_too_few_windows() -> None:
    result = classify_v3_3_decision(_base_inputs(n_windows=1))
    assert result.decision == V3_3Decision.INSUFFICIENT_EVIDENCE


def test_reject_when_spread_not_positive() -> None:
    result = classify_v3_3_decision(_base_inputs(primary_q5_q1_spread=-0.001))
    assert result.decision == V3_3Decision.REJECT


def test_reject_when_does_not_beat_random() -> None:
    result = classify_v3_3_decision(_base_inputs(random_baseline_spread=0.01))
    assert result.decision == V3_3Decision.REJECT


def test_reject_when_does_not_beat_simple_baselines() -> None:
    result = classify_v3_3_decision(
        _base_inputs(momentum_baseline_spread=0.01, v2_score_baseline_spread=0.01)
    )
    assert result.decision == V3_3Decision.REJECT


def test_weak_evidence_when_core_bootstrap_gate_fails() -> None:
    result = classify_v3_3_decision(_base_inputs(day_cluster_ci_low=-0.001))
    assert result.decision == V3_3Decision.WEAK_EVIDENCE


def test_accept_candidate_when_everything_passes() -> None:
    result = classify_v3_3_decision(_base_inputs())
    assert result.decision == V3_3Decision.ACCEPT_CANDIDATE
    assert result.question_a_predictive_power is True
    assert result.question_b_ranking_valid is True
    assert result.question_c_topn_viable is True
    assert result.question_d_beats_v1_v2 is True


def test_question_c_can_be_no_while_question_a_is_yes() -> None:
    """spec section 27's own example: predictive power without viable
    Top-N selection is a valid, separate outcome."""
    result = classify_v3_3_decision(_base_inputs(top5_mean_return=-0.001))
    assert result.question_a_predictive_power is True
    assert result.question_c_topn_viable is False
