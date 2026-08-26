from __future__ import annotations

from v2.validation.decision import V2Decision, V2DecisionInputs, classify_v2_decision


def _robust_inputs(**overrides: object) -> V2DecisionInputs:
    defaults: dict[str, object] = dict(
        q5_q1_spread=0.01,
        block_bootstrap_ci_low=0.001,
        permutation_p_value=0.01,
        fdr_significant=True,
        holding_period_positive_fraction=0.8,
        year_positive_fraction=0.8,
        survives_event_exclusion=True,
        topn_reproduces=True,
        survives_low_cost=True,
        regime_dependent=False,
        holding_period_dependent=False,
        segment_dependent=False,
    )
    defaults.update(overrides)
    return V2DecisionInputs(**defaults)


def test_robust_candidate_when_everything_passes() -> None:
    result = classify_v2_decision(_robust_inputs())
    assert result.decision == V2Decision.ROBUST_CANDIDATE


def test_reject_when_spread_non_positive() -> None:
    result = classify_v2_decision(_robust_inputs(q5_q1_spread=-0.001))
    assert result.decision == V2Decision.REJECT


def test_reject_when_spread_none() -> None:
    result = classify_v2_decision(_robust_inputs(q5_q1_spread=None))
    assert result.decision == V2Decision.REJECT


def test_reject_when_block_bootstrap_crosses_zero() -> None:
    result = classify_v2_decision(_robust_inputs(block_bootstrap_ci_low=-0.001))
    assert result.decision == V2Decision.REJECT


def test_reject_when_permutation_not_significant() -> None:
    result = classify_v2_decision(_robust_inputs(permutation_p_value=0.5))
    assert result.decision == V2Decision.REJECT


def test_reject_when_fdr_not_significant() -> None:
    result = classify_v2_decision(_robust_inputs(fdr_significant=False))
    assert result.decision == V2Decision.REJECT


def test_reject_when_no_year_reproducibility() -> None:
    result = classify_v2_decision(_robust_inputs(year_positive_fraction=0.1))
    assert result.decision == V2Decision.REJECT


def test_reject_when_cost_kills_edge() -> None:
    result = classify_v2_decision(_robust_inputs(survives_low_cost=False))
    assert result.decision == V2Decision.REJECT


def test_reject_when_event_concentrated() -> None:
    result = classify_v2_decision(_robust_inputs(survives_event_exclusion=False))
    assert result.decision == V2Decision.REJECT


def test_weak_evidence_when_core_gate_fails_but_spread_positive() -> None:
    # spread positive and bootstrap CI stays above zero (so no REJECT
    # reason fires), but FDR significance is unknown (None, not False) -
    # fails the core significance gate without being an outright REJECT.
    result = classify_v2_decision(_robust_inputs(fdr_significant=None))
    assert result.decision == V2Decision.WEAK_EVIDENCE


def test_weak_evidence_when_holding_period_not_reproducible() -> None:
    result = classify_v2_decision(_robust_inputs(holding_period_positive_fraction=0.1))
    assert result.decision == V2Decision.WEAK_EVIDENCE


def test_weak_evidence_when_topn_does_not_reproduce() -> None:
    result = classify_v2_decision(_robust_inputs(topn_reproduces=False))
    assert result.decision == V2Decision.WEAK_EVIDENCE


def test_conditional_candidate_when_regime_dependent() -> None:
    result = classify_v2_decision(_robust_inputs(regime_dependent=True))
    assert result.decision == V2Decision.CONDITIONAL_CANDIDATE
    assert "regime" in result.reasons[0]


def test_conditional_candidate_when_holding_period_dependent() -> None:
    result = classify_v2_decision(_robust_inputs(holding_period_dependent=True))
    assert result.decision == V2Decision.CONDITIONAL_CANDIDATE


def test_conditional_candidate_when_segment_dependent() -> None:
    result = classify_v2_decision(_robust_inputs(segment_dependent=True))
    assert result.decision == V2Decision.CONDITIONAL_CANDIDATE


def test_reject_takes_priority_over_conditional() -> None:
    result = classify_v2_decision(
        _robust_inputs(regime_dependent=True, survives_low_cost=False)
    )
    assert result.decision == V2Decision.REJECT
