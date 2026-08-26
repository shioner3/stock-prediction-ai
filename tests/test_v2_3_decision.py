from __future__ import annotations

from v2.causal.decision import V2_3Decision, V2_3DecisionInputs, classify_v2_3_decision


def _base_inputs(**overrides) -> V2_3DecisionInputs:
    defaults = dict(
        primary_q5_q1_spread=-0.001,
        day_cluster_spread_ci_high=-0.0002,
        block_spread_ci_high=-0.0001,
        q1_permutation_p_value=0.001,
        fdr_significant=True,
        holding_period_negative_fraction=6 / 7,
        year_negative_fraction=4 / 5,
        regime_negative_fraction=3 / 3,
        survives_event_exclusion=True,
    )
    defaults.update(overrides)
    return V2_3DecisionInputs(**defaults)


def test_reject_when_spread_not_negative() -> None:
    result = classify_v2_3_decision(_base_inputs(primary_q5_q1_spread=0.001))
    assert result.decision == V2_3Decision.REJECT


def test_reject_when_spread_missing() -> None:
    result = classify_v2_3_decision(_base_inputs(primary_q5_q1_spread=None))
    assert result.decision == V2_3Decision.REJECT


def test_event_dependent_negative_when_exclusion_fails() -> None:
    result = classify_v2_3_decision(_base_inputs(survives_event_exclusion=False))
    assert result.decision == V2_3Decision.EVENT_DEPENDENT_NEGATIVE


def test_weak_evidence_when_core_gate_fails() -> None:
    result = classify_v2_3_decision(_base_inputs(fdr_significant=False))
    assert result.decision == V2_3Decision.WEAK_EVIDENCE


def test_weak_evidence_when_block_ci_crosses_zero() -> None:
    result = classify_v2_3_decision(_base_inputs(block_spread_ci_high=0.0005))
    assert result.decision == V2_3Decision.WEAK_EVIDENCE


def test_structurally_robust_negative_when_all_axes_pass() -> None:
    result = classify_v2_3_decision(_base_inputs())
    assert result.decision == V2_3Decision.STRUCTURALLY_ROBUST_NEGATIVE


def test_conditional_negative_when_reproducibility_partial() -> None:
    result = classify_v2_3_decision(
        _base_inputs(holding_period_negative_fraction=0.2, year_negative_fraction=0.2)
    )
    assert result.decision == V2_3Decision.CONDITIONAL_NEGATIVE
