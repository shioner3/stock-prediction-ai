from __future__ import annotations

from ensemble.decision import EnsembleDecision, EnsembleDecisionInputs, classify_ensemble


def _inputs(**overrides: object) -> EnsembleDecisionInputs:
    base: dict[str, object] = dict(
        n_sample=100,
        pct_trading_days_with_occurrence=0.10,
        expectancy_ci_low=0.001,
        profit_factor_high_cost=1.2,
        permutation_p_value=0.01,
        positive_excluding_aug2024=True,
        positive_in_bull=True,
        positive_in_neutral=True,
        positive_in_bear=True,
    )
    base.update(overrides)
    return EnsembleDecisionInputs(**base)  # type: ignore[arg-type]


def test_insufficient_evidence_below_min_sample() -> None:
    assert classify_ensemble(_inputs(n_sample=29)) == EnsembleDecision.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_gate_checked_before_frequency() -> None:
    result = classify_ensemble(_inputs(n_sample=5, pct_trading_days_with_occurrence=0.001))
    assert result == EnsembleDecision.INSUFFICIENT_EVIDENCE


def test_frequency_too_low() -> None:
    result = classify_ensemble(_inputs(pct_trading_days_with_occurrence=0.01))
    assert result == EnsembleDecision.FREQUENCY_TOO_LOW


def test_reject_when_expectancy_ci_not_positive() -> None:
    assert classify_ensemble(_inputs(expectancy_ci_low=-0.001)) == EnsembleDecision.REJECT
    assert classify_ensemble(_inputs(expectancy_ci_low=None)) == EnsembleDecision.REJECT


def test_reject_when_high_cost_pf_not_above_one() -> None:
    assert classify_ensemble(_inputs(profit_factor_high_cost=0.9)) == EnsembleDecision.REJECT
    assert classify_ensemble(_inputs(profit_factor_high_cost=None)) == EnsembleDecision.REJECT


def test_reject_when_permutation_not_significant() -> None:
    assert classify_ensemble(_inputs(permutation_p_value=0.5)) == EnsembleDecision.REJECT
    assert classify_ensemble(_inputs(permutation_p_value=None)) == EnsembleDecision.REJECT


def test_event_dependent_when_negative_excluding_aug2024() -> None:
    result = classify_ensemble(_inputs(positive_excluding_aug2024=False))
    assert result == EnsembleDecision.EVENT_DEPENDENT_ENSEMBLE


def test_regime_dependent_when_one_regime_negative() -> None:
    result = classify_ensemble(_inputs(positive_in_bear=False))
    assert result == EnsembleDecision.REGIME_DEPENDENT_ENSEMBLE


def test_regime_dependent_takes_priority_only_after_event_check() -> None:
    # Event-dependence is checked FIRST - if it fails, regime state is irrelevant.
    result = classify_ensemble(
        _inputs(positive_excluding_aug2024=False, positive_in_bear=False)
    )
    assert result == EnsembleDecision.EVENT_DEPENDENT_ENSEMBLE


def test_robust_ensemble_when_all_axes_pass() -> None:
    assert classify_ensemble(_inputs()) == EnsembleDecision.ROBUST_ENSEMBLE


def test_robust_ensemble_when_regime_axis_untested() -> None:
    # None (untested) must never count as "passed" on its own, but if
    # every TESTED regime axis passes and event-exclusion also passes,
    # the result is still ROBUST_ENSEMBLE - an untested axis doesn't
    # block it either (there's simply no evidence against it).
    result = classify_ensemble(
        _inputs(positive_in_bull=None, positive_in_neutral=True, positive_in_bear=True)
    )
    assert result == EnsembleDecision.ROBUST_ENSEMBLE


def test_event_dependence_none_does_not_trigger_event_dependent() -> None:
    # None means "could not test" (e.g. zero trades outside the event
    # window), not "found to be event-dependent" - must not be
    # misclassified as EVENT_DEPENDENT_ENSEMBLE.
    result = classify_ensemble(_inputs(positive_excluding_aug2024=None))
    assert result == EnsembleDecision.ROBUST_ENSEMBLE
