from __future__ import annotations

from backtest.decision import Decision, DecisionInputs, classify, window_consistency


def _inputs(**overrides: object) -> DecisionInputs:
    base = dict(
        oos_trade_count=50,
        min_oos_trades=30,
        total_windows=5,
        windows_with_positive_pf=4,
        aggregate_oos_expectancy=0.02,
        aggregate_oos_profit_factor=1.5,
        expectancy_ci_low=0.005,
        permutation_p_value=0.03,
        high_cost_expectancy=0.01,
    )
    base.update(overrides)
    return DecisionInputs(**base)  # type: ignore[arg-type]


def test_accept_candidate_when_every_condition_met() -> None:
    assert classify(_inputs()) == Decision.ACCEPT_CANDIDATE


def test_insufficient_evidence_below_min_trades() -> None:
    result = classify(_inputs(oos_trade_count=10))
    assert result == Decision.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_no_windows() -> None:
    result = classify(_inputs(total_windows=0, windows_with_positive_pf=0))
    assert result == Decision.INSUFFICIENT_EVIDENCE


def test_reject_when_expectancy_and_pf_both_negative() -> None:
    result = classify(
        _inputs(
            aggregate_oos_expectancy=-0.01,
            aggregate_oos_profit_factor=0.7,
            windows_with_positive_pf=1,
        )
    )
    assert result == Decision.REJECT


def test_reject_when_high_cost_kills_edge_plus_one_other_reason() -> None:
    result = classify(
        _inputs(
            high_cost_expectancy=-0.001,
            windows_with_positive_pf=1,  # consistency < 0.5 too
        )
    )
    assert result == Decision.REJECT


def test_insufficient_evidence_when_ci_crosses_zero() -> None:
    result = classify(_inputs(expectancy_ci_low=-0.002))
    assert result == Decision.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_when_permutation_p_value_too_high() -> None:
    result = classify(_inputs(permutation_p_value=0.5))
    assert result == Decision.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_when_only_one_window() -> None:
    result = classify(
        _inputs(total_windows=1, windows_with_positive_pf=1, oos_trade_count=40)
    )
    assert result == Decision.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_not_reject_for_a_single_borderline_metric() -> None:
    """Only ONE reject-leaning signal (weak window consistency) - not
    enough alone to REJECT (requires >= 2 per backtest/decision.py), but
    also doesn't clear the ACCEPT_CANDIDATE bar - lands in the middle.
    """
    result = classify(_inputs(windows_with_positive_pf=2, total_windows=5))
    assert result == Decision.INSUFFICIENT_EVIDENCE


def test_window_consistency_helper() -> None:
    assert window_consistency(_inputs(windows_with_positive_pf=3, total_windows=5)) == 0.6
    assert window_consistency(_inputs(total_windows=0)) == 0.0


def test_missing_aggregate_metrics_gives_insufficient_evidence() -> None:
    result = classify(_inputs(aggregate_oos_expectancy=None, aggregate_oos_profit_factor=None))
    assert result == Decision.INSUFFICIENT_EVIDENCE
