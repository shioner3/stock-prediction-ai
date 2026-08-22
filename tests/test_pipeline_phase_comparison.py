from __future__ import annotations

import pytest

from pipeline.phase_comparison import classify_case, compare_reports


def _signal_result(
    direction: str,
    signal_name: str,
    pf_base: float | None,
    expectancy_base: float = 0.0,
    n_trades: int = 100,
    win_rate: float = 0.5,
    n_tickers: int = 10,
    permutation_p: float | None = 0.5,
    decision: str = "REJECT",
) -> dict:
    return {
        "direction": direction,
        "signal_name": signal_name,
        "oos_metrics_by_cost_tier": {
            "base": {
                "n_trades": n_trades, "win_rate": win_rate,
                "profit_factor": pf_base, "expectancy": expectancy_base,
            },
        },
        "ticker_metrics": {f"T{i}": {} for i in range(n_tickers)},
        "permutation": {"p_value": permutation_p} if permutation_p is not None else None,
        "decision": decision,
    }


def _report(*signal_results: dict) -> dict:
    return {"signal_results": list(signal_results)}


# --- classify_case -----------------------------------------------------------


def test_classify_case_a_both_below_one() -> None:
    assert classify_case(0.9, 0.8) == "A"


def test_classify_case_b_regime_dependence() -> None:
    assert classify_case(0.9, 1.2) == "B"


def test_classify_case_c_did_not_reproduce() -> None:
    assert classify_case(1.2, 0.9) == "C"


def test_classify_case_d_both_above_one() -> None:
    assert classify_case(1.1, 1.3) == "D"


def test_classify_case_boundary_pf_exactly_one_counts_as_not_above() -> None:
    assert classify_case(1.0, 1.0) == "A"


def test_classify_case_insufficient_data_when_either_side_missing() -> None:
    assert classify_case(None, 1.2) == "INSUFFICIENT_DATA"
    assert classify_case(1.2, None) == "INSUFFICIENT_DATA"
    assert classify_case(None, None) == "INSUFFICIENT_DATA"


# --- compare_reports -----------------------------------------------------------


def test_compare_reports_matches_signals_by_direction_and_name() -> None:
    p65 = _report(_signal_result("LONG", "long_pullback", pf_base=0.9))
    p7 = _report(_signal_result("LONG", "long_pullback", pf_base=1.2))

    result = compare_reports(p65, p7)
    assert len(result) == 1
    c = result[0]
    assert c.direction == "LONG"
    assert c.signal_name == "long_pullback"
    assert c.pf_base_p65 == 0.9
    assert c.pf_base_p7 == 1.2
    assert c.case == "B"
    assert c.present_in_phase6_5 is True
    assert c.present_in_phase7 is True


def test_compare_reports_computes_deltas() -> None:
    p65 = _report(_signal_result("LONG", "x", pf_base=1.0, expectancy_base=0.001))
    p7 = _report(_signal_result("LONG", "x", pf_base=1.3, expectancy_base=0.004))
    c = compare_reports(p65, p7)[0]
    assert c.pf_base_delta == pytest.approx(0.3)
    assert c.expectancy_base_delta == pytest.approx(0.003)


def test_compare_reports_handles_signal_missing_from_one_side() -> None:
    p65 = _report(_signal_result("LONG", "only_in_p65", pf_base=0.8))
    p7 = _report()

    result = compare_reports(p65, p7)
    assert len(result) == 1
    c = result[0]
    assert c.present_in_phase6_5 is True
    assert c.present_in_phase7 is False
    assert c.pf_base_p7 is None
    assert c.case == "INSUFFICIENT_DATA"


def test_compare_reports_handles_null_profit_factor() -> None:
    p65 = _report(_signal_result("LONG", "x", pf_base=None))
    p7 = _report(_signal_result("LONG", "x", pf_base=1.1))
    c = compare_reports(p65, p7)[0]
    assert c.case == "INSUFFICIENT_DATA"
    assert c.pf_base_delta is None


def test_compare_reports_sorted_by_direction_then_name() -> None:
    p65 = _report(
        _signal_result("SHORT", "b", pf_base=0.9),
        _signal_result("LONG", "a", pf_base=0.9),
        _signal_result("LONG", "z", pf_base=0.9),
    )
    p7 = _report(
        _signal_result("SHORT", "b", pf_base=0.9),
        _signal_result("LONG", "a", pf_base=0.9),
        _signal_result("LONG", "z", pf_base=0.9),
    )
    result = compare_reports(p65, p7)
    keys = [(c.direction, c.signal_name) for c in result]
    assert keys == sorted(keys)


def test_compare_reports_carries_permutation_p_and_decision() -> None:
    p65 = _report(
        _signal_result(
            "LONG", "x", pf_base=1.1, permutation_p=0.03, decision="INSUFFICIENT_EVIDENCE"
        )
    )
    p7 = _report(
        _signal_result("LONG", "x", pf_base=1.2, permutation_p=0.01, decision="ACCEPT_CANDIDATE")
    )
    c = compare_reports(p65, p7)[0]
    assert c.permutation_p_p65 == 0.03
    assert c.permutation_p_p7 == 0.01
    assert c.decision_p65 == "INSUFFICIENT_EVIDENCE"
    assert c.decision_p7 == "ACCEPT_CANDIDATE"
