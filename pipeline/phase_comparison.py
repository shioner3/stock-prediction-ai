"""Phase 7 section 18/21: compares two WalkForwardReport JSON dumps
(e.g. Phase 6.5's 2022-01~2024-06 result vs Phase 7's independent
2024-07~2026-08 result) signal-by-signal, and classifies each into
Case A/B/C/D per the Phase 7 spec's own definitions:

    A: PF(base) <= 1 in BOTH phases  -> no support for the Signal.
    B: PF(base) <= 1 in Phase 6.5, > 1 in Phase 7 -> flag as a possible
       regime/period-dependence finding, NOT grounds to add a new Signal.
    C: PF(base) > 1 in Phase 6.5, <= 1 in Phase 7 -> the earlier result
       did not reproduce.
    D: PF(base) > 1 in BOTH phases -> the only case worth treating as a
       real candidate for further (human) review - even then, never
       auto-adopted.

This module only reads two already-computed reports and reshapes them
for reporting; it recomputes nothing and does not touch
backtest/decision.py's Decision framework (which is applied independently
by run_walk_forward() itself, once per phase, and is not re-derived here).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalComparison:
    direction: str
    signal_name: str
    present_in_phase6_5: bool
    present_in_phase7: bool
    trades_p65: int | None
    trades_p7: int | None
    unique_tickers_p65: int | None
    unique_tickers_p7: int | None
    win_rate_p65: float | None
    win_rate_p7: float | None
    pf_base_p65: float | None
    pf_base_p7: float | None
    pf_base_delta: float | None
    expectancy_base_p65: float | None
    expectancy_base_p7: float | None
    expectancy_base_delta: float | None
    permutation_p_p65: float | None
    permutation_p_p7: float | None
    decision_p65: str | None
    decision_p7: str | None
    case: str  # "A", "B", "C", "D", or "INSUFFICIENT_DATA"


def _signal_lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {(s["direction"], s["signal_name"]): s for s in report.get("signal_results", [])}


def _base_pf(signal_result: dict | None) -> float | None:
    if signal_result is None:
        return None
    tier = signal_result.get("oos_metrics_by_cost_tier", {}).get("base")
    return tier.get("profit_factor") if tier else None


def _base_expectancy(signal_result: dict | None) -> float | None:
    if signal_result is None:
        return None
    tier = signal_result.get("oos_metrics_by_cost_tier", {}).get("base")
    return tier.get("expectancy") if tier else None


def classify_case(pf_base_p65: float | None, pf_base_p7: float | None) -> str:
    if pf_base_p65 is None or pf_base_p7 is None:
        return "INSUFFICIENT_DATA"
    above_p65 = pf_base_p65 > 1.0
    above_p7 = pf_base_p7 > 1.0
    if not above_p65 and not above_p7:
        return "A"
    if not above_p65 and above_p7:
        return "B"
    if above_p65 and not above_p7:
        return "C"
    return "D"


def compare_reports(phase6_5_report: dict, phase7_report: dict) -> list[SignalComparison]:
    p65_by_key = _signal_lookup(phase6_5_report)
    p7_by_key = _signal_lookup(phase7_report)
    all_keys = sorted(set(p65_by_key) | set(p7_by_key))

    comparisons = []
    for direction, signal_name in all_keys:
        s65 = p65_by_key.get((direction, signal_name))
        s7 = p7_by_key.get((direction, signal_name))

        pf65 = _base_pf(s65)
        pf7 = _base_pf(s7)
        exp65 = _base_expectancy(s65)
        exp7 = _base_expectancy(s7)

        trades65 = s65["oos_metrics_by_cost_tier"]["base"]["n_trades"] if s65 else None
        trades7 = s7["oos_metrics_by_cost_tier"]["base"]["n_trades"] if s7 else None
        tickers65 = len(s65["ticker_metrics"]) if s65 else None
        tickers7 = len(s7["ticker_metrics"]) if s7 else None
        win65 = s65["oos_metrics_by_cost_tier"]["base"]["win_rate"] if s65 else None
        win7 = s7["oos_metrics_by_cost_tier"]["base"]["win_rate"] if s7 else None
        perm65 = s65["permutation"]["p_value"] if s65 and s65.get("permutation") else None
        perm7 = s7["permutation"]["p_value"] if s7 and s7.get("permutation") else None

        comparisons.append(
            SignalComparison(
                direction=direction,
                signal_name=signal_name,
                present_in_phase6_5=s65 is not None,
                present_in_phase7=s7 is not None,
                trades_p65=trades65,
                trades_p7=trades7,
                unique_tickers_p65=tickers65,
                unique_tickers_p7=tickers7,
                win_rate_p65=win65,
                win_rate_p7=win7,
                pf_base_p65=pf65,
                pf_base_p7=pf7,
                pf_base_delta=(pf7 - pf65) if pf65 is not None and pf7 is not None else None,
                expectancy_base_p65=exp65,
                expectancy_base_p7=exp7,
                expectancy_base_delta=(
                    (exp7 - exp65) if exp65 is not None and exp7 is not None else None
                ),
                permutation_p_p65=perm65,
                permutation_p_p7=perm7,
                decision_p65=s65["decision"] if s65 else None,
                decision_p7=s7["decision"] if s7 else None,
                case=classify_case(pf65, pf7),
            )
        )
    return comparisons
