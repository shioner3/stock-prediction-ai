"""Signal Decision classification (Phase 6 section 23-26).

Produces one of ACCEPT_CANDIDATE / REJECT / INSUFFICIENT_EVIDENCE. This
is a SUGGESTION for a human reviewer, never an automatic accept/reject
action - nothing in this codebase deploys, removes, or reconfigures a
Signal based on this output. See README's "Signal Decision" section for
the full reasoning behind each threshold; none of them were tuned
against this project's own OOS results (they were fixed as part of the
Phase 6 spec itself, before any OOS run).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class DecisionInputs:
    oos_trade_count: int
    min_oos_trades: int
    total_windows: int
    windows_with_positive_pf: int
    aggregate_oos_expectancy: float | None
    aggregate_oos_profit_factor: float | None
    # Bootstrap CI lower bound for expectancy/mean_return, at the "base"
    # cost tier - "does the CI exclude zero" is the stability check.
    expectancy_ci_low: float | None
    permutation_p_value: float | None
    # expectancy at the "high" transaction-cost tier - a Signal that
    # only works at zero cost is not a usable candidate.
    high_cost_expectancy: float | None


def window_consistency(inputs: DecisionInputs) -> float:
    if inputs.total_windows == 0:
        return 0.0
    return inputs.windows_with_positive_pf / inputs.total_windows


def classify(inputs: DecisionInputs) -> Decision:
    if inputs.oos_trade_count < inputs.min_oos_trades or inputs.total_windows == 0:
        return Decision.INSUFFICIENT_EVIDENCE
    if inputs.aggregate_oos_expectancy is None or inputs.aggregate_oos_profit_factor is None:
        return Decision.INSUFFICIENT_EVIDENCE

    consistency = window_consistency(inputs)

    reject_reasons = 0
    if inputs.aggregate_oos_expectancy <= 0:
        reject_reasons += 1
    if inputs.aggregate_oos_profit_factor < 1.0:
        reject_reasons += 1
    if consistency < 0.5:
        reject_reasons += 1
    if inputs.high_cost_expectancy is not None and inputs.high_cost_expectancy <= 0:
        reject_reasons += 1
    if reject_reasons >= 2:
        return Decision.REJECT

    accept_conditions = [
        inputs.aggregate_oos_expectancy > 0,
        inputs.aggregate_oos_profit_factor > 1.0,
        inputs.expectancy_ci_low is not None and inputs.expectancy_ci_low > 0,
        inputs.permutation_p_value is not None and inputs.permutation_p_value < 0.10,
        inputs.total_windows >= 2 and consistency >= 0.5,
        inputs.high_cost_expectancy is not None and inputs.high_cost_expectancy > 0,
    ]
    if all(accept_conditions):
        return Decision.ACCEPT_CANDIDATE

    return Decision.INSUFFICIENT_EVIDENCE
