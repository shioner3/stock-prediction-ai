"""Spec section 17: Economic Significance - does this look like a
tradeable swing-trading edge, not just a statistically-significant one?
Most fields (mean return/win rate/PF/expectancy via `v2.stats.
ReturnStats`, MaxDD/Sharpe/Turnover via `v3.validation.topn_portfolio.
TopNPortfolioMetrics`) are already computed by existing V2/V3-3
primitives - only Max Losing Streak and Annualized Return are genuinely
new here (Sharpe/MaxDD's OWN "overlapping trades" caveat, documented in
`v3/validation/topn_portfolio.py`, applies identically to every number in
this module).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v3.validation.topn_portfolio import TRADING_DAYS_PER_YEAR, TopNPortfolioMetrics


def max_losing_streak(returns: np.ndarray) -> int:
    """Longest run of consecutive negative-return days in ARRIVAL (date)
    order - `returns` must already be date-sorted by the caller.
    """
    streak = 0
    longest = 0
    for r in returns:
        if r < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest


def annualized_return(
    mean_daily_return: float, window_days: int,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """(1 + mean_return)^(periods_per_year / window_days) - 1, treating
    each h-day return as if compounded non-overlapping - the SAME
    simplification `TopNPortfolioMetrics.sharpe`/`.cumulative_return`
    already documents (overlapping windows, not a real trading schedule).
    """
    periods = periods_per_year / window_days
    return float((1.0 + mean_daily_return) ** periods - 1.0)


@dataclass(frozen=True)
class EconomicSignificance:
    n: int
    window_days: int
    expected_return_per_trade: float | None
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    max_drawdown: float | None
    max_losing_streak_days: int
    turnover: float | None
    average_holding_period_days: int
    annualized_return_pct: float | None
    sharpe: float | None


def compute_economic_significance(topn_metrics: TopNPortfolioMetrics) -> EconomicSignificance:
    stats = topn_metrics.base.stats
    returns_in_date_order = [
        d.equal_weight_return for d in topn_metrics.base.daily_returns
        if d.equal_weight_return is not None
    ]
    streak = max_losing_streak(np.array(returns_in_date_order, dtype=float))
    ann_return = (
        annualized_return(stats.mean_return, topn_metrics.window_days)
        if stats.mean_return is not None else None
    )
    return EconomicSignificance(
        n=stats.n, window_days=topn_metrics.window_days,
        expected_return_per_trade=stats.mean_return, win_rate=stats.win_rate,
        profit_factor=stats.profit_factor, expectancy=stats.expectancy,
        max_drawdown=topn_metrics.max_drawdown, max_losing_streak_days=streak,
        turnover=topn_metrics.turnover, average_holding_period_days=topn_metrics.window_days,
        annualized_return_pct=ann_return, sharpe=topn_metrics.sharpe,
    )
