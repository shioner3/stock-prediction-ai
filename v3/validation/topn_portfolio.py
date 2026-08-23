"""Top-N daily portfolio simulation (spec section 9). Builds on Phase
V2-2's `v2/validation/topn.py::compute_topn_daily_returns()`/
`summarize_topn()` (frozen, unmodified - `score_col` swapped to a Model
prediction column instead of V2's `total_score`) with the ADDITIONAL
portfolio-style metrics spec section 9 asks for that V2-2 never needed
(Cumulative Return / Max Drawdown / Sharpe / Sortino / Turnover) - these
are genuinely new, since V2-2 evaluated Top-N purely as a ranking-quality
statistic, never as a virtual equity curve.

IMPORTANT SIMPLIFICATION (documented, not hidden): each day's Top-N
equal-weight return is the horizon-h FORWARD return (Close[t+h]/Close[t]-1),
so consecutive days' "trades" OVERLAP in time (a 5-day return on Monday
and on Tuesday share 4 of the same 5 trading days). Chaining these
day-by-day into one cumulative-return/drawdown/Sharpe series therefore
does NOT represent a real, non-overlapping trading strategy - it is the
same "daily entry, ranking-quality" convention spec section 9 itself
frames as "実際の運用シミュレーション...ではなく" (not a real simulation,
not a trading-adoption decision). Reported as such in the Phase report's
Limitations section.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd

from v2.validation.topn import TopNResult, compute_topn_daily_returns, summarize_topn

TRADING_DAYS_PER_YEAR = 252
ZERO_STD_EPSILON = 1e-10


@dataclass(frozen=True)
class TopNPortfolioMetrics:
    n: int
    window_days: int
    base: TopNResult
    cumulative_return: float | None
    max_drawdown: float | None
    sharpe: float | None
    sortino: float | None
    turnover: float | None


def compute_daily_top_n_tickers(
    panel: pd.DataFrame, n: int, score_col: str = "prediction", date_col: str = "date"
) -> dict[date_type, set[str]]:
    result: dict[date_type, set[str]] = {}
    for day, group in panel.groupby(date_col, sort=True):
        valid = group.dropna(subset=[score_col])
        top = valid.nlargest(n, score_col)
        result[day] = set(top["ticker"])
    return result


def compute_turnover(daily_tickers: dict[date_type, set[str]]) -> float | None:
    """Mean day-over-day fraction of the N-slot portfolio that changes
    membership (0 = identical selection every day, 1 = fully replaced).
    """
    days = sorted(daily_tickers.keys())
    if len(days) < 2:
        return None
    changes = []
    for i in range(1, len(days)):
        prev, curr = daily_tickers[days[i - 1]], daily_tickers[days[i]]
        denom = max(len(prev), len(curr), 1)
        changed = len(prev.symmetric_difference(curr))
        changes.append(changed / (2 * denom))
    return float(np.mean(changes))


def compute_topn_portfolio_metrics(
    panel: pd.DataFrame,
    n: int,
    return_col: str,
    window_days: int,
    score_col: str = "prediction",
    date_col: str = "date",
) -> TopNPortfolioMetrics:
    daily_returns = compute_topn_daily_returns(
        panel, n, return_col, score_col=score_col, date_col=date_col
    )
    base = summarize_topn(daily_returns, n, window_days)

    returns = np.array(
        [d.equal_weight_return for d in daily_returns if d.equal_weight_return is not None],
        dtype=float,
    )
    if len(returns) == 0:
        return TopNPortfolioMetrics(n, window_days, base, None, None, None, None, None)

    cumulative_return = float(np.prod(1 + returns) - 1)
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1
    max_drawdown = float(drawdowns.min())

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    periods_per_year = TRADING_DAYS_PER_YEAR / window_days
    # A "perfectly constant" return series still yields a tiny nonzero
    # std (~1e-18, float64 variance-computation noise, not a real signal)
    # - ZERO_STD_EPSILON is far above that noise floor and far below any
    # realistic daily-return std (order 1e-3-1e-2), so it only ever
    # suppresses a numerically-undefined, not a genuinely small-but-real,
    # Sharpe/Sortino.
    sharpe = mean_r / std_r * (periods_per_year**0.5) if std_r > ZERO_STD_EPSILON else None

    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        mean_r / downside_std * (periods_per_year**0.5)
        if downside_std > ZERO_STD_EPSILON
        else None
    )

    daily_tickers = compute_daily_top_n_tickers(panel, n, score_col=score_col, date_col=date_col)
    turnover = compute_turnover(daily_tickers)

    return TopNPortfolioMetrics(
        n=n, window_days=window_days, base=base, cumulative_return=cumulative_return,
        max_drawdown=max_drawdown, sharpe=sharpe, sortino=sortino, turnover=turnover,
    )
