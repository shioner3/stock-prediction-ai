"""Long Candidate Test (spec section 9): pick the top N tickers by V2
Score each day (from within Q5, per the spec's own wording), equal-
weight their Forward Return, evaluate as a pure ranking-quality
statistic - explicitly NOT a real Backtest (no Entry/Exit timing, no
overlap suppression, no position sizing; just "if you'd equally
weighted today's top N candidates, what was the average N-day-later
return").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from v2.stats import ReturnStats, compute_return_stats

TOP_N_VALUES = (5, 10, 20)


@dataclass(frozen=True)
class TopNDailyReturn:
    date: object
    n_requested: int
    n_available: int
    equal_weight_return: float | None  # mean Forward Return of the (up to N) selected tickers


@dataclass(frozen=True)
class TopNResult:
    n: int
    window_days: int
    daily_returns: list[TopNDailyReturn]
    stats: ReturnStats  # over the per-day equal-weight returns (one observation per day)


def compute_topn_daily_returns(
    panel: pd.DataFrame, n: int, return_col: str, score_col: str = "total_score",
    date_col: str = "date",
) -> list[TopNDailyReturn]:
    results: list[TopNDailyReturn] = []
    for day, group in panel.groupby(date_col, sort=True):
        valid = group.dropna(subset=[score_col])
        top = valid.nlargest(n, score_col)
        resolved = top[return_col].dropna()
        results.append(
            TopNDailyReturn(
                date=day, n_requested=n, n_available=len(top),
                equal_weight_return=float(resolved.mean()) if len(resolved) else None,
            )
        )
    return results


def summarize_topn(daily_returns: list[TopNDailyReturn], n: int, window_days: int) -> TopNResult:
    returns = pd.Series([d.equal_weight_return for d in daily_returns], dtype=float)
    return TopNResult(
        n=n, window_days=window_days, daily_returns=daily_returns,
        stats=compute_return_stats(returns),
    )
