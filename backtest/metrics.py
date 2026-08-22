"""Aggregate metrics from a Trade Record DataFrame (backtest/engine.py's
output).

Phase 4 implements only descriptive statistics - win rate, average/median
return, total return, gross profit/loss, profit factor, expectancy.
Bootstrap confidence intervals, permutation tests, and return-distribution
percentiles are Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float | None
    average_return: float | None
    median_return: float | None
    total_return: float | None
    gross_profit: float | None
    gross_loss: float | None
    profit_factor: float | None
    expectancy: float | None


def compute_metrics(trades: pd.DataFrame) -> BacktestMetrics:
    n = len(trades)
    if n == 0:
        return BacktestMetrics(
            n_trades=0, win_rate=None, average_return=None, median_return=None,
            total_return=None, gross_profit=None, gross_loss=None,
            profit_factor=None, expectancy=None,
        )

    returns = trades["return"].to_numpy(dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(np.sum(losses)) if len(losses) else 0.0  # <= 0

    if gross_loss < 0:
        profit_factor: float | None = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None  # no wins and no losses - every trade was exactly flat

    average_return = float(np.mean(returns))

    return BacktestMetrics(
        n_trades=n,
        win_rate=len(wins) / n,
        average_return=average_return,
        median_return=float(np.median(returns)),
        # Sum of per-trade returns under equal position sizing (no
        # compounding, no position-sizing model in Phase 4).
        total_return=float(np.sum(returns)),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        # Equal position sizing -> per-trade expectancy is just the mean
        # return; kept as a separate field because Phase 4's spec asks
        # for it explicitly and a future position-sizing model would
        # make the two diverge.
        expectancy=average_return,
    )


def compute_metrics_by_signal(trades: pd.DataFrame) -> dict[str, BacktestMetrics]:
    if trades.empty:
        return {}
    return {name: compute_metrics(group) for name, group in trades.groupby("signal_name")}


def compute_metrics_by_ticker(trades: pd.DataFrame) -> dict[str, BacktestMetrics]:
    """Phase 6.5 section 17/18: per-ticker breakdown of an already-computed
    Trade Record set (e.g. one signal's aggregate OOS trades) - the basis
    for cross-sectional concentration analysis. Reuses compute_metrics()
    unchanged, exactly like compute_metrics_by_signal() above.
    """
    if trades.empty:
        return {}
    return {str(name): compute_metrics(group) for name, group in trades.groupby("ticker")}
