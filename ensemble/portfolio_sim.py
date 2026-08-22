"""Phase 12 section 24/25: OPTIONAL research-only Top-5 selection
simulation and its equity-curve metrics (CAGR/Sharpe/max drawdown - none
of these exist anywhere else in the codebase; backtest/metrics.py's
BacktestMetrics deliberately has no equity-curve concept at all, since
the frozen Backtest Engine has no position-sizing model - see that
module's docstring). This is new, independent analysis/aggregation code
only (spec section 1's explicit exception), never a change to the
frozen Backtest Engine, Signal, or Score logic.

Position sizing reuses the SAME fixed-notional convention Forward Test's
Paper Portfolio already uses (forward_test/portfolio.py) rather than
inventing a new one: initial_capital * per_trade_notional_fraction per
trade, no compounding.

Top-5 selection rule (FIXED before running, spec section 25 - never
tuned after seeing results): each trading day, among candidates whose
Ensemble dominant_direction is LONG or SHORT (never NEUTRAL), rank by
(total_signal_count DESC, total_score DESC) and take the top N. A
ticker/date with multiple co-firing same-direction Signals is ONE
candidate (Entry/Exit only depend on ticker+date+direction+hold_days,
never on which specific Signal fired - see
ensemble/portfolio_sim.py::dedupe_trades_by_ticker_date_direction).

Simplification (explicitly disclosed, not hidden - spec section 24's
"単なるForward Return分析と実際のBacktest結果を混同しない"): the equity
curve marks P&L only at each trade's EXIT date (a realized-only curve,
same convention as forward_test/portfolio.py's Paper Portfolio) and does
NOT enforce a real concurrent-open-position capital cap across
overlapping holding periods - it is a simple/lightweight ("簡易")
simulation as the spec itself requests, not a full portfolio state
machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics

TOP_N = 5
INITIAL_CAPITAL = 10_000_000.0
PER_TRADE_NOTIONAL_FRACTION = 0.01
TRADING_DAYS_PER_YEAR = 252


def dedupe_trades_by_ticker_date_direction(trades: pd.DataFrame) -> pd.DataFrame:
    """Multiple Signals co-firing on the same (ticker, signal_date,
    direction) all produce IDENTICAL trade economics (Entry=Open[t+1],
    Exit=Close[t+1+hold_days-1] depend only on ticker/date/direction/
    hold_days, never on signal_name) - keep exactly one row per key so a
    single Ensemble candidate is never counted as N simultaneous
    positions.
    """
    return trades.drop_duplicates(subset=["ticker", "signal_date", "direction"], keep="first")


def select_top_n_candidates(
    signal_counts_df: pd.DataFrame,
    score_by_key: dict[tuple[str, date], float],
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """signal_counts_df: ensemble.signal_count.aggregate_signal_counts()
    output. score_by_key: (ticker, date) -> total_score for that
    ticker's dominant-direction Signal that day (caller resolves ties/
    multiple scores before calling - e.g. by taking the max total_score
    among that direction's triggered Signals, matching Phase 5's Score
    being per (ticker,date,direction,signal_name) but Ensemble ranking
    needing one scalar per (ticker,date)).
    """
    candidates = signal_counts_df[signal_counts_df["dominant_direction"] != "NEUTRAL"].copy()
    candidates["score"] = [
        score_by_key.get((t, d), np.nan)
        for t, d in zip(candidates["ticker"], candidates["date"])
    ]
    candidates = candidates.sort_values(
        ["date", "total_signal_count", "score"], ascending=[True, False, False]
    )
    return candidates.groupby("date", group_keys=False).head(top_n)


@dataclass(frozen=True)
class EquityCurveMetrics:
    n_trades: int
    start_date: date | None
    end_date: date | None
    total_return: float | None
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    average_holding_days: float | None


def compute_equity_curve_metrics(
    selected_trades: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    notional_per_trade_fraction: float = PER_TRADE_NOTIONAL_FRACTION,
) -> EquityCurveMetrics:
    """selected_trades: trade rows (backtest/engine.py's TRADE_COLUMNS
    shape) for the candidates select_top_n_candidates() chose, already
    deduped via dedupe_trades_by_ticker_date_direction(). Marks P&L at
    each trade's exit_date only (see module docstring).
    """
    if selected_trades.empty:
        return EquityCurveMetrics(0, None, None, None, None, None, None, None, None, None, None)

    notional = initial_capital * notional_per_trade_fraction
    trades = selected_trades.assign(pnl=notional * selected_trades["return"])
    by_exit = trades.sort_values("exit_date")
    daily_pnl = by_exit.groupby("exit_date")["pnl"].sum()
    equity = initial_capital + daily_pnl.cumsum()

    start_date = by_exit["exit_date"].min()
    end_date = by_exit["exit_date"].max()
    n_calendar_days = (end_date - start_date).days
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity - initial_capital) / initial_capital

    cagr = None
    if n_calendar_days > 0 and final_equity > 0:
        cagr = (final_equity / initial_capital) ** (365.25 / n_calendar_days) - 1

    daily_returns = equity.pct_change().dropna()
    sharpe = None
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(
            daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    base_metrics = compute_metrics(trades)
    avg_holding_days = None
    holding = (by_exit["exit_date"] - by_exit["entry_date"]).apply(lambda td: td.days)
    if not holding.empty:
        avg_holding_days = float(holding.mean())

    return EquityCurveMetrics(
        n_trades=base_metrics.n_trades,
        start_date=start_date,
        end_date=end_date,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=base_metrics.win_rate,
        profit_factor=base_metrics.profit_factor,
        expectancy=base_metrics.expectancy,
        average_holding_days=avg_holding_days,
    )
