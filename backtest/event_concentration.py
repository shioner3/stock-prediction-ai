"""Phase 9 section 8: how much of a Signal's total P&L and trade count is
concentrated in a small number of individual TRADING DAYS - generalizes
Phase 8's "one 9-day episode = 71.6% of BEAR P&L" finding into a
systematic top-N-day breakdown. This is DAY-level concentration, distinct
from backtest/cross_sectional.py's TICKER-level concentration and
backtest/episode_analysis.py's contiguous-regime-episode grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


@dataclass
class DayConcentrationMetrics:
    n_days: int
    n_trades: int
    # Cumulative share of TOTAL RETURN contributed by the top-K days,
    # ranked by each day's own summed return, descending. Can exceed 1.0
    # or be negative for the same reason backtest/cross_sectional.py's
    # return shares can - not clipped, see that module's docstring.
    pnl_share_by_k: dict[int, float | None]
    # Cumulative share of TOTAL TRADE COUNT contributed by the top-K days.
    trade_share_by_k: dict[int, float | None]
    # Gini coefficient of per-day P&L (mean-absolute-difference form,
    # which stays well-defined for the signed/negative daily P&L values
    # trading days can have - NOT the classic non-negative-income Gini,
    # so treat this as "day-level inequality, roughly 0=even 1=very
    # concentrated" rather than a textbook-calibrated Gini value). None
    # when fewer than 2 days or when total P&L nets to exactly zero.
    gini_coefficient: float | None


def _gini_of_signed_values(values: np.ndarray) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean_value = float(np.mean(values))
    if mean_value == 0:
        return None
    mean_abs_diff = float(np.abs(values[:, None] - values[None, :]).mean())
    return mean_abs_diff / (2 * abs(mean_value))


def compute_day_concentration(
    trades: pd.DataFrame,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    date_col: str = "signal_date",
) -> DayConcentrationMetrics:
    n_trades = len(trades)
    if n_trades == 0:
        return DayConcentrationMetrics(
            n_days=0, n_trades=0,
            pnl_share_by_k={k: None for k in k_values},
            trade_share_by_k={k: None for k in k_values},
            gini_coefficient=None,
        )

    by_day_return = trades.groupby(date_col)["return"].sum().sort_values(ascending=False)
    by_day_count = trades.groupby(date_col).size().sort_values(ascending=False)
    total_return = float(by_day_return.sum())
    n_days = len(by_day_return)

    pnl_share: dict[int, float | None] = {}
    trade_share: dict[int, float | None] = {}
    for k in k_values:
        pnl_share[k] = (
            float(by_day_return.iloc[:k].sum() / total_return) if total_return != 0 else None
        )
        trade_share[k] = float(by_day_count.iloc[:k].sum() / n_trades)

    return DayConcentrationMetrics(
        n_days=n_days,
        n_trades=n_trades,
        pnl_share_by_k=pnl_share,
        trade_share_by_k=trade_share,
        gini_coefficient=_gini_of_signed_values(by_day_return.to_numpy(dtype=float)),
    )
