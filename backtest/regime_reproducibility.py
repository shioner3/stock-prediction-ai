"""Phase 8 section 5 (B1/B2): does a Signal's Market Regime-conditioned
edge hold up across MULTIPLE independent slices of time - individual WFO
Windows, individual calendar years - or is "BEAR regime PF is high"
really just one aggregate number hiding a single lucky stretch?

Reuses backtest/market_regime.py's regime classification and
backtest/walk_forward.py's WalkForwardWindow OOS boundaries unchanged;
this module only slices an already-computed Trade Record DataFrame by
(window, regime) or (year, regime) and reports descriptive metrics per
slice - it does not alter regime thresholds, WFO windows, or trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.walk_forward import WalkForwardWindow


def _merge_regime(trades: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    return trades.merge(regime_df, left_on="signal_date", right_on="date", how="left")


@dataclass
class RegimeWindowMetrics:
    window_index: int
    regime: str
    metrics: BacktestMetrics


def compute_regime_by_window(
    trades: pd.DataFrame, regime_df: pd.DataFrame, windows: list[WalkForwardWindow], regime: str
) -> list[RegimeWindowMetrics]:
    """One RegimeWindowMetrics per window, restricted to trades whose
    signal_date falls in that window's OOS period AND is classified as
    `regime`. A window with zero matching trades still gets an entry
    (BacktestMetrics.n_trades=0) - windows are never silently dropped,
    so "no BEAR trades in this window" is visible, not absent.
    """
    merged = _merge_regime(trades, regime_df)
    results = []
    for w in windows:
        oos_subset = merged[
            (merged["signal_date"] >= w.oos_start) & (merged["signal_date"] <= w.oos_end)
        ]
        regime_subset = oos_subset[oos_subset["regime"] == regime]
        results.append(RegimeWindowMetrics(w.index, regime, compute_metrics(regime_subset)))
    return results


@dataclass
class RegimeYearMetrics:
    year: int
    regime: str
    metrics: BacktestMetrics | None  # None means NO_BEAR_DATA (or no data for this regime/year)


def compute_regime_by_year(
    trades: pd.DataFrame, regime_df: pd.DataFrame, regime: str, years: list[int]
) -> dict[int, RegimeYearMetrics]:
    """One entry per requested calendar year. A year with zero trading
    days classified as `regime` (spec section 5B2's NO_BEAR_DATA case)
    gets metrics=None rather than a zero-trade BacktestMetrics, so
    "regime never occurred this year" (data absence) stays distinguishable
    from "regime occurred but the Signal never triggered" (zero trades).
    """
    merged = _merge_regime(trades, regime_df)
    regime_trades = merged[merged["regime"] == regime]
    regime_dates_by_year = {
        year: set(d for d in regime_df.loc[regime_df["regime"] == regime, "date"] if d.year == year)
        for year in years
    }

    result: dict[int, RegimeYearMetrics] = {}
    for year in years:
        if not regime_dates_by_year[year]:
            result[year] = RegimeYearMetrics(year, regime, None)
            continue
        year_trades = regime_trades[regime_trades["signal_date"].apply(lambda d: d.year) == year]
        result[year] = RegimeYearMetrics(year, regime, compute_metrics(year_trades))
    return result
