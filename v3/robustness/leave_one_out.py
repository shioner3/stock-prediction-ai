"""Spec sections 6/7/8/9/10: Regime / Event(day) / Year / Stock leave-one-
out robustness. All four follow the same shape - identify a group (regime
label, calendar year, trading day, ticker), evaluate performance WITH and
WITHOUT that group, and (for day/ticker) rank contributors by Gini/Top-K
share to find out how concentrated Q5's edge is. Regime/Year definitions
are reused UNMODIFIED (`backtest.market_regime.compute_market_regime`,
`pipeline.run_phase9_analysis.YEARS`) - only the exclusion/ranking logic
is new.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from pipeline.run_phase9_analysis import YEARS
from scoring.validation import assign_quantile_buckets
from v3.robustness.common_eval import SliceEvaluation, evaluate_slice
from v3.robustness.gini import compute_contribution_ranking, gini_coefficient

REGIMES = ("BULL", "NEUTRAL", "BEAR")
TOP_K_VALUES = (1, 5, 10, 20)


@dataclass(frozen=True)
class RegimeRobustness:
    breakdown: dict[str, SliceEvaluation]
    leave_one_out: dict[str, SliceEvaluation]
    all_regimes: SliceEvaluation
    regime_dependent: bool  # True if excluding BEAR alone drives Q5-Q1 spread <= 0


def run_regime_robustness(
    predictions: pd.DataFrame, regime_df: pd.DataFrame, window_days: int
) -> RegimeRobustness:
    merged = predictions.merge(regime_df, on="date", how="left")
    breakdown = {
        r: evaluate_slice(merged[merged["regime"] == r], r, window_days) for r in REGIMES
    }
    leave_one_out = {
        f"excl_{r}": evaluate_slice(merged[merged["regime"] != r], f"excl_{r}", window_days)
        for r in REGIMES
    }
    all_regimes = evaluate_slice(merged, "all_regimes", window_days)
    excl_bear_spread = leave_one_out["excl_BEAR"].ranking.q5_q1_spread
    regime_dependent = excl_bear_spread is not None and excl_bear_spread <= 0
    return RegimeRobustness(
        breakdown=breakdown, leave_one_out=leave_one_out, all_regimes=all_regimes,
        regime_dependent=regime_dependent,
    )


@dataclass(frozen=True)
class YearRobustness:
    breakdown: dict[int, SliceEvaluation]
    leave_one_out: dict[int, SliceEvaluation]


def run_year_robustness(predictions: pd.DataFrame, window_days: int) -> YearRobustness:
    years_present = sorted({d.year for d in predictions["date"]} & set(YEARS))
    breakdown = {}
    leave_one_out = {}
    for year in years_present:
        year_mask = predictions["date"].apply(lambda d: d.year) == year
        year_df = predictions[year_mask]
        if not year_df.empty:
            breakdown[year] = evaluate_slice(year_df, str(year), window_days)
        excl_df = predictions[~year_mask]
        leave_one_out[year] = evaluate_slice(excl_df, f"excl_{year}", window_days)
    return YearRobustness(breakdown=breakdown, leave_one_out=leave_one_out)


@dataclass(frozen=True)
class DayConcentrationRobustness:
    n_unique_q5_days: int
    gini_day_contribution: float | None
    top_k_exclusion: dict[str, SliceEvaluation]  # "top1", "top5", "top10", "top20", "top1pct"
    top_1pct_n_days: int


def run_day_concentration_robustness(
    predictions: pd.DataFrame, window_days: int, prediction_col: str = "prediction",
    actual_col: str = "actual",
) -> DayConcentrationRobustness:
    valid = predictions.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    q5 = valid[valid["_bucket"] == "Q5"]
    day_contribution = compute_contribution_ranking(q5, "date", actual_col)
    gini = gini_coefficient(day_contribution.to_numpy())

    n_unique_days = valid["date"].nunique()
    top_1pct_n = max(1, math.ceil(n_unique_days * 0.01))

    top_k_exclusion: dict[str, SliceEvaluation] = {}
    for k in TOP_K_VALUES:
        excluded_days = set(day_contribution.head(k).index)
        remainder = valid[~valid["date"].isin(excluded_days)]
        top_k_exclusion[f"top{k}"] = evaluate_slice(
            remainder, f"excl_top{k}_days", window_days, prediction_col, actual_col
        )
    excluded_1pct = set(day_contribution.head(top_1pct_n).index)
    top_k_exclusion["top1pct"] = evaluate_slice(
        valid[~valid["date"].isin(excluded_1pct)], "excl_top1pct_days", window_days,
        prediction_col, actual_col,
    )

    return DayConcentrationRobustness(
        n_unique_q5_days=len(day_contribution), gini_day_contribution=gini,
        top_k_exclusion=top_k_exclusion, top_1pct_n_days=top_1pct_n,
    )


@dataclass(frozen=True)
class StockConcentrationRobustness:
    n_unique_q5_tickers: int
    gini_ticker_contribution: float | None
    top_k_exclusion: dict[str, SliceEvaluation]  # tickers removed ACROSS ALL buckets


def run_stock_concentration_robustness(
    predictions: pd.DataFrame, window_days: int, prediction_col: str = "prediction",
    actual_col: str = "actual",
) -> StockConcentrationRobustness:
    valid = predictions.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    q5 = valid[valid["_bucket"] == "Q5"]
    ticker_contribution = compute_contribution_ranking(q5, "ticker", actual_col)
    gini = gini_coefficient(ticker_contribution.to_numpy())

    top_k_exclusion: dict[str, SliceEvaluation] = {}
    for k in TOP_K_VALUES:
        excluded_tickers = set(ticker_contribution.head(k).index)
        # top contributors are removed from the WHOLE dataset (every
        # bucket, not just Q5) - the question is whether the model's
        # OVERALL edge depends on a handful of tickers, not just Q5's.
        remainder = valid[~valid["ticker"].isin(excluded_tickers)]
        top_k_exclusion[f"top{k}"] = evaluate_slice(
            remainder, f"excl_top{k}_stocks", window_days, prediction_col, actual_col
        )

    return StockConcentrationRobustness(
        n_unique_q5_tickers=len(ticker_contribution), gini_ticker_contribution=gini,
        top_k_exclusion=top_k_exclusion,
    )
