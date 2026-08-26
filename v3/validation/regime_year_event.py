"""Regime / Year / Event breakdown of pooled OOS predictions (spec
section 16/17/18). Reuses `backtest.market_regime.compute_market_regime()`
(V1, unmodified, same TOPIX-trailing-return definition every prior Phase
uses - no new Regime definition), `pipeline.run_phase9_analysis.YEARS`/
`AUG_2024_EVENT_START`/`AUG_2024_EVENT_END` (V1, unmodified - the same
2024-08 window every prior Phase's event analysis already uses), and
`scoring.validation.assign_quantile_buckets()` + `v2.stats.
compute_quantile_bucket_stats()`/`compute_q5_q1_spread()` (V1/V2,
unmodified) for the actual bucket statistics within each slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.run_phase9_analysis import AUG_2024_EVENT_END, AUG_2024_EVENT_START, YEARS
from scoring.validation import assign_quantile_buckets
from v2.stats import QuantileBucketStats, compute_q5_q1_spread, compute_quantile_bucket_stats

REGIMES = ("BULL", "NEUTRAL", "BEAR")


@dataclass(frozen=True)
class SliceResult:
    label: str
    n: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None


def _slice_result(
    label: str, df: pd.DataFrame, window_days: int, prediction_col: str, actual_col: str
) -> SliceResult:
    valid = df.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    bucket_stats = compute_quantile_bucket_stats(valid, "_bucket", actual_col, window_days)
    return SliceResult(
        label=label, n=len(valid), bucket_stats=bucket_stats,
        q5_q1_spread=compute_q5_q1_spread(bucket_stats),
    )


def analyze_by_regime(
    predictions: pd.DataFrame,
    regime_df: pd.DataFrame,
    window_days: int,
    prediction_col: str = "prediction",
    actual_col: str = "actual",
    date_col: str = "date",
) -> list[SliceResult]:
    merged = predictions.merge(regime_df, on=date_col, how="left")
    return [
        _slice_result(
            regime, merged[merged["regime"] == regime], window_days, prediction_col, actual_col
        )
        for regime in REGIMES
    ]


def analyze_by_year(
    predictions: pd.DataFrame,
    window_days: int,
    prediction_col: str = "prediction",
    actual_col: str = "actual",
    date_col: str = "date",
) -> dict[int, SliceResult]:
    results: dict[int, SliceResult] = {}
    for year in YEARS:
        year_df = predictions[predictions[date_col].apply(lambda d: d.year) == year]
        if year_df.empty:
            continue
        results[year] = _slice_result(str(year), year_df, window_days, prediction_col, actual_col)
    return results


def analyze_event_exclusion(
    predictions: pd.DataFrame,
    window_days: int,
    prediction_col: str = "prediction",
    actual_col: str = "actual",
    date_col: str = "date",
) -> dict[str, SliceResult]:
    full = _slice_result("full_period", predictions, window_days, prediction_col, actual_col)
    aug_mask = (
        (predictions[date_col] >= AUG_2024_EVENT_START)
        & (predictions[date_col] <= AUG_2024_EVENT_END)
    )
    excl_aug2024 = _slice_result(
        "excl_2024_08", predictions[~aug_mask], window_days, prediction_col, actual_col
    )
    year_2024_mask = predictions[date_col].apply(lambda d: d.year) == 2024
    excl_2024_full = _slice_result(
        "excl_2024_full_year", predictions[~year_2024_mask], window_days, prediction_col,
        actual_col,
    )
    return {
        "full_period": full, "excl_2024_08": excl_aug2024, "excl_2024_full_year": excl_2024_full,
    }
