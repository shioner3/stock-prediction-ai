"""Event exclusion and Year-by-Year analysis (spec section 19/20).

Reuses pipeline.run_phase9_analysis.AUG_2024_EVENT_START/END and YEARS
(unmodified V1 constants - the exact same 2024-08 window and year list
every other Phase's event/year analysis already uses) rather than
picking a new window after seeing V2-2's own results (spec section 19's
explicit prohibition: "V1の結果を見て都合よくイベントを追加しては
いけない").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.run_phase9_analysis import AUG_2024_EVENT_END, AUG_2024_EVENT_START, YEARS
from v2.stats import (
    QuantileBucketStats,
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
)


@dataclass(frozen=True)
class SliceResult:
    label: str
    n: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None


def _slice_result(label: str, df: pd.DataFrame, return_col: str, window_days: int) -> SliceResult:
    window_df = df.dropna(subset=[return_col])
    bucket_stats = compute_quantile_bucket_stats(window_df, "score_bucket", return_col, window_days)
    return SliceResult(
        label=label, n=len(window_df), bucket_stats=bucket_stats,
        q5_q1_spread=compute_q5_q1_spread(bucket_stats),
    )


def analyze_event_exclusion(
    scored: pd.DataFrame, return_col: str, window_days: int, date_col: str = "date",
) -> dict[str, SliceResult]:
    """spec section 19: full period vs 2024-08 excluded vs (if a
    dominant single day exists) that day excluded.
    """
    full = _slice_result("full_period", scored, return_col, window_days)

    aug_mask = (scored[date_col] >= AUG_2024_EVENT_START) & (scored[date_col] <= AUG_2024_EVENT_END)
    excl_aug2024 = _slice_result("excl_2024_08", scored[~aug_mask], return_col, window_days)

    max_day_excl = analyze_max_contribution_day_exclusion(
        scored, return_col, window_days, date_col
    )

    return {
        "full_period": full,
        "excl_2024_08": excl_aug2024,
        "excl_max_contribution_day": max_day_excl,
    }


def find_max_contribution_day(
    scored: pd.DataFrame, return_col: str, date_col: str = "date", bucket: str = "Q5",
) -> object | None:
    """The single calendar day contributing the most to `bucket`'s
    (default Q5) TOTAL cumulative Forward Return - V2's analogue of
    Phase 9's "single day P&L contribution" concentration check, applied
    to a Score bucket's cumulative return series rather than a Trade's.
    """
    bucket_rows = scored[scored["score_bucket"] == bucket].dropna(subset=[return_col])
    if bucket_rows.empty:
        return None
    daily_sum = bucket_rows.groupby(date_col)[return_col].sum()
    return daily_sum.idxmax()


def analyze_max_contribution_day_exclusion(
    scored: pd.DataFrame, return_col: str, window_days: int, date_col: str = "date",
) -> SliceResult:
    max_day = find_max_contribution_day(scored, return_col, date_col)
    if max_day is None:
        return SliceResult("excl_max_contribution_day", 0, [], None)
    remaining = scored[scored[date_col] != max_day]
    result = _slice_result("excl_max_contribution_day", remaining, return_col, window_days)
    return SliceResult(
        label=f"excl_max_contribution_day({max_day})", n=result.n,
        bucket_stats=result.bucket_stats, q5_q1_spread=result.q5_q1_spread,
    )


def analyze_year_by_year(
    scored: pd.DataFrame, return_col: str, window_days: int, date_col: str = "date",
) -> dict[int, SliceResult]:
    results = {}
    for year in YEARS:
        year_df = scored[scored[date_col].apply(lambda d: d.year) == year]
        if year_df.empty:
            continue
        results[year] = _slice_result(str(year), year_df, return_col, window_days)
    return results
