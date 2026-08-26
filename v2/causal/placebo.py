"""Timing Placebo Test (spec section 26): does Q1 membership predict
Forward Return ONLY at its true (lag-0) timing, or does an artificially
shifted Q1 flag "predict" a return just as well - which would indicate
the effect is really about persistently mispriced TICKERS rather than
about the Score's timing?

For lag k > 0: `score_bucket` shifted k trading days BACKWARD in a
ticker's own row order is compared against TODAY's Forward Return (was
this ticker in Q1 k trading days ago?). For lag k < 0: `score_bucket`
shifted forward (was this ticker ABOUT TO enter Q1 in |k| trading days?)
is compared against today's return - a "leading" placebo. Both directions
are pure placebos (only lag 0 is the real, already-measured relationship);
neither uses any V1 Phase 8/9 placebo result as its criterion (spec
section 26's explicit prohibition on reusing V1's own placebo threshold).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from v2.stats import ReturnStats, compute_return_stats

PLACEBO_SHIFTS = (-15, -10, -5, 5, 10, 15)


@dataclass(frozen=True)
class PlaceboLagResult:
    lag_days: int
    n: int
    stats: ReturnStats


def run_timing_placebo(
    scored: pd.DataFrame,
    return_col: str,
    shifts: tuple[int, ...] = PLACEBO_SHIFTS,
    bucket_col: str = "score_bucket",
    bucket_label: str = "Q1",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> list[PlaceboLagResult]:
    df = scored[[ticker_col, date_col, bucket_col, return_col]].dropna(subset=[return_col]).copy()
    df = df.sort_values([ticker_col, date_col])

    real = df[df[bucket_col] == bucket_label]
    results = [PlaceboLagResult(0, len(real), compute_return_stats(real[return_col]))]

    for lag in shifts:
        shifted_bucket = df.groupby(ticker_col, group_keys=False)[bucket_col].shift(lag)
        placebo_subset = df.loc[shifted_bucket == bucket_label]
        results.append(
            PlaceboLagResult(
                lag, len(placebo_subset), compute_return_stats(placebo_subset[return_col])
            )
        )
    return results
