"""Concentration analysis (spec section 21): does Q5's (the "long
candidate" bucket's) aggregate performance depend on a small number of
tickers or days, rather than being broadly distributed across the
Universe? Mirrors the "contribution share" concept Phase 8/9 already
established for Trade Records (README's "BEAR regime concentration"
findings), applied here to a Score bucket's cumulative Forward Return
instead of a Trade's cumulative return - a new, small computation (no
existing V1 primitive computes ticker/day contribution shares in a
reusable, generic form), built directly on pandas groupby/cumsum.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TOP_K_VALUES = (1, 5, 10, 20)


@dataclass(frozen=True)
class ContributionShare:
    top_k: int
    share_of_total: float | None  # sum of the top-K contributors' cumulative return / grand total


@dataclass(frozen=True)
class ConcentrationResult:
    bucket: str
    window_days: int
    ticker_contribution: list[ContributionShare]
    day_contribution: list[ContributionShare]


def _top_k_shares(
    contributions: pd.Series, top_k_values: tuple[int, ...]
) -> list[ContributionShare]:
    total = contributions.sum()
    sorted_desc = contributions.sort_values(ascending=False)
    results = []
    for k in top_k_values:
        top_sum = sorted_desc.head(k).sum()
        share = float(top_sum / total) if total != 0 else None
        results.append(ContributionShare(top_k=k, share_of_total=share))
    return results


def compute_concentration(
    scored: pd.DataFrame, return_col: str, window_days: int, bucket: str = "Q5",
    bucket_col: str = "score_bucket", date_col: str = "date",
) -> ConcentrationResult:
    rows = scored[scored[bucket_col] == bucket].dropna(subset=[return_col])
    ticker_contribution = rows.groupby("ticker")[return_col].sum()
    day_contribution = rows.groupby(date_col)[return_col].sum()
    return ConcentrationResult(
        bucket=bucket, window_days=window_days,
        ticker_contribution=_top_k_shares(ticker_contribution, TOP_K_VALUES),
        day_contribution=_top_k_shares(day_contribution, TOP_K_VALUES),
    )
