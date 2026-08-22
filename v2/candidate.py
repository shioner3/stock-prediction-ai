"""Candidate Ranking (spec section 12/13).

Turns one day's Universe-wide V2 Score into a ranked table. NOT a buy/
sell decision, NOT a probability, NOT investment advice - spec section
9/25 are explicit that a high V2 Score never means "future return is
guaranteed higher", only "ranked more favorably by this fixed, equal-
weight rule set on this one day".

classify_candidate() (spec section 13) is a DELIBERATELY MECHANICAL,
score-percentile-only placeholder - CANDIDATE = today's top
CANDIDATE_PERCENTILE, AVOID = today's bottom AVOID_PERCENTILE, WATCH =
everything between. No other signal, threshold, or judgment feeds into
it. This exists so the schema/design can carry a classification field
into later phases (spec section 13's own "分類を追加可能な設計にして
おく"), not because V2-1 claims this split is meaningful for real
decisions - no "should I buy" logic is implemented here or anywhere in
V2-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import pandas as pd

CANDIDATE_PERCENTILE = 0.80  # score percentile >= this -> "CANDIDATE"
AVOID_PERCENTILE = 0.20  # score percentile <= this -> "AVOID"

FORWARD_RETURN_COLUMNS = [
    "forward_return_5d", "forward_return_10d", "forward_return_15d", "forward_return_20d",
]


@dataclass(frozen=True)
class CandidateRecord:
    date: date_type
    ticker: str
    score: float
    rank: int  # 1-based, 1 = highest score that day
    score_percentile: float
    classification: str  # "CANDIDATE" / "WATCH" / "AVOID" - see module docstring
    market: str | None
    category_ranks: dict[str, float]
    forward_returns: dict[str, float | None]  # forward_return_{5,10,15,20}d, may be unresolved


def classify_candidate(score_percentile: float) -> str:
    if score_percentile >= CANDIDATE_PERCENTILE:
        return "CANDIDATE"
    if score_percentile <= AVOID_PERCENTILE:
        return "AVOID"
    return "WATCH"


def build_candidate_table(
    scored_day: pd.DataFrame,
    category_rank_columns: list[str],
    market_by_ticker: dict[str, str] | None = None,
) -> list[CandidateRecord]:
    """scored_day: ONE date's rows only (ticker, total_score, the
    category_rank_columns, and whichever forward_return_{n}d columns are
    present - unresolved windows are simply absent/NaN, never dropped
    from the table). Ranked descending by total_score; ties keep pandas'
    default stable rank ordering (average method for the percentile,
    first-seen order for the integer rank - deterministic given the
    same input row order, matching v2/tests's determinism requirement).
    """
    if scored_day.empty:
        return []
    df = scored_day.dropna(subset=["total_score"]).copy()
    if df.empty:
        return []

    df["score_percentile"] = df["total_score"].rank(pct=True, ascending=True)
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    market_by_ticker = market_by_ticker or {}
    records = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        category_ranks = {col: row_dict.get(col) for col in category_rank_columns}
        forward_returns = {
            col: row_dict.get(col) for col in FORWARD_RETURN_COLUMNS if col in row_dict
        }
        records.append(
            CandidateRecord(
                date=row_dict["date"],
                ticker=row_dict["ticker"],
                score=row_dict["total_score"],
                rank=row_dict["rank"],
                score_percentile=row_dict["score_percentile"],
                classification=classify_candidate(row_dict["score_percentile"]),
                market=market_by_ticker.get(row_dict["ticker"]),
                category_ranks=category_ranks,
                forward_returns=forward_returns,
            )
        )
    return records


def top_n(records: list[CandidateRecord], n: int) -> list[CandidateRecord]:
    return sorted(records, key=lambda r: r.rank)[:n]
