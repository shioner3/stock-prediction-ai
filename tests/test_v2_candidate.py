from __future__ import annotations

from datetime import date

import pandas as pd

from v2.candidate import (
    AVOID_PERCENTILE,
    CANDIDATE_PERCENTILE,
    build_candidate_table,
    classify_candidate,
    top_n,
)


def _scored_day(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date(2026, 1, 1)] * n,
            "ticker": [f"T{i}" for i in range(n)],
            "total_score": [float(i) for i in range(n)],
            "momentum_rank": [0.5] * n,
            "trend_rank": [0.5] * n,
            "volume_rank": [0.5] * n,
            "volatility_rank": [0.5] * n,
            "relative_strength_rank": [0.5] * n,
            "pullback_rank": [0.5] * n,
            "forward_return_5d": [0.01 * i for i in range(n)],
        }
    )


def test_classify_candidate_thresholds() -> None:
    assert classify_candidate(1.0) == "CANDIDATE"
    assert classify_candidate(CANDIDATE_PERCENTILE) == "CANDIDATE"
    assert classify_candidate(0.5) == "WATCH"
    assert classify_candidate(AVOID_PERCENTILE) == "AVOID"
    assert classify_candidate(0.0) == "AVOID"


def test_build_candidate_table_ranks_by_score_descending() -> None:
    day = _scored_day(10)
    records = build_candidate_table(day, ["momentum_rank", "trend_rank"])
    assert [r.ticker for r in records] == [f"T{i}" for i in range(9, -1, -1)]
    assert [r.rank for r in records] == list(range(1, 11))


def test_top_candidate_is_classified_candidate() -> None:
    day = _scored_day(10)
    records = build_candidate_table(day, [])
    assert records[0].classification == "CANDIDATE"
    assert records[-1].classification == "AVOID"


def test_top_n_slices_correctly() -> None:
    day = _scored_day(20)
    records = build_candidate_table(day, [])
    top5 = top_n(records, 5)
    assert len(top5) == 5
    assert [r.rank for r in top5] == [1, 2, 3, 4, 5]


def test_empty_day_returns_empty_list() -> None:
    empty = _scored_day(0)
    assert build_candidate_table(empty, []) == []


def test_rows_with_nan_total_score_are_excluded() -> None:
    day = _scored_day(5)
    day.loc[0, "total_score"] = float("nan")
    records = build_candidate_table(day, [])
    assert len(records) == 4
    assert all(r.ticker != "T0" for r in records)


def test_forward_returns_captured_when_present() -> None:
    day = _scored_day(3)
    records = build_candidate_table(day, [])
    assert records[0].forward_returns["forward_return_5d"] is not None


def test_market_lookup_applied() -> None:
    day = _scored_day(3)
    records = build_candidate_table(day, [], market_by_ticker={"T0": "Prime"})
    by_ticker = {r.ticker: r for r in records}
    assert by_ticker["T0"].market == "Prime"
    assert by_ticker["T1"].market is None
