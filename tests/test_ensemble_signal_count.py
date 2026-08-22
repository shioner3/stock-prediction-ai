from __future__ import annotations

from datetime import date

import pandas as pd

from ensemble.signal_count import (
    aggregate_signal_counts,
    net_signal_count_bucket,
    signal_count_bucket,
)


def _records(rows: list[tuple[str, date, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "date", "signal_name", "direction"])


def test_signal_count_bucket() -> None:
    assert signal_count_bucket(1) == "1"
    assert signal_count_bucket(2) == "2"
    assert signal_count_bucket(3) == "3"
    assert signal_count_bucket(4) == "4+"
    assert signal_count_bucket(10) == "4+"


def test_net_signal_count_bucket() -> None:
    assert net_signal_count_bucket(-10) == "<=-4"
    assert net_signal_count_bucket(-4) == "<=-4"
    assert net_signal_count_bucket(-3) == "-3"
    assert net_signal_count_bucket(-1) == "-1"
    assert net_signal_count_bucket(0) == "0"
    assert net_signal_count_bucket(1) == "+1"
    assert net_signal_count_bucket(3) == "+3"
    assert net_signal_count_bucket(4) == ">=+4"
    assert net_signal_count_bucket(10) == ">=+4"


def test_aggregate_signal_counts_empty_input() -> None:
    out = aggregate_signal_counts(_records([]))
    assert out.empty
    assert list(out.columns) == [
        "ticker", "date", "long_count", "short_count", "total_signal_count",
        "net_signal_count", "dominant_direction", "direction_consensus",
        "long_signals", "short_signals",
    ]


def test_aggregate_signal_counts_pure_long_consensus() -> None:
    d = date(2026, 1, 5)
    records = _records(
        [
            ("7203", d, "long_pullback", "LONG"),
            ("7203", d, "long_ma_rebound", "LONG"),
            ("7203", d, "long_oversold_rebound", "LONG"),
        ]
    )
    out = aggregate_signal_counts(records)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["ticker"] == "7203"
    assert row["date"] == d
    assert row["long_count"] == 3
    assert row["short_count"] == 0
    assert row["total_signal_count"] == 3
    assert row["net_signal_count"] == 3
    assert row["dominant_direction"] == "LONG"
    assert row["direction_consensus"] == 1.0
    assert row["long_signals"] == ("long_ma_rebound", "long_oversold_rebound", "long_pullback")
    assert row["short_signals"] == ()


def test_aggregate_signal_counts_mixed_direction() -> None:
    d = date(2026, 1, 5)
    records = _records(
        [
            ("7203", d, "long_pullback", "LONG"),
            ("7203", d, "long_ma_rebound", "LONG"),
            ("7203", d, "short_breakdown", "SHORT"),
        ]
    )
    out = aggregate_signal_counts(records)
    row = out.iloc[0]
    assert row["long_count"] == 2
    assert row["short_count"] == 1
    assert row["net_signal_count"] == 1
    assert row["dominant_direction"] == "LONG"
    assert row["direction_consensus"] == 2 / 3


def test_aggregate_signal_counts_tie_is_neutral() -> None:
    d = date(2026, 1, 5)
    records = _records(
        [
            ("7203", d, "long_pullback", "LONG"),
            ("7203", d, "short_breakdown", "SHORT"),
        ]
    )
    out = aggregate_signal_counts(records)
    row = out.iloc[0]
    assert row["dominant_direction"] == "NEUTRAL"
    assert row["direction_consensus"] == 0.5


def test_aggregate_signal_counts_separates_tickers_and_dates() -> None:
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    records = _records(
        [
            ("7203", d1, "long_pullback", "LONG"),
            ("6758", d1, "short_breakdown", "SHORT"),
            ("7203", d2, "long_ma_rebound", "LONG"),
        ]
    )
    out = aggregate_signal_counts(records)
    assert len(out) == 3
    keys = set(zip(out["ticker"], out["date"]))
    assert keys == {("7203", d1), ("6758", d1), ("7203", d2)}
