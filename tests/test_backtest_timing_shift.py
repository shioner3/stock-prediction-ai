from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from backtest.timing_shift import shift_signal_records


def _panel(n: int) -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({"date": dates, "close": [100.0 + i for i in range(n)]})


def _signal_records(rows: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["T"] * len(rows),
            "date": rows,
            "signal_name": ["long_oversold_rebound"] * len(rows),
            "direction": ["LONG"] * len(rows),
        }
    )


def test_negative_offset_shifts_earlier() -> None:
    panel = _panel(20)
    records = _signal_records([panel["date"].iloc[10]])
    shifted = shift_signal_records(records, panel, offset_days=-3)
    assert len(shifted) == 1
    assert shifted["date"].iloc[0] == panel["date"].iloc[7]


def test_positive_offset_shifts_later() -> None:
    panel = _panel(20)
    records = _signal_records([panel["date"].iloc[10]])
    shifted = shift_signal_records(records, panel, offset_days=5)
    assert len(shifted) == 1
    assert shifted["date"].iloc[0] == panel["date"].iloc[15]


def test_zero_offset_is_identity() -> None:
    panel = _panel(20)
    records = _signal_records([panel["date"].iloc[10]])
    shifted = shift_signal_records(records, panel, offset_days=0)
    assert shifted["date"].iloc[0] == panel["date"].iloc[10]


def test_row_dropped_when_shift_goes_before_start() -> None:
    panel = _panel(20)
    records = _signal_records([panel["date"].iloc[2]])
    shifted = shift_signal_records(records, panel, offset_days=-5)
    assert shifted.empty


def test_row_dropped_when_shift_goes_past_end() -> None:
    panel = _panel(20)
    records = _signal_records([panel["date"].iloc[18]])
    shifted = shift_signal_records(records, panel, offset_days=5)
    assert shifted.empty


def test_never_produces_a_date_outside_the_panel() -> None:
    panel = _panel(30)
    records = _signal_records(list(panel["date"]))
    for offset in (-15, -10, -5, -3, -1, 5, 10):
        shifted = shift_signal_records(records, panel, offset_days=offset)
        assert set(shifted["date"]).issubset(set(panel["date"]))


def test_row_with_date_not_in_panel_is_dropped() -> None:
    panel = _panel(20)
    foreign_date = date(2099, 1, 1)
    records = _signal_records([foreign_date])
    shifted = shift_signal_records(records, panel, offset_days=-1)
    assert shifted.empty


def test_preserves_non_date_columns() -> None:
    panel = _panel(20)
    records = _signal_records([panel["date"].iloc[10]])
    shifted = shift_signal_records(records, panel, offset_days=-2)
    assert shifted["ticker"].iloc[0] == "T"
    assert shifted["signal_name"].iloc[0] == "long_oversold_rebound"
    assert shifted["direction"].iloc[0] == "LONG"
    assert list(shifted.columns) == list(records.columns)
