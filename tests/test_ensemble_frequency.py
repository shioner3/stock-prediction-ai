from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ensemble.frequency import compute_frequency_metrics


def _dates(n: int) -> list[date]:
    return [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]


def test_frequency_metrics_basic() -> None:
    all_dates = pd.Series(_dates(10))
    # occurrences on day0(x2 tickers), day1(x1), day5(x1) -> 3 unique dates with occurrence
    occ_dates = pd.Series([all_dates[0], all_dates[0], all_dates[1], all_dates[5]])
    occ_tickers = pd.Series(["A", "B", "A", "C"])

    m = compute_frequency_metrics("test", occ_dates, occ_tickers, all_dates)

    assert m.total_occurrences == 4
    assert m.unique_tickers == 3
    assert m.unique_dates == 3
    assert m.total_trading_days_in_period == 10
    assert m.pct_trading_days_with_occurrence == 3 / 10


def test_frequency_metrics_avg_signals_per_day_includes_zero_days() -> None:
    all_dates = pd.Series(_dates(5))
    # occurrence only on day0, with 5 tickers
    occ_dates = pd.Series([all_dates[0]] * 5)
    occ_tickers = pd.Series([f"T{i}" for i in range(5)])

    m = compute_frequency_metrics("test", occ_dates, occ_tickers, all_dates)
    # 5 occurrences on day0, 0 on days1-4 -> mean = 5/5 = 1.0
    assert m.avg_signals_per_day == 1.0
    assert m.median_signals_per_day == 0.0


def test_frequency_metrics_no_occurrences() -> None:
    all_dates = pd.Series(_dates(10))
    empty = pd.Series([], dtype=object)
    m = compute_frequency_metrics("test", empty, empty, all_dates)
    assert m.total_occurrences == 0
    assert m.pct_trading_days_with_occurrence == 0.0
    assert m.avg_signals_per_day == 0.0
