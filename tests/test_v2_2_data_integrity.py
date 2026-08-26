from __future__ import annotations

from datetime import date
from pathlib import Path

from conftest import make_synthetic_ohlcv

from storage.parquet_store import save_ohlcv
from v2.validation.data_integrity import (
    has_critical_integrity_issues,
    run_data_integrity_preflight,
)


def test_preflight_reports_clean_data(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    tickers = [f"T{i}" for i in range(5)]
    for t in tickers:
        ohlcv = make_synthetic_ohlcv(50, seed=1, ticker=t)
        save_ohlcv(ohlcv, t, processed_dir)

    summary = run_data_integrity_preflight(
        tickers, processed_dir, date(2020, 1, 1), date(2020, 3, 1)
    )
    assert summary.n_tickers_checked == 5
    assert summary.n_tickers_missing == []
    assert summary.total_invalid_ohlc_rows == 0
    assert summary.total_negative_volume_rows == 0


def test_preflight_flags_missing_tickers(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    ohlcv = make_synthetic_ohlcv(50, seed=1, ticker="T0")
    save_ohlcv(ohlcv, "T0", processed_dir)

    summary = run_data_integrity_preflight(
        ["T0", "MISSING"], processed_dir, date(2020, 1, 1), date(2020, 3, 1)
    )
    assert summary.n_tickers_checked == 1
    assert summary.n_tickers_missing == ["MISSING"]


def test_has_critical_integrity_issues_true_when_majority_missing(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    ohlcv = make_synthetic_ohlcv(50, seed=1, ticker="T0")
    save_ohlcv(ohlcv, "T0", processed_dir)

    summary = run_data_integrity_preflight(
        ["T0", "M1", "M2", "M3"], processed_dir, date(2020, 1, 1), date(2020, 3, 1)
    )
    assert has_critical_integrity_issues(summary) is True


def test_has_critical_integrity_issues_false_when_healthy(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    tickers = [f"T{i}" for i in range(10)]
    for t in tickers:
        ohlcv = make_synthetic_ohlcv(50, seed=1, ticker=t)
        save_ohlcv(ohlcv, t, processed_dir)

    summary = run_data_integrity_preflight(
        tickers, processed_dir, date(2020, 1, 1), date(2020, 3, 1)
    )
    assert has_critical_integrity_issues(summary) is False


def test_has_critical_integrity_issues_true_when_zero_tickers() -> None:
    from v2.validation.data_integrity import DataIntegrityPreflightSummary

    summary = DataIntegrityPreflightSummary(
        n_tickers_checked=0, n_tickers_missing=[], total_rows=0,
        total_duplicate_dates=0, total_invalid_ohlc_rows=0, total_negative_volume_rows=0,
        total_zero_volume_rows=0, total_nan_rows=0, min_coverage_ratio=None,
        date_min=None, date_max=None, per_ticker=[],
    )
    assert has_critical_integrity_issues(summary) is True
