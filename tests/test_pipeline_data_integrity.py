from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pipeline.data_integrity import build_data_integrity_report, compute_ticker_coverage
from storage.parquet_store import save_ohlcv


def _ohlcv(
    ticker: str, n_days: int = 5, start: date = date(2024, 1, 1), volume: int = 1000
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days).date
    return pd.DataFrame(
        {
            "ticker": [ticker] * n_days,
            "date": dates,
            "open": [100.0] * n_days,
            "high": [105.0] * n_days,
            "low": [95.0] * n_days,
            "close": [100.0] * n_days,
            "volume": [volume] * n_days,
        }
    )


# --- compute_ticker_coverage --------------------------------------------------


def test_compute_ticker_coverage_clean_data() -> None:
    df = _ohlcv("7203", n_days=5, start=date(2024, 1, 1))
    start, end = date(2024, 1, 1), date(2024, 1, 5)
    coverage = compute_ticker_coverage(df, "7203", start, end)

    assert coverage.row_count == 5
    assert coverage.date_min == df["date"].min()
    assert coverage.date_max == df["date"].max()
    assert coverage.duplicate_dates == 0
    assert coverage.invalid_ohlc_rows == 0
    assert coverage.negative_volume_rows == 0
    assert coverage.zero_volume_rows == 0
    assert coverage.nan_rows == 0
    assert coverage.coverage_ratio == pytest.approx(1.0)


def test_compute_ticker_coverage_empty_dataframe() -> None:
    coverage = compute_ticker_coverage(
        pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"]),
        "9999", date(2024, 1, 1), date(2024, 1, 5),
    )
    assert coverage.row_count == 0
    assert coverage.date_min is None
    assert coverage.coverage_ratio == 0.0


def test_compute_ticker_coverage_detects_duplicate_dates() -> None:
    df = _ohlcv("7203", n_days=3)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate the first date
    coverage = compute_ticker_coverage(df, "7203", date(2024, 1, 1), date(2024, 1, 5))
    assert coverage.duplicate_dates == 1


def test_compute_ticker_coverage_detects_invalid_ohlc() -> None:
    df = _ohlcv("7203", n_days=3)
    df.loc[0, "close"] = -5.0  # non-positive price
    df.loc[1, "high"] = 1.0
    df.loc[1, "low"] = 50.0  # high < low
    coverage = compute_ticker_coverage(df, "7203", date(2024, 1, 1), date(2024, 1, 5))
    assert coverage.invalid_ohlc_rows == 2


def test_compute_ticker_coverage_detects_negative_and_zero_volume() -> None:
    df = _ohlcv("7203", n_days=3)
    df.loc[0, "volume"] = -10
    df.loc[1, "volume"] = 0
    coverage = compute_ticker_coverage(df, "7203", date(2024, 1, 1), date(2024, 1, 5))
    assert coverage.negative_volume_rows == 1
    assert coverage.zero_volume_rows == 1  # only the explicit ==0 row, not the negative one


def test_compute_ticker_coverage_detects_nan_rows() -> None:
    df = _ohlcv("7203", n_days=3)
    df.loc[0, "close"] = float("nan")
    coverage = compute_ticker_coverage(df, "7203", date(2024, 1, 1), date(2024, 1, 5))
    assert coverage.nan_rows == 1


def test_compute_ticker_coverage_ratio_reflects_missing_days() -> None:
    df = _ohlcv("7203", n_days=2, start=date(2024, 1, 1))
    # Requested a 5-business-day window but only 2 rows were fetched.
    coverage = compute_ticker_coverage(df, "7203", date(2024, 1, 1), date(2024, 1, 5))
    assert coverage.expected_business_days == 5
    assert coverage.coverage_ratio == pytest.approx(2 / 5)


# --- build_data_integrity_report ----------------------------------------------


def _write_manifest(path: Path, tickers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"generated_at": "2026-08-20T00:00:00", "tickers": tickers}),
        encoding="utf-8",
    )


def test_build_data_integrity_report_funnel_counts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"

    save_ohlcv(_ohlcv("7203"), "7203", raw_dir)
    save_ohlcv(_ohlcv("6758"), "6758", raw_dir)
    save_ohlcv(_ohlcv("1332"), "1332", raw_dir)  # fetched OK but excluded by liquidity

    _write_manifest(
        manifest_path,
        {
            "7203": {"status": "success", "included_in_universe": True},
            "6758": {"status": "partial", "included_in_universe": True},
            "1332": {"status": "success", "included_in_universe": False},
            "9999": {"status": "failed", "included_in_universe": False},
        },
    )

    report = build_data_integrity_report(
        jpx_master_candidates=4444,
        static_filter_included=3713,
        static_filter_excluded=731,
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
    )

    f = report.funnel
    assert f.jpx_master_candidates == 4444
    assert f.static_filter_included == 3713
    assert f.static_filter_excluded == 731
    assert f.fetch_attempted == 4
    assert f.fetch_success == 2
    assert f.fetch_partial == 1
    assert f.fetch_failed == 1
    assert f.price_liquidity_excluded == 1  # 1332: fetched OK, not in final universe
    assert f.final_universe == 2

    assert {c.ticker for c in report.ticker_coverage} == {"7203", "6758", "1332"}


def test_build_data_integrity_report_skips_failed_tickers_for_coverage(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"
    save_ohlcv(_ohlcv("7203"), "7203", raw_dir)

    _write_manifest(
        manifest_path,
        {
            "7203": {"status": "success", "included_in_universe": True},
            # "9999" has no raw file on disk - a failed fetch never wrote one.
            "9999": {"status": "failed", "included_in_universe": False},
        },
    )

    report = build_data_integrity_report(
        jpx_master_candidates=10, static_filter_included=10, static_filter_excluded=0,
        manifest_path=manifest_path, raw_dir=raw_dir,
        start=date(2024, 1, 1), end=date(2024, 1, 5),
    )

    # Would raise FileNotFoundError if it tried to load "9999"'s (nonexistent) raw file.
    assert [c.ticker for c in report.ticker_coverage] == ["7203"]


def test_build_data_integrity_report_empty_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {})

    report = build_data_integrity_report(
        jpx_master_candidates=100, static_filter_included=80, static_filter_excluded=20,
        manifest_path=manifest_path, raw_dir=raw_dir,
        start=date(2024, 1, 1), end=date(2024, 1, 5),
    )
    assert report.funnel.fetch_attempted == 0
    assert report.funnel.final_universe == 0
    assert report.ticker_coverage == []
