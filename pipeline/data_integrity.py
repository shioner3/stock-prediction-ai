"""Phase 6.5 section 27: Data Integrity funnel report.

Answers "how did we get from ~4,400 JPX-listed instruments down to the
final Signal-eligible Universe, and what does the RAW fetched data for
each surviving ticker actually look like" - reusing
pipeline/universe_ingest.py's manifest (already the single source of
truth for fetch outcomes) and validation/ohlcv.py's validate_ohlcv()
(already the single source of truth for row-level data-quality issues)
rather than re-deriving either.

This module only DESCRIBES what happened; it does not filter, clean, or
otherwise change any data - it runs strictly after ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.universe_ingest import load_manifest
from storage.parquet_store import load_ohlcv
from validation.ohlcv import validate_ohlcv


@dataclass
class TickerCoverage:
    ticker: str
    row_count: int
    date_min: date | None
    date_max: date | None
    expected_business_days: int
    coverage_ratio: float | None
    duplicate_dates: int
    invalid_ohlc_rows: int  # non-positive O/H/L/C, or high < low
    negative_volume_rows: int
    zero_volume_rows: int
    nan_rows: int  # any of open/high/low/close/volume is NaN


@dataclass
class DataIntegrityFunnel:
    jpx_master_candidates: int
    static_filter_included: int
    static_filter_excluded: int
    fetch_attempted: int
    fetch_success: int
    fetch_partial: int
    fetch_failed: int
    price_liquidity_excluded: int
    final_universe: int


@dataclass
class DataIntegrityReport:
    funnel: DataIntegrityFunnel
    ticker_coverage: list[TickerCoverage]


def compute_ticker_coverage(
    df: pd.DataFrame, ticker: str, start: date, end: date
) -> TickerCoverage:
    """df: the RAW (as-fetched, pre-clean_ohlcv) OHLCV for one ticker -
    this audits what was actually returned by the Provider, not what
    survived validation/ohlcv.py's row-dropping.
    """
    expected = max(len(pd.bdate_range(start, end)), 1)

    if df.empty:
        return TickerCoverage(
            ticker=ticker, row_count=0, date_min=None, date_max=None,
            expected_business_days=expected, coverage_ratio=0.0,
            duplicate_dates=0, invalid_ohlc_rows=0, negative_volume_rows=0,
            zero_volume_rows=0, nan_rows=0,
        )

    report = validate_ohlcv(df, ticker)
    duplicate_dates = sum(1 for i in report.issues if i.rule == "duplicate_date")
    invalid_ohlc = sum(
        1 for i in report.issues if i.rule in ("non_positive_price", "high_below_low")
    )
    negative_volume = sum(1 for i in report.issues if i.rule == "negative_volume")
    zero_volume = int((df["volume"] == 0).sum())
    ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    nan_rows = int(df[ohlcv_cols].isna().any(axis=1).sum())

    return TickerCoverage(
        ticker=ticker,
        row_count=len(df),
        date_min=df["date"].min(),
        date_max=df["date"].max(),
        expected_business_days=expected,
        coverage_ratio=len(df) / expected,
        duplicate_dates=duplicate_dates,
        invalid_ohlc_rows=invalid_ohlc,
        negative_volume_rows=negative_volume,
        zero_volume_rows=zero_volume,
        nan_rows=nan_rows,
    )


def build_data_integrity_report(
    jpx_master_candidates: int,
    static_filter_included: int,
    static_filter_excluded: int,
    manifest_path: Path,
    raw_dir: Path,
    start: date,
    end: date,
) -> DataIntegrityReport:
    manifest = load_manifest(manifest_path)
    entries: dict = manifest.get("tickers", {})

    fetch_success = sum(1 for e in entries.values() if e["status"] == "success")
    fetch_partial = sum(1 for e in entries.values() if e["status"] == "partial")
    fetch_failed = sum(1 for e in entries.values() if e["status"] == "failed")
    final_universe = sum(1 for e in entries.values() if e.get("included_in_universe"))
    fetched_ok = fetch_success + fetch_partial
    price_liquidity_excluded = fetched_ok - final_universe

    coverage: list[TickerCoverage] = []
    for ticker, entry in sorted(entries.items()):
        if entry["status"] not in ("success", "partial"):
            continue
        df = load_ohlcv(ticker, raw_dir)
        coverage.append(compute_ticker_coverage(df, ticker, start, end))

    funnel = DataIntegrityFunnel(
        jpx_master_candidates=jpx_master_candidates,
        static_filter_included=static_filter_included,
        static_filter_excluded=static_filter_excluded,
        fetch_attempted=len(entries),
        fetch_success=fetch_success,
        fetch_partial=fetch_partial,
        fetch_failed=fetch_failed,
        price_liquidity_excluded=price_liquidity_excluded,
        final_universe=final_universe,
    )
    return DataIntegrityReport(funnel=funnel, ticker_coverage=coverage)
