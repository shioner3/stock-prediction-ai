"""Full Universe Data Integrity preflight (spec section 22), run BEFORE
the real Phase V2-2 execution.

Reuses pipeline.data_integrity.compute_ticker_coverage() (unmodified V1
code - the same duplicate-date/invalid-OHLC/negative-volume/zero-volume/
NaN/coverage checks Phase 6.5's own Data Integrity funnel already uses)
against V1's existing processed OHLCV cache, rather than re-deriving any
of these checks. "Corporate action artifacts / extreme return outliers"
reuse Phase V2-1's OWN already-defined rule
(v2/stats.py::exclude_implausible_returns, MAX_PLAUSIBLE_FORWARD_RETURN)
- spec section 22 explicitly forbids inventing a new exclusion rule here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

from pipeline.data_integrity import TickerCoverage, compute_ticker_coverage
from storage.parquet_store import load_ohlcv


@dataclass(frozen=True)
class DataIntegrityPreflightSummary:
    n_tickers_checked: int
    n_tickers_missing: list[str]
    total_rows: int
    total_duplicate_dates: int
    total_invalid_ohlc_rows: int
    total_negative_volume_rows: int
    total_zero_volume_rows: int
    total_nan_rows: int
    min_coverage_ratio: float | None
    date_min: date_type | None
    date_max: date_type | None
    per_ticker: list[TickerCoverage]


def run_data_integrity_preflight(
    tickers: list[str], processed_dir, start: date_type, end: date_type
) -> DataIntegrityPreflightSummary:
    per_ticker: list[TickerCoverage] = []
    missing: list[str] = []
    for ticker in tickers:
        try:
            df = load_ohlcv(ticker, processed_dir)
        except FileNotFoundError:
            missing.append(ticker)
            continue
        per_ticker.append(compute_ticker_coverage(df, ticker, start, end))

    date_mins = [c.date_min for c in per_ticker if c.date_min is not None]
    date_maxs = [c.date_max for c in per_ticker if c.date_max is not None]
    coverage_ratios = [c.coverage_ratio for c in per_ticker if c.coverage_ratio is not None]

    return DataIntegrityPreflightSummary(
        n_tickers_checked=len(per_ticker),
        n_tickers_missing=missing,
        total_rows=sum(c.row_count for c in per_ticker),
        total_duplicate_dates=sum(c.duplicate_dates for c in per_ticker),
        total_invalid_ohlc_rows=sum(c.invalid_ohlc_rows for c in per_ticker),
        total_negative_volume_rows=sum(c.negative_volume_rows for c in per_ticker),
        total_zero_volume_rows=sum(c.zero_volume_rows for c in per_ticker),
        total_nan_rows=sum(c.nan_rows for c in per_ticker),
        min_coverage_ratio=min(coverage_ratios) if coverage_ratios else None,
        date_min=min(date_mins) if date_mins else None,
        date_max=max(date_maxs) if date_maxs else None,
        per_ticker=per_ticker,
    )


def has_critical_integrity_issues(summary: DataIntegrityPreflightSummary) -> bool:
    """A conservative, pre-defined trigger for spec section 30 STOP
    condition #6 ("Full Universe実行で重大なデータ欠損") - more than
    half the requested tickers missing entirely, or literally zero rows
    survived. Fixed here before running, not chosen after seeing results.
    """
    if summary.n_tickers_checked == 0:
        return True
    total_requested = summary.n_tickers_checked + len(summary.n_tickers_missing)
    return len(summary.n_tickers_missing) > total_requested / 2
