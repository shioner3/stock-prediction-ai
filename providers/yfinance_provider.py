"""yfinance-backed implementation of OHLCVProvider.

This is one of only two modules in the project allowed to import yfinance
directly (the other is providers/market_index.py, which reuses the
fetch/standardize helpers here). Domain layers added in later phases
(features/signals/scoring) must never depend on this module or on
yfinance's API shape.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from providers.base import OHLCV_COLUMNS, FetchResult, FetchStatus, OHLCVProvider

logger = logging.getLogger(__name__)


def to_yahoo_symbol(ticker: str) -> str:
    """Convert a bare JPX security code (e.g. '7203') to a Yahoo Finance
    symbol ('7203.T'). Symbols that already contain a suffix (e.g. index
    symbols like '^TOPX') are passed through unchanged.
    """
    ticker = ticker.strip()
    if "." in ticker or ticker.startswith("^"):
        return ticker
    return f"{ticker}.T"


def standardize_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance `Ticker.history()` result to OHLCV_COLUMNS with
    lower-case column names and a plain `date` column (not the index).
    """
    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    df = df[keep]
    return df.sort_values("date").reset_index(drop=True)


def fetch_yf_history(
    symbol: str,
    start: date,
    end: date,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
    timeout_seconds: float = 15.0,
) -> pd.DataFrame:
    """Download daily history for a single Yahoo Finance symbol, retrying
    transient failures with exponential backoff.

    Raises the last exception if every attempt fails; callers turn that
    into a FetchResult(status=FAILED) rather than letting it propagate.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            ticker_obj = yf.Ticker(symbol)
            history = ticker_obj.history(
                start=start,
                end=end + timedelta(days=1),
                interval="1d",
                timeout=timeout_seconds,
                auto_adjust=False,
            )
            return history
        except Exception as exc:  # yfinance raises assorted exception types
            last_exc = exc
            if attempt < max_retries:
                sleep_seconds = min(
                    backoff_base_seconds * (2 ** (attempt - 1)), backoff_max_seconds
                )
                logger.warning(
                    "fetch attempt %d/%d failed for %s: %s (retrying in %.1fs)",
                    attempt, max_retries, symbol, exc, sleep_seconds,
                )
                time.sleep(sleep_seconds)
            else:
                logger.error(
                    "fetch failed for %s after %d attempts: %s", symbol, max_retries, exc
                )
    assert last_exc is not None
    raise last_exc


class YFinanceProvider(OHLCVProvider):
    def __init__(
        self,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        timeout_seconds: float = 15.0,
        min_expected_coverage: float = 0.8,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.timeout_seconds = timeout_seconds
        # Fraction of expected business days in [start, end] that must be
        # present for a fetch to count as SUCCESS rather than PARTIAL. This
        # is an approximation (uses pandas business days, not the real JPX
        # trading calendar) - see README known limitations.
        self.min_expected_coverage = min_expected_coverage

    def fetch(self, ticker: str, start: date, end: date) -> FetchResult:
        symbol = to_yahoo_symbol(ticker)
        try:
            raw = fetch_yf_history(
                symbol,
                start,
                end,
                self.max_retries,
                self.backoff_base_seconds,
                self.backoff_max_seconds,
                self.timeout_seconds,
            )
        except Exception as exc:
            return FetchResult(
                ticker=ticker, status=FetchStatus.FAILED, data=None,
                error=str(exc), attempts=self.max_retries,
            )

        if raw is None or raw.empty:
            return FetchResult(
                ticker=ticker, status=FetchStatus.FAILED, data=None,
                error="no data returned", attempts=self.max_retries,
            )

        df = standardize_history(raw, ticker)
        missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing_cols:
            return FetchResult(
                ticker=ticker, status=FetchStatus.FAILED, data=None,
                error=f"missing expected columns: {missing_cols}", attempts=self.max_retries,
            )

        expected_days = max(len(pd.bdate_range(start, end)), 1)
        coverage = len(df) / expected_days
        status = (
            FetchStatus.SUCCESS if coverage >= self.min_expected_coverage else FetchStatus.PARTIAL
        )
        error = (
            None if status == FetchStatus.SUCCESS
            else f"coverage {coverage:.0%} of expected business days"
        )
        return FetchResult(ticker=ticker, status=status, data=df, error=error, attempts=1)
