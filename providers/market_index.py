"""Market benchmark index providers (e.g. TOPIX) used for relative strength.

Relative-strength feature code (added in Phase 3) must depend only on
MarketIndexProvider, never on a specific vendor implementation - this keeps
the door open for swapping yfinance for J-Quants (or another vendor) later
without touching features/relative_strength.py.
"""

from __future__ import annotations

import logging
from datetime import date

from providers.base import FetchResult, FetchStatus, MarketIndexMeta, MarketIndexProvider
from providers.yfinance_provider import fetch_yf_history, standardize_history

logger = logging.getLogger(__name__)


class YFinanceMarketIndexProvider(MarketIndexProvider):
    def __init__(
        self,
        index_symbol: str = "1306.T",
        name: str = "TOPIX Proxy",
        index_type: str = "ETF_PROXY",
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.index_symbol = index_symbol
        self.name = name
        self.index_type = index_type
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.timeout_seconds = timeout_seconds

    def describe(self) -> MarketIndexMeta:
        return MarketIndexMeta(
            name=self.name, symbol=self.index_symbol, type=self.index_type, source="yfinance"
        )

    def fetch_index(self, start: date, end: date) -> FetchResult:
        try:
            raw = fetch_yf_history(
                self.index_symbol,
                start,
                end,
                self.max_retries,
                self.backoff_base_seconds,
                self.backoff_max_seconds,
                self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "%s (%s) fetch failed (%s) - relative strength features will be "
                "unavailable until this is resolved", self.name, self.index_symbol, exc,
            )
            return FetchResult(
                ticker=self.index_symbol, status=FetchStatus.FAILED, data=None,
                error=str(exc), attempts=self.max_retries,
            )

        if raw is None or raw.empty:
            logger.warning(
                "%s (%s) fetch returned no data - relative strength features "
                "will be unavailable until this is resolved", self.name, self.index_symbol,
            )
            return FetchResult(
                ticker=self.index_symbol, status=FetchStatus.FAILED, data=None,
                error="no data returned", attempts=self.max_retries,
            )

        df = standardize_history(raw, self.index_symbol)
        return FetchResult(
            ticker=self.index_symbol, status=FetchStatus.SUCCESS, data=df, attempts=1
        )
