"""Abstract interfaces for OHLCV and market-index data providers.

Domain code (features/signals/scoring, added in later phases) must depend
only on these interfaces, never on a specific vendor (yfinance, J-Quants,
...). This is what lets the data source be swapped later without touching
analysis logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd

OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


class FetchStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass
class FetchResult:
    """Outcome of a single fetch call.

    Providers must never raise on ordinary failures (network errors, no
    data for a delisted ticker, etc.) - those are reported via `status`
    so the pipeline can log them and continue with the remaining tickers
    instead of aborting the whole run.
    """

    ticker: str
    status: FetchStatus
    data: pd.DataFrame | None
    error: str | None = None
    attempts: int = 1


class OHLCVProvider(ABC):
    """Fetches daily OHLCV data for a single ticker.

    Implementations must return data with columns OHLCV_COLUMNS.
    """

    @abstractmethod
    def fetch(self, ticker: str, start: date, end: date) -> FetchResult:
        raise NotImplementedError


@dataclass
class MarketIndexMeta:
    """Identifies what a MarketIndexProvider actually returns.

    Exists so callers (relative-strength logging, FeatureMeta
    descriptions) never have to guess or hardcode that assumption
    themselves - they call describe() instead. Concretely: today's
    provider returns "TOPIX Proxy" data (an ETF that tracks TOPIX, not
    the raw TOPIX index itself, since the index isn't available through
    this data feed - see providers/market_index.py), and every consumer
    of this metadata should say so rather than saying "TOPIX".
    """

    name: str
    symbol: str
    type: str  # e.g. "ETF_PROXY", "INDEX"
    source: str  # e.g. "yfinance"


class MarketIndexProvider(ABC):
    """Fetches a market benchmark index (e.g. TOPIX) used for relative strength.

    Kept as a separate interface from OHLCVProvider because index data
    (symbol format, availability) often differs from individual equities
    even within the same vendor.
    """

    @abstractmethod
    def fetch_index(self, start: date, end: date) -> FetchResult:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> MarketIndexMeta:
        raise NotImplementedError
