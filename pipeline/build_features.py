"""Phase 2/3 feature-building pipeline: reads validated data/processed/*
OHLCV, computes the full feature panel per ticker (Trend/Momentum/
Volatility/Volume/Indicators/Breakout/Pullback, plus Relative Strength
when the market benchmark is available), writes to data/features/*.

Does not touch data/raw/ or data/processed/ - those remain exactly what
pipeline/ingest.py produced. A single ticker's failure is logged and
skipped, same convention as pipeline/ingest.py, so one bad file can't
abort the whole build. The market benchmark (data/processed/TOPIX.parquet
- see providers/market_index.py for why the file is named "TOPIX" while
actually holding the TOPIX Proxy, 1306.T) is loaded once and reused for
every ticker; this module reads that already-fetched file, it never
calls a Provider itself - pipeline/ingest.py is the only place that
talks to MarketIndexProvider.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config.loader import AppConfig
from features.pipeline import compute_feature_panel
from providers.base import MarketIndexMeta
from storage.parquet_store import load_ohlcv, save_feature_panel

logger = logging.getLogger(__name__)

# TOPIX Proxy (or whatever the configured market-index proxy is) is a
# benchmark, not a tradeable universe member - it is read separately as
# the Relative Strength input, never treated as "just another ticker".
_MARKET_INDEX_FILENAME = "TOPIX"
_NON_TICKER_FILES = {_MARKET_INDEX_FILENAME}


@dataclass
class BuildFeaturesSummary:
    ticker_count: int
    success_count: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    market_data_available: bool = False
    duration_seconds: float = 0.0


def _load_market_df(config: AppConfig) -> pd.DataFrame | None:
    processed_dir = Path(config.data.processed_dir)
    market_meta = MarketIndexMeta(
        name=config.data.market_index.name,
        symbol=config.data.market_index.ticker,
        type=config.data.market_index.type,
        source=config.data.market_index.provider,
    )
    try:
        market_df = load_ohlcv(_MARKET_INDEX_FILENAME, processed_dir)
    except FileNotFoundError:
        logger.warning(
            "%s (%s) data not found in %s - Relative Strength disabled, "
            "rs_5d/rs_20d/rs_60d will be NaN for every ticker",
            market_meta.name, market_meta.symbol, processed_dir,
        )
        return None

    logger.info(
        "Relative Strength benchmark: %s (%s, type=%s, source=%s)",
        market_meta.name, market_meta.symbol, market_meta.type, market_meta.source,
    )
    return market_df


def run_build_features(
    config: AppConfig, tickers: list[str] | None = None
) -> BuildFeaturesSummary:
    start_time = time.monotonic()
    processed_dir = Path(config.data.processed_dir)

    if tickers is None:
        tickers = sorted(
            p.stem for p in processed_dir.glob("*.parquet") if p.stem not in _NON_TICKER_FILES
        )

    market_df = _load_market_df(config)

    summary = BuildFeaturesSummary(
        ticker_count=len(tickers), market_data_available=market_df is not None
    )
    for ticker in tickers:
        try:
            ohlcv = load_ohlcv(ticker, processed_dir)
            panel = compute_feature_panel(
                ohlcv,
                recent_high_window=config.features.pullback.recent_high_window,
                market_df=market_df,
            )
            save_feature_panel(panel, ticker, config.data.features_dir)
            summary.success_count += 1
        except Exception:
            logger.exception("feature build failed for %s", ticker)
            summary.failed_tickers.append(ticker)

    summary.duration_seconds = time.monotonic() - start_time
    logger.info(
        "feature build complete: tickers=%d success=%d failed=%d "
        "market_data_available=%s duration=%.1fs",
        summary.ticker_count, summary.success_count, len(summary.failed_tickers),
        summary.market_data_available, summary.duration_seconds,
    )
    return summary
