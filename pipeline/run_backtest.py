"""Phase 4 backtest orchestration: reads data/signals/* (SignalRecord)
and data/features/* (OHLCV, via the Feature panel) per ticker, runs
backtest/engine.py::run_backtest_for_ticker() for each, and combines the
Trade Records into one DataFrame.

Deliberately thin: all backtest RULES (Entry=t+1 Open, fixed hold_days,
overlap suppression) live in backtest/engine.py, not here - this module
only does the multi-ticker looping and file I/O, mirroring
pipeline/build_features.py and pipeline/build_signals.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from backtest.engine import TRADE_COLUMNS, SkippedSignal, run_backtest_for_ticker
from config.loader import AppConfig
from storage.parquet_store import load_feature_panel, load_signal_records

logger = logging.getLogger(__name__)


@dataclass
class RunBacktestSummary:
    ticker_count: int
    success_count: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    trades: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS))
    skipped: list[SkippedSignal] = field(default_factory=list)
    duration_seconds: float = 0.0


def run_backtest(config: AppConfig, tickers: list[str] | None = None) -> RunBacktestSummary:
    start_time = time.monotonic()
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)

    if tickers is None:
        tickers = sorted(p.stem for p in signals_dir.glob("*.parquet"))

    all_trades: list[pd.DataFrame] = []
    all_skipped: list[SkippedSignal] = []
    summary = RunBacktestSummary(ticker_count=len(tickers))

    for ticker in tickers:
        try:
            signal_records = load_signal_records(ticker, signals_dir)
            if signal_records.empty:
                summary.success_count += 1
                continue
            ohlcv = load_feature_panel(ticker, features_dir)
            result = run_backtest_for_ticker(signal_records, ohlcv, config.backtest)
            all_trades.append(result.trades)
            all_skipped.extend(result.skipped)
            summary.success_count += 1
        except Exception:
            logger.exception("backtest failed for %s", ticker)
            summary.failed_tickers.append(ticker)

    if all_trades:
        summary.trades = pd.concat(all_trades, ignore_index=True)
    else:
        summary.trades = pd.DataFrame(columns=TRADE_COLUMNS)
    summary.skipped = all_skipped
    summary.duration_seconds = time.monotonic() - start_time

    logger.info(
        "backtest complete: tickers=%d success=%d failed=%d trades=%d skipped=%d duration=%.1fs",
        summary.ticker_count, summary.success_count, len(summary.failed_tickers),
        len(summary.trades), len(summary.skipped), summary.duration_seconds,
    )
    return summary
