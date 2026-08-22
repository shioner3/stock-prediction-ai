"""Phase 4 signal-building pipeline: reads data/features/* Feature panels,
evaluates all 12 Signals (signals/registry.py) per ticker, writes the
triggered-only SignalRecord rows to data/signals/*.

Does not touch data/raw/, data/processed/, or data/features/. A single
ticker's failure is logged and skipped, same convention as
pipeline/ingest.py and pipeline/build_features.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from config.loader import AppConfig
from signals.pipeline import compute_signal_records
from storage.parquet_store import load_feature_panel, save_signal_records

logger = logging.getLogger(__name__)


@dataclass
class BuildSignalsSummary:
    ticker_count: int
    success_count: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    total_signals_triggered: int = 0
    duration_seconds: float = 0.0


def run_build_signals(
    config: AppConfig, tickers: list[str] | None = None
) -> BuildSignalsSummary:
    start_time = time.monotonic()
    features_dir = Path(config.data.features_dir)

    if tickers is None:
        tickers = sorted(p.stem for p in features_dir.glob("*.parquet"))

    summary = BuildSignalsSummary(ticker_count=len(tickers))
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
            records = compute_signal_records(panel, config.signals)
            save_signal_records(records, ticker, config.data.signals_dir)
            summary.success_count += 1
            summary.total_signals_triggered += len(records)
        except Exception:
            logger.exception("signal build failed for %s", ticker)
            summary.failed_tickers.append(ticker)

    summary.duration_seconds = time.monotonic() - start_time
    logger.info(
        "signal build complete: tickers=%d success=%d failed=%d triggered=%d duration=%.1fs",
        summary.ticker_count, summary.success_count, len(summary.failed_tickers),
        summary.total_signals_triggered, summary.duration_seconds,
    )
    return summary
