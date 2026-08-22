"""Phase 5 score-building pipeline: reads data/features/* Feature panels
and data/signals/* triggered SignalRecords, computes Score Records
(scoring/pipeline.py) per ticker, writes to data/scores/*.

Does not touch data/raw/, data/processed/, data/features/, or
data/signals/. A single ticker's failure is logged and skipped, same
convention as every other Phase 1-4 pipeline module.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from config.loader import AppConfig
from scoring.pipeline import compute_score_records
from storage.parquet_store import load_feature_panel, load_signal_records, save_score_records

logger = logging.getLogger(__name__)


@dataclass
class BuildScoresSummary:
    ticker_count: int
    success_count: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    total_scores_computed: int = 0
    duration_seconds: float = 0.0


def run_build_scores(config: AppConfig, tickers: list[str] | None = None) -> BuildScoresSummary:
    start_time = time.monotonic()
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)

    if tickers is None:
        tickers = sorted(p.stem for p in signals_dir.glob("*.parquet"))

    summary = BuildScoresSummary(ticker_count=len(tickers))
    for ticker in tickers:
        try:
            signal_records = load_signal_records(ticker, signals_dir)
            if signal_records.empty:
                summary.success_count += 1
                continue
            panel = load_feature_panel(ticker, features_dir)
            score_records = compute_score_records(panel, signal_records, config.scoring)
            save_score_records(score_records, ticker, config.data.scores_dir)
            summary.success_count += 1
            summary.total_scores_computed += len(score_records)
        except Exception:
            logger.exception("score build failed for %s", ticker)
            summary.failed_tickers.append(ticker)

    summary.duration_seconds = time.monotonic() - start_time
    logger.info(
        "score build complete: tickers=%d success=%d failed=%d scored=%d duration=%.1fs",
        summary.ticker_count, summary.success_count, len(summary.failed_tickers),
        summary.total_scores_computed, summary.duration_seconds,
    )
    return summary
