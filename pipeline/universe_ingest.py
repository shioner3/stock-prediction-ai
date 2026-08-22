"""Phase 6.5 Full Universe ingestion: checkpoint/resume wrapper around
pipeline/ingest.py::ingest_one_ticker() for the ~3,700-ticker Prime +
Standard + Growth universe (vs Phase 1-6's 4 hand-picked tickers).

Reuses ingest_one_ticker() UNCHANGED for every ticker - this module only
adds (a) a persisted per-ticker manifest so an interrupted run can resume
without re-fetching already-successful tickers, and (b) skip-if-cached
logic. It does not alter what gets fetched, how it's validated, or what
gets written to data/raw//data/processed/ - see
tests/test_universe_ingest.py's "produces identical output to
run_ingest()" test for the direct proof (Phase 6.5 spec section 4.8).

Phase 7 bug fix (2026-08-20): this module originally never fetched the
TOPIX Proxy market index at all, unlike pipeline/ingest.py::run_ingest()
which does. This went unnoticed through Phase 6.5 because
data/processed/TOPIX.parquet already existed there from Phase 1's
original run_ingest() call, long before this module existed - Phase 6.5
silently reused that leftover file. Phase 7's data/phase7/processed/ is a
brand-new directory with no such file, which exposed the gap (Relative
Strength went silently NaN, then run_walk_forward's Market Regime step
crashed outright with FileNotFoundError). Fixed by fetching+saving the
index here too, mirroring run_ingest()'s own index_provider.fetch_index()
call exactly. Phase 6.5's/Phase 6's own already-published results are
unaffected - they used a real, correctly-fetched TOPIX Proxy file, not a
missing one.

Manifest schema (JSON, one entry per ticker):
    {
      "generated_at": "...",
      "tickers": {
        "7203": {"status": "success", "included_in_universe": true,
                  "error": null, "fetched_at": "..."},
        ...
      }
    }

status is one of "success" (fetch OK), "partial" (FetchStatus.PARTIAL),
or "failed" (FetchStatus.FAILED) - always the FETCH outcome, independent
of included_in_universe (whether it also passed the liquidity filter -
a ticker can fetch successfully and still be excluded from the
Universe). A "pending" ticker is simply absent from the manifest.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from config.loader import AppConfig, UniverseFilterConfig
from pipeline.ingest import ingest_one_ticker
from providers.base import FetchStatus, MarketIndexProvider, OHLCVProvider
from providers.market_index import YFinanceMarketIndexProvider
from providers.yfinance_provider import YFinanceProvider
from storage.parquet_store import save_ohlcv
from validation.ohlcv import clean_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class UniverseIngestSummary:
    ticker_count: int
    attempted: int = 0
    skipped_cached: int = 0
    fetch_success: int = 0
    fetch_partial: int = 0
    fetch_failed: int = 0
    processed_count: int = 0
    excluded_by_liquidity: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    topix_available: bool = False
    duration_seconds: float = 0.0


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"generated_at": None, "tickers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_universe_ingest(
    config: AppConfig,
    filter_config: UniverseFilterConfig,
    tickers: list[str],
    manifest_path: Path,
    provider: OHLCVProvider | None = None,
    index_provider: MarketIndexProvider | None = None,
    force_refetch: bool = False,
    save_every: int = 20,
) -> UniverseIngestSummary:
    """tickers: the full candidate list (already static-filtered, e.g.
    universe/jpx_master.py's Prime+Standard+Growth stock rows). Every
    ticker already marked "success" or "partial" in the manifest at
    manifest_path is skipped (its data is reused as-is) unless
    force_refetch=True. save_every controls how often the manifest is
    flushed to disk (every ticker would also be correct, just slower for
    a run of thousands - flushing periodically still bounds how much
    work an interruption can lose).

    Also fetches the TOPIX Proxy market index into
    config.data.raw_dir/processed_dir (see this module's docstring,
    "Phase 7 bug fix") - skipped if a processed TOPIX.parquet already
    exists there and force_refetch=False, matching the per-ticker
    caching behavior above.
    """
    start_time = time.monotonic()
    provider = provider or YFinanceProvider(
        max_retries=config.data.fetch.max_retries,
        backoff_base_seconds=config.data.fetch.backoff_base_seconds,
        backoff_max_seconds=config.data.fetch.backoff_max_seconds,
        timeout_seconds=config.data.fetch.timeout_seconds,
    )
    index_provider = index_provider or YFinanceMarketIndexProvider(
        index_symbol=config.data.market_index.ticker,
        name=config.data.market_index.name,
        index_type=config.data.market_index.type,
        max_retries=config.data.fetch.max_retries,
        backoff_base_seconds=config.data.fetch.backoff_base_seconds,
        backoff_max_seconds=config.data.fetch.backoff_max_seconds,
        timeout_seconds=config.data.fetch.timeout_seconds,
    )

    manifest = load_manifest(manifest_path)
    entries: dict = manifest["tickers"]

    end_date = config.data.end_date or date.today()
    start_date = config.data.start_date

    summary = UniverseIngestSummary(ticker_count=len(tickers))
    since_last_save = 0

    for ticker in tickers:
        existing = entries.get(ticker)
        if not force_refetch and existing is not None and existing["status"] in (
            "success", "partial",
        ):
            summary.skipped_cached += 1
            if existing["status"] == "success":
                summary.fetch_success += 1
            else:
                summary.fetch_partial += 1
            if existing.get("included_in_universe"):
                summary.processed_count += 1
            else:
                summary.excluded_by_liquidity += 1
            continue

        summary.attempted += 1
        outcome = ingest_one_ticker(
            ticker, provider, filter_config, start_date, end_date,
            config.data.raw_dir, config.data.processed_dir,
        )

        if outcome.fetch_status == FetchStatus.FAILED:
            status = "failed"
            summary.fetch_failed += 1
            summary.failed_tickers.append(ticker)
        elif outcome.fetch_status == FetchStatus.PARTIAL:
            status = "partial"
            summary.fetch_partial += 1
        else:
            status = "success"
            summary.fetch_success += 1

        if outcome.excluded_by_liquidity:
            summary.excluded_by_liquidity += 1
        elif outcome.processed:
            summary.processed_count += 1

        entries[ticker] = {
            "status": status,
            "included_in_universe": outcome.processed,
            "error": outcome.error,
            "fetched_at": datetime.now().isoformat(),
        }

        since_last_save += 1
        if since_last_save >= save_every:
            _save_manifest(manifest_path, manifest)
            since_last_save = 0
            logger.info(
                "universe ingest checkpoint: %d/%d attempted this run "
                "(%d skipped as already cached)",
                summary.attempted, len(tickers), summary.skipped_cached,
            )

    _save_manifest(manifest_path, manifest)

    processed_topix_path = Path(config.data.processed_dir) / "TOPIX.parquet"
    if not force_refetch and processed_topix_path.exists():
        summary.topix_available = True
    else:
        index_result = index_provider.fetch_index(start_date, end_date)
        if index_result.status == FetchStatus.SUCCESS:
            assert index_result.data is not None
            save_ohlcv(index_result.data, "TOPIX", config.data.raw_dir)
            cleaned_index, _ = clean_ohlcv(index_result.data, "TOPIX")
            save_ohlcv(cleaned_index, "TOPIX", config.data.processed_dir)
            summary.topix_available = True
        else:
            summary.topix_available = False
            logger.warning(
                "TOPIX unavailable (%s) - relative strength features will be "
                "disabled until this is resolved", index_result.error,
            )

    summary.duration_seconds = time.monotonic() - start_time
    logger.info(
        "universe ingest complete: candidates=%d attempted=%d skipped_cached=%d "
        "success=%d partial=%d failed=%d processed=%d excluded_liquidity=%d topix=%s "
        "duration=%.1fs",
        summary.ticker_count, summary.attempted, summary.skipped_cached,
        summary.fetch_success, summary.fetch_partial, summary.fetch_failed,
        summary.processed_count, summary.excluded_by_liquidity, summary.topix_available,
        summary.duration_seconds,
    )
    return summary
