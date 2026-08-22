"""Phase 1 ingestion pipeline: universe -> fetch OHLCV -> validate -> save.

Intentionally narrow - this does not compute features, signals, or scores
(those arrive in Phase 2+). Its only job is to produce an audited
data/raw + validated data/processed dataset that later phases build on,
and to never let a single ticker's failure abort the whole run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from config.loader import AppConfig, UniverseFilterConfig
from providers.base import FetchStatus, MarketIndexProvider, OHLCVProvider
from providers.market_index import YFinanceMarketIndexProvider
from providers.yfinance_provider import YFinanceProvider
from storage.parquet_store import save_ohlcv, save_universe_snapshot
from universe.build import apply_static_filters, load_master_list
from universe.filters import check_price_and_liquidity
from validation.ohlcv import ValidationIssue, clean_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    execution_date: str
    universe_static_count: int
    fetch_success: int = 0
    fetch_partial: int = 0
    fetch_failed: int = 0
    processed_count: int = 0
    excluded_by_liquidity: int = 0
    topix_available: bool = False
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    validation_issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class TickerIngestOutcome:
    """Result of fetching+validating+filtering ONE ticker - the reusable
    unit pipeline/universe_ingest.py's checkpoint/resume wrapper (Phase
    6.5) drives per-ticker, identical to what run_ingest()'s own loop
    below does. Extracting this does not change run_ingest()'s behaviour
    or public signature - see tests/test_pipeline_ingest.py, unchanged.
    """

    ticker: str
    fetch_status: FetchStatus
    error: str | None
    processed: bool  # True iff it passed liquidity filtering and processed/ was written
    excluded_by_liquidity: bool
    validation_issues: list[ValidationIssue] = field(default_factory=list)


def ingest_one_ticker(
    ticker: str,
    provider: OHLCVProvider,
    filter_config: UniverseFilterConfig,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    processed_dir: Path,
) -> TickerIngestOutcome:
    """Fetch, save to raw/, validate+clean, apply the liquidity filter,
    and (if it passes) save to processed/ - exactly the body of
    run_ingest()'s per-ticker loop, factored out so
    pipeline/universe_ingest.py's checkpoint/resume wrapper (Phase 6.5)
    can call the identical logic instead of re-implementing it.
    """
    result = provider.fetch(ticker, start_date, end_date)
    if result.status == FetchStatus.FAILED:
        logger.warning("fetch FAILED for %s: %s", ticker, result.error)
        return TickerIngestOutcome(ticker, result.status, result.error, False, False)

    if result.status == FetchStatus.PARTIAL:
        logger.warning("fetch PARTIAL for %s: %s", ticker, result.error)
    assert result.data is not None
    save_ohlcv(result.data, ticker, raw_dir)

    cleaned, report = clean_ohlcv(result.data, ticker)
    if report.issues:
        for issue in report.issues:
            logger.warning(
                "validation issue [%s] ticker=%s date=%s detail=%s",
                issue.rule, issue.ticker, issue.date, issue.detail,
            )

    liquidity_result = check_price_and_liquidity(cleaned, ticker, filter_config)
    if not liquidity_result.passed:
        logger.info("excluded by liquidity filter: %s (%s)", ticker, liquidity_result.reason)
        return TickerIngestOutcome(
            ticker, result.status, result.error, False, True, report.issues
        )

    save_ohlcv(cleaned, ticker, processed_dir)
    return TickerIngestOutcome(ticker, result.status, result.error, True, False, report.issues)


def run_ingest(
    config: AppConfig,
    filter_config: UniverseFilterConfig,
    tickers_override: list[str] | None = None,
    provider: OHLCVProvider | None = None,
    index_provider: MarketIndexProvider | None = None,
) -> IngestSummary:
    start_time = time.monotonic()
    execution_date = datetime.now().strftime("%Y-%m-%d")

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

    if tickers_override is not None:
        tickers = tickers_override
        universe_static_count = len(tickers)
    else:
        master = load_master_list(config.universe.master_list_path)
        build_result = apply_static_filters(
            master,
            segments=config.universe.segments,
            exclude_etf=config.universe.exclude_etf,
            exclude_reit=config.universe.exclude_reit,
        )
        tickers = build_result.included["code"].tolist()
        universe_static_count = len(tickers)
        logger.info(
            "universe: %d included after static filters, %d excluded",
            len(build_result.included), len(build_result.excluded),
        )
        save_universe_snapshot(
            build_result.included, execution_date, Path("data/processed/universe")
        )

    end_date = config.data.end_date or date.today()
    start_date = config.data.start_date

    summary = IngestSummary(
        execution_date=execution_date, universe_static_count=universe_static_count
    )

    for ticker in tickers:
        outcome = ingest_one_ticker(
            ticker, provider, filter_config, start_date, end_date,
            config.data.raw_dir, config.data.processed_dir,
        )
        summary.validation_issues.extend(outcome.validation_issues)

        if outcome.fetch_status == FetchStatus.FAILED:
            summary.fetch_failed += 1
            summary.errors.append(f"{ticker}: {outcome.error}")
            continue
        if outcome.fetch_status == FetchStatus.PARTIAL:
            summary.fetch_partial += 1
        else:
            summary.fetch_success += 1

        if outcome.excluded_by_liquidity:
            summary.excluded_by_liquidity += 1
            continue

        if outcome.processed:
            summary.processed_count += 1

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
        "ingest complete: universe=%d success=%d partial=%d failed=%d processed=%d "
        "excluded_liquidity=%d topix=%s duration=%.1fs",
        summary.universe_static_count, summary.fetch_success, summary.fetch_partial,
        summary.fetch_failed, summary.processed_count, summary.excluded_by_liquidity,
        summary.topix_available, summary.duration_seconds,
    )
    return summary
