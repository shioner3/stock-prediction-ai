"""Phase 10/11: Frozen Strategy Forward Test daily orchestration.

Every day this runs, it:

1. Rebuilds the Universe candidate list from a FRESH JPX master snapshot
   using the SAME frozen filter logic (universe/build.py::
   apply_static_filters, universe/filters.py::check_price_and_liquidity -
   unchanged) - spec section 5: the FILTER LOGIC is frozen, not a frozen
   ticker list, so the Forward Test Universe stays current as listings
   change.
2. Fetches each Universe ticker's OHLCV from a fixed lookback anchor
   through run_date, via pipeline.universe_ingest.run_universe_ingest()
   (UNCHANGED) with force_refetch=True - Forward Test always re-fetches
   the full lookback window rather than trying to append incrementally,
   since the existing Provider has no incremental-append mode and a
   full ~400-trading-day fetch is cheap at a once-a-day cadence (the
   same cost Phase 6.5-9 already paid once). This overwrites the Forward
   Test's OWN raw/processed cache daily - that is NOT the same as
   overwriting the immutable Signal Log or Paper Portfolio (which stay
   strictly append-only, see forward_test/portfolio.py).
3. Recomputes Feature/Signal/Score via pipeline.build_features/
   build_signals/build_scores (all UNCHANGED, the same batch functions
   every prior phase used).
4. Runs Data Integrity checks (forward_test/integrity.py) - a ticker
   whose fetch didn't actually reach run_date (is_stale=True) simply has
   no Feature/Signal row for run_date at all, so it is NATURALLY excluded
   from "new signal today" detection with no extra filtering code needed
   (spec section 6). A global STALE_THRESHOLD_EXCEEDED SAFE_ABORT still
   guards against "most of the Universe is broken today".
5. Appends any NEWLY-detected long_oversold_rebound signal occurrences
   to an immutable, append-only Signal Log (never overwritten - spec
   section 7/11), capturing the Score/Feature/Regime values AS COMPUTED
   at detection time.
6. Re-derives ALL resolvable Trade Records via pipeline.run_backtest
   (UNCHANGED, itself built on backtest/engine.py) over the accumulated
   history, and appends any NEWLY-resolved trades to the Paper Portfolio
   (append-only by key - safe to re-derive the full set every day).
7. Computes still-OPEN positions (entered but not yet exited) marked to
   the latest available Close (forward_test/portfolio.py::
   compute_open_positions - descriptive only, not a re-implementation of
   the frozen Backtest Engine, which never reports partial trades).
8. Appends one row to the append-only Daily Performance Log
   (forward_test/performance_log.py).
9. Runs Trading Integrity checks and the Strategy Hash check.

This is NOT a real order execution system (spec section 13/22) - step 6/7
only mark a Position as economically resolved/open once the relevant
Close is available in already-fetched history; nothing is ever sent to
a broker, and no real money is involved.

T0 filtering (spec section 3): the lookback fetch pulls ~650 calendar
days of PRE-T0 history purely for Feature warmup (SMA200 etc need it) -
but every Signal Log entry and every Paper Portfolio trade is filtered
to signal_date >= T0 (read from the Strategy manifest, never a caller
argument, so it cannot drift day to day). Pre-T0 signals exist only to
let TODAY's Features compute correctly; they are never logged or traded
as Forward Test evidence, since that evidence is already fully covered
by Phase 6.5-9's historical OOS analysis.

SAFE_ABORT (spec section 29): rather than proceeding on incomplete or
corrupted state, run_forward_test_day() raises SafeAbortError with one
of a small fixed set of reason codes when a systemic problem is detected
(not a single ticker's ordinary fetch failure, which is already handled
per-ticker throughout Phase 1-10's pipeline).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from backtest.market_regime import compute_market_regime
from config.loader import AppConfig, UniverseFilterConfig
from forward_test.integrity import (
    DataIntegrityResult,
    TradingIntegrityResult,
    check_data_integrity,
    check_trading_integrity,
)
from forward_test.manifest import load_manifest_raw, verify_strategy_hashes_unchanged
from forward_test.performance_log import (
    PerformanceLogEntry,
    compute_cumulative_return,
    compute_daily_return,
    compute_max_drawdown,
)
from forward_test.performance_log import (
    append_entry as append_performance_log_entry,
)
from forward_test.performance_log import (
    load_entries as load_performance_log_entries,
)
from forward_test.portfolio import (
    OpenPosition,
    PortfolioState,
    Position,
    compute_open_positions,
    load_portfolio,
    save_portfolio,
)
from pipeline.build_features import run_build_features
from pipeline.build_scores import run_build_scores
from pipeline.build_signals import run_build_signals
from pipeline.run_backtest import run_backtest
from pipeline.universe_ingest import run_universe_ingest
from providers.base import MarketIndexProvider, OHLCVProvider
from storage.parquet_store import load_ohlcv, load_score_records, load_signal_records
from universe.build import apply_static_filters
from universe.jpx_master import load_jpx_master

logger = logging.getLogger(__name__)

# ~400 trading days is enough to warm up every existing Feature (the
# widest window in the codebase is SMA_WINDOWS' 200-day SMA,
# features/trend.py) with a comfortable margin - see this module's
# docstring point 2 for why a full re-fetch (not incremental append) is
# used each day.
DEFAULT_LOOKBACK_CALENDAR_DAYS = 650

DEFAULT_STALE_FRACTION_ABORT_THRESHOLD = 0.5


class StrategyHashMismatchError(RuntimeError):
    """Raised when the code/config this Forward Test is locked to has
    changed since the manifest was created - spec section 2: the run
    must be treated as INVALID, never silently continued.
    """


class SafeAbortError(RuntimeError):
    """Raised instead of proceeding on incomplete/corrupted state (spec
    section 29). `reason` is one of a small fixed set of codes:
    MARKET_DATA_UNAVAILABLE, UNIVERSE_DATA_INCOMPLETE,
    STALE_THRESHOLD_EXCEEDED, FEATURE_GENERATION_FAILURE,
    SIGNAL_GENERATION_FAILURE, PORTFOLIO_STATE_CORRUPTION.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"SAFE_ABORT[{reason}]: {detail}")


@dataclass
class SignalLogEntry:
    ticker: str
    signal_date: str
    direction: str
    signal_name: str
    total_score: float | None
    regime: str | None
    logged_at: str


@dataclass
class DailyRunResult:
    run_date: date
    universe_candidate_count: int
    fetch_success: int
    fetch_partial: int
    fetch_failed: int
    final_universe_count: int
    new_signal_log_entries: list[SignalLogEntry]
    new_closed_positions: list[Position]
    open_positions: list[OpenPosition]
    portfolio_equity: float
    portfolio_realized_pnl: float
    portfolio_unrealized_pnl: float
    data_integrity_issues: dict[str, DataIntegrityResult]
    trading_integrity: TradingIntegrityResult
    strategy_hash_unchanged: bool
    strategy_hash_mismatches: list[str]
    performance_log_written: bool


def _load_existing_signal_log_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            keys.add(
                (entry["ticker"], entry["signal_date"], entry["direction"], entry["signal_name"])
            )
    return keys


def _load_all_signal_log_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_signal_log_entries(path: Path, entries: list[SignalLogEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def run_forward_test_day(
    config: AppConfig,
    filter_config: UniverseFilterConfig,
    run_date: date,
    strategy_manifest_path: Path,
    fetch_manifest_path: Path,
    portfolio_path: Path,
    signal_log_path: Path,
    performance_log_path: Path,
    jpx_master_path: Path,
    target_direction: str = "LONG",
    target_signal_name: str = "long_oversold_rebound",
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    stale_fraction_abort_threshold: float = DEFAULT_STALE_FRACTION_ABORT_THRESHOLD,
    provider: OHLCVProvider | None = None,
    index_provider: MarketIndexProvider | None = None,
) -> DailyRunResult:
    if not strategy_manifest_path.exists():
        raise RuntimeError(
            f"no Strategy manifest at {strategy_manifest_path} - "
            "run forward_test initialization (build_manifest + save_manifest) first"
        )
    saved_manifest = load_manifest_raw(strategy_manifest_path)
    hash_unchanged, hash_mismatches = verify_strategy_hashes_unchanged(saved_manifest)
    if not hash_unchanged:
        raise StrategyHashMismatchError(
            f"CONFIG/CODE CHANGED since Forward Test T0 - mismatched fields: {hash_mismatches}. "
            "Per spec section 11, this must be treated as a NEW Strategy Version, not a "
            "continuation of the existing one."
        )
    t0 = date.fromisoformat(saved_manifest["t0"])

    if portfolio_path.exists():
        try:
            portfolio = load_portfolio(portfolio_path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SafeAbortError(
                "PORTFOLIO_STATE_CORRUPTION", f"{portfolio_path} could not be parsed: {exc}"
            ) from exc
    else:
        portfolio = PortfolioState(
            initial_capital=saved_manifest["initial_capital"],
            per_trade_notional_fraction=saved_manifest["per_trade_notional_fraction"],
        )

    config.data.start_date = run_date - timedelta(days=lookback_calendar_days)
    config.data.end_date = run_date

    master = load_jpx_master(jpx_master_path)
    build_result = apply_static_filters(
        master,
        segments=config.universe.segments,
        exclude_etf=config.universe.exclude_etf,
        exclude_reit=config.universe.exclude_reit,
    )
    tickers = build_result.included["code"].tolist()

    logger.info(
        "Forward Test %s: %d candidate tickers, fetching %s..%s",
        run_date, len(tickers), config.data.start_date, run_date,
    )
    ingest_summary = run_universe_ingest(
        config, filter_config, tickers, fetch_manifest_path,
        provider=provider, index_provider=index_provider, force_refetch=True,
    )
    final_tickers = sorted(
        t for t, e in json.loads(fetch_manifest_path.read_text(encoding="utf-8"))["tickers"].items()
        if e.get("included_in_universe")
    )
    if not final_tickers:
        raise SafeAbortError(
            "UNIVERSE_DATA_INCOMPLETE",
            f"0 tickers passed the frozen Universe filter out of {len(tickers)} candidates",
        )

    feat_summary = run_build_features(config, tickers=final_tickers)
    if len(feat_summary.failed_tickers) == len(final_tickers):
        raise SafeAbortError(
            "FEATURE_GENERATION_FAILURE", f"all {len(final_tickers)} tickers failed Feature build"
        )
    sig_summary = run_build_signals(config, tickers=final_tickers)
    if len(sig_summary.failed_tickers) == len(final_tickers):
        raise SafeAbortError(
            "SIGNAL_GENERATION_FAILURE", f"all {len(final_tickers)} tickers failed Signal build"
        )
    run_build_scores(config, tickers=final_tickers)

    try:
        topix = load_ohlcv("TOPIX", config.data.processed_dir)
    except FileNotFoundError as exc:
        raise SafeAbortError("MARKET_DATA_UNAVAILABLE", "TOPIX Proxy fetch failed") from exc
    if topix.empty:
        raise SafeAbortError("MARKET_DATA_UNAVAILABLE", "TOPIX Proxy data is empty")
    regime_df = compute_market_regime(topix, config.validation.market_regime)
    regime_by_date = dict(zip(regime_df["date"], regime_df["regime"]))

    data_integrity: dict[str, DataIntegrityResult] = {}
    for ticker in final_tickers:
        df = load_ohlcv(ticker, config.data.raw_dir)
        result = check_data_integrity(df, ticker, expected_date=run_date)
        if not result.is_clean:
            data_integrity[ticker] = result
    stale_count = sum(1 for r in data_integrity.values() if r.is_stale)
    stale_fraction = stale_count / len(final_tickers)
    if stale_fraction > stale_fraction_abort_threshold:
        raise SafeAbortError(
            "STALE_THRESHOLD_EXCEEDED",
            f"{stale_count}/{len(final_tickers)} tickers ({stale_fraction:.0%}) are stale, "
            f"threshold is {stale_fraction_abort_threshold:.0%}",
        )

    existing_log_keys = _load_existing_signal_log_keys(signal_log_path)
    new_log_entries: list[SignalLogEntry] = []
    ohlcv_by_ticker: dict[str, object] = {}
    for ticker in final_tickers:
        signal_records = load_signal_records(ticker, config.data.signals_dir)
        target = signal_records[
            (signal_records["direction"] == target_direction)
            & (signal_records["signal_name"] == target_signal_name)
            & (signal_records["date"] >= t0)
        ]
        if target.empty:
            continue
        score_records = load_score_records(ticker, config.data.scores_dir)
        score_target = score_records[
            (score_records["direction"] == target_direction)
            & (score_records["signal_name"] == target_signal_name)
        ]
        score_by_date = dict(zip(score_target["date"], score_target["total_score"]))

        for row in target.itertuples(index=False):
            key = (ticker, row.date.isoformat(), target_direction, target_signal_name)
            if key in existing_log_keys:
                continue
            new_log_entries.append(
                SignalLogEntry(
                    ticker=ticker,
                    signal_date=row.date.isoformat(),
                    direction=target_direction,
                    signal_name=target_signal_name,
                    total_score=score_by_date.get(row.date),
                    regime=regime_by_date.get(row.date),
                    logged_at=datetime.now().isoformat(),
                )
            )
            existing_log_keys.add(key)

    _append_signal_log_entries(signal_log_path, new_log_entries)

    backtest_summary = run_backtest(config, tickers=final_tickers)
    trades = backtest_summary.trades
    target_trades = trades[
        (trades["direction"] == target_direction)
        & (trades["signal_name"] == target_signal_name)
        & (trades["signal_date"] >= t0)
    ]

    base_cost_bps = saved_manifest["transaction_cost_bps"]
    newly_closed = portfolio.record_closed_trades(target_trades, base_cost_bps)
    save_portfolio(portfolio, portfolio_path)

    all_log_entries = _load_all_signal_log_entries(signal_log_path)
    closed_keys = portfolio.recorded_keys()
    pending_signals = [
        (e["ticker"], e["signal_name"], e["direction"], date.fromisoformat(e["signal_date"]))
        for e in all_log_entries
        if (e["ticker"], e["signal_name"], e["direction"], e["signal_date"]) not in closed_keys
    ]
    for ticker, _, _, _ in pending_signals:
        if ticker not in ohlcv_by_ticker:
            try:
                ohlcv_by_ticker[ticker] = load_ohlcv(ticker, config.data.processed_dir)
            except FileNotFoundError:
                continue
    open_positions = compute_open_positions(
        pending_signals, ohlcv_by_ticker, portfolio.notional_per_trade  # type: ignore[arg-type]
    )

    trading_integrity = check_trading_integrity(portfolio.closed_positions)

    unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
    total_equity = portfolio.equity + unrealized_pnl
    prior_entries = load_performance_log_entries(performance_log_path)
    previous_equity = prior_entries[-1]["equity"] if prior_entries else None
    equity_history = [e["equity"] for e in prior_entries] + [total_equity]

    hashes = saved_manifest["hashes"]
    performance_entry = PerformanceLogEntry(
        date=run_date.isoformat(),
        strategy_version=saved_manifest["strategy_version"],
        strategy_hash=hashes["config_hash"],
        universe_size=len(final_tickers),
        data_timestamp=datetime.now().isoformat(),
        market_regime_summary=str(regime_by_date.get(run_date, "UNKNOWN")),
        signal_count=len(new_log_entries),
        candidate_count=len(tickers),
        open_positions=len(open_positions),
        closed_positions=len(portfolio.closed_positions),
        realized_pnl=portfolio.realized_pnl,
        unrealized_pnl=unrealized_pnl,
        equity=total_equity,
        daily_return=compute_daily_return(total_equity, previous_equity),
        cumulative_return=compute_cumulative_return(total_equity, portfolio.initial_capital),
        max_drawdown=compute_max_drawdown(equity_history),
        data_quality_status="OK" if not data_integrity else f"DEGRADED({len(data_integrity)})",
    )
    performance_log_written = append_performance_log_entry(performance_log_path, performance_entry)

    return DailyRunResult(
        run_date=run_date,
        universe_candidate_count=len(tickers),
        fetch_success=ingest_summary.fetch_success,
        fetch_partial=ingest_summary.fetch_partial,
        fetch_failed=ingest_summary.fetch_failed,
        final_universe_count=len(final_tickers),
        new_signal_log_entries=new_log_entries,
        new_closed_positions=newly_closed,
        open_positions=open_positions,
        portfolio_equity=portfolio.equity,
        portfolio_realized_pnl=portfolio.realized_pnl,
        portfolio_unrealized_pnl=unrealized_pnl,
        data_integrity_issues=data_integrity,
        trading_integrity=trading_integrity,
        strategy_hash_unchanged=hash_unchanged,
        strategy_hash_mismatches=hash_mismatches,
        performance_log_written=performance_log_written,
    )
