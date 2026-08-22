"""Phase 10 CLI entry point: initializes the Frozen Strategy Forward Test
on first run (writes data/forward_test/manifest.json, freezing T0 and
every code/config hash), then runs one day's Signal/Portfolio update
every time it's invoked afterward. Safe to re-run on the same day
(idempotent - see forward_test/portfolio.py's append-only design and
pipeline/run_forward_test.py's signal-log dedup).

Usage:
    python scripts/run_forward_test_day.py                  # today, JST
    python scripts/run_forward_test_day.py --run-date 2026-08-21
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config, load_universe_filters  # noqa: E402
from forward_test import FORWARD_TEST_ROOT  # noqa: E402
from forward_test.manifest import build_manifest, save_manifest  # noqa: E402
from pipeline.run_forward_test import (  # noqa: E402
    DailyRunResult,
    SafeAbortError,
    StrategyHashMismatchError,
    run_forward_test_day,
)


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Phase 10 Forward Test day")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--filters-config", type=Path, default=Path("config/universe_filters.yaml")
    )
    parser.add_argument(
        "--run-date", type=datetime.date.fromisoformat, default=datetime.date.today(),
    )
    parser.add_argument("--root", type=Path, default=FORWARD_TEST_ROOT)
    parser.add_argument(
        "--jpx-master-path", type=Path, default=Path("data/reference/jpx_master_current.xls"),
    )
    parser.add_argument("--target-direction", type=str, default="LONG")
    parser.add_argument("--target-signal-name", type=str, default="long_oversold_rebound")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--per-trade-notional-fraction", type=float, default=0.01)
    parser.add_argument("--strategy-version", type=str, default="v1")
    return parser.parse_args()


def _print_summary(result: DailyRunResult) -> None:
    print(f"run_date: {result.run_date}")
    print(f"strategy_hash_unchanged: {result.strategy_hash_unchanged}")
    print(f"universe_candidate_count: {result.universe_candidate_count}")
    print(
        f"fetch success/partial/failed: "
        f"{result.fetch_success}/{result.fetch_partial}/{result.fetch_failed}"
    )
    print(f"final_universe_count: {result.final_universe_count}")
    print(f"new_signal_log_entries: {len(result.new_signal_log_entries)}")
    for e in result.new_signal_log_entries[:20]:
        print(f"  {e.ticker} {e.signal_date} score={e.total_score} regime={e.regime}")
    print(f"new_closed_positions: {len(result.new_closed_positions)}")
    for p in result.new_closed_positions[:20]:
        print(f"  {p.ticker} {p.signal_date} return={p.trade_return} pnl={p.pnl}")
    print(f"portfolio_equity: {result.portfolio_equity}")
    print(f"portfolio_realized_pnl: {result.portfolio_realized_pnl}")
    print(f"data_integrity_issues (tickers with issues): {len(result.data_integrity_issues)}")
    for ticker, issue in list(result.data_integrity_issues.items())[:10]:
        print(f"  {ticker}: stale={issue.is_stale} n_issues={len(issue.issues)}")
    print(f"trading_integrity.is_clean: {result.trading_integrity.is_clean}")
    if not result.trading_integrity.is_clean:
        print(f"  duplicate_keys={result.trading_integrity.duplicate_keys}")
        print(f"  impossible_entries={result.trading_integrity.impossible_entries}")
        print(f"  impossible_exits={result.trading_integrity.impossible_exits}")
        print(f"  non_positive_prices={result.trading_integrity.non_positive_prices}")
    print(f"open_positions: {len(result.open_positions)}")
    for op in result.open_positions[:20]:
        print(
            f"  {op.ticker} {op.signal_date} entry={op.entry_price} mark={op.mark_price} "
            f"unrealized_return={op.unrealized_return:.4f}"
        )
    print(f"portfolio_unrealized_pnl: {result.portfolio_unrealized_pnl}")
    print(f"performance_log_written: {result.performance_log_written}")


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    filter_config = load_universe_filters(args.filters_config)
    setup_logging(config.logging.level, config.logging.log_dir)

    root = args.root
    config.data.raw_dir = root / "raw"
    config.data.processed_dir = root / "processed"
    config.data.features_dir = root / "features"
    config.data.signals_dir = root / "signals"
    config.data.scores_dir = root / "scores"

    strategy_manifest_path = root / "manifest.json"
    fetch_manifest_path = root / "_fetch_manifest.json"
    portfolio_path = root / "portfolio" / "portfolio.json"
    signal_log_path = root / "signals_log" / "signal_log.jsonl"
    performance_log_path = root / "performance_log" / "performance_log.jsonl"
    daily_dir = root / "daily"

    if not strategy_manifest_path.exists():
        print(f"No Strategy manifest found - initializing Forward Test with T0={args.run_date}")
        manifest = build_manifest(
            config, args.strategy_version, args.run_date,
            args.target_direction, args.target_signal_name,
            args.initial_capital, args.per_trade_notional_fraction,
        )
        save_manifest(manifest, strategy_manifest_path)
        print(f"Strategy manifest saved: {strategy_manifest_path}")
        print(f"  strategy_version={manifest.strategy_version}")
        print(f"  hashes={manifest.hashes}")

    try:
        result = run_forward_test_day(
            config, filter_config, args.run_date,
            strategy_manifest_path, fetch_manifest_path,
            portfolio_path, signal_log_path, performance_log_path, args.jpx_master_path,
            target_direction=args.target_direction, target_signal_name=args.target_signal_name,
        )
    except StrategyHashMismatchError as exc:
        print(f"STRATEGY_HASH_MISMATCH: {exc}")
        sys.exit(1)
    except SafeAbortError as exc:
        print(f"SAFE_ABORT[{exc.reason}]: {exc.detail}")
        sys.exit(2)

    _print_summary(result)

    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / f"{result.run_date.isoformat()}.json"
    daily_path.write_text(
        json.dumps(dataclasses.asdict(result), default=_json_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved: {daily_path}")

    # Spec section 21's directory layout also lists trades/ (a flat
    # ledger view of the Paper Portfolio, refreshed every run) and
    # reports/ (created separately when a report is written).
    trades_dir = root / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    portfolio_snapshot = json.loads(portfolio_path.read_text(encoding="utf-8"))
    (trades_dir / "trades.json").write_text(
        json.dumps(portfolio_snapshot["closed_positions"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "reports").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
