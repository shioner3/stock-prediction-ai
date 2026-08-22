"""CLI entry point for the Phase 1 data ingestion pipeline.

Usage:
    python scripts/run_ingest.py
    python scripts/run_ingest.py --tickers 7203 6758 9984
    python scripts/run_ingest.py --start-date 2023-01-01 --end-date 2023-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging
from config.loader import load_app_config, load_universe_filters
from pipeline.ingest import run_ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 OHLCV ingestion pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--filters-config", type=Path, default=Path("config/universe_filters.yaml")
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Override the universe with an explicit ticker list (skips the master list)",
    )
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    filter_config = load_universe_filters(args.filters_config)

    if args.start_date:
        config.data.start_date = date.fromisoformat(args.start_date)
    if args.end_date:
        config.data.end_date = date.fromisoformat(args.end_date)

    setup_logging(config.logging.level, config.logging.log_dir)

    summary = run_ingest(config, filter_config, tickers_override=args.tickers)

    print(f"execution_date: {summary.execution_date}")
    print(f"universe: {summary.universe_static_count}")
    print(
        f"fetch success/partial/failed: "
        f"{summary.fetch_success}/{summary.fetch_partial}/{summary.fetch_failed}"
    )
    print(
        f"processed: {summary.processed_count}, "
        f"excluded_by_liquidity: {summary.excluded_by_liquidity}"
    )
    print(f"topix_available: {summary.topix_available}")
    print(f"duration: {summary.duration_seconds:.1f}s")
    if summary.errors:
        print(f"errors ({len(summary.errors)}):")
        for err in summary.errors[:20]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
