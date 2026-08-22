"""CLI entry point for Phase 6.5 Full Universe ingestion.

Usage:
    python scripts/run_universe_ingest.py --limit 100     # Stage 1: smoke test
    python scripts/run_universe_ingest.py --limit 500     # Stage 2: scale test
    python scripts/run_universe_ingest.py                 # Stage 3: full universe

Downloads (or reuses a cached copy of) the official JPX listed-issues
master file, applies the EXISTING static filters
(config.universe.segments/exclude_etf/exclude_reit), then fetches OHLCV
for each resulting ticker with checkpoint/resume - see
pipeline/universe_ingest.py.

--start-date/--end-date default to Phase 6's actual data range
(2022-01-04..2024-06-28, see README's "OOS期間の固定" section) rather
than config/settings.yaml's own start_date/end_date=null("today").
This is deliberate: config.data.end_date=None means "today", which
drifts forward every day this script is run and would silently grow the
Walk Forward window count beyond Phase 6's 5 committed windows (spec
section 6: "no window count changes"). Pinning the same end date Phase 6
already used keeps Stage 1/2/3 runs on the exact same OOS windows.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config, load_universe_filters  # noqa: E402
from pipeline.universe_ingest import run_universe_ingest  # noqa: E402
from storage.parquet_store import save_universe_snapshot  # noqa: E402
from universe.build import apply_static_filters  # noqa: E402
from universe.jpx_master import load_jpx_master  # noqa: E402

# Phase 6's actual data range (README "OOS期間の固定") - the base case
# for Phase 6.5's Full Universe expansion (spec section 5).
PHASE6_BASE_START_DATE = date(2022, 1, 4)
PHASE6_BASE_END_DATE = date(2024, 6, 28)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6.5 Full Universe ingestion")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--filters-config", type=Path, default=Path("config/universe_filters.yaml")
    )
    parser.add_argument(
        "--jpx-master-path", type=Path, default=Path("data/reference/jpx_master_current.xls"),
        help="Where to cache the downloaded JPX master file (reused on later runs)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/raw/_universe_fetch_manifest.json"),
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None,
        help="Override config.data.raw_dir (e.g. data/phase7/raw, to keep a later phase's "
        "fetch fully separate from this one's - see Phase 7 spec section 23)",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=None, help="Override config.data.processed_dir",
    )
    parser.add_argument(
        "--snapshot-name", type=str, default="phase6_5_full_universe",
        help="Label used for the saved Universe snapshot Parquet file",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, default=Path("data/processed/universe"),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N tickers (Stage 1/2 smoke/scale tests)",
    )
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=PHASE6_BASE_START_DATE,
    )
    parser.add_argument(
        "--end-date", type=date.fromisoformat, default=PHASE6_BASE_END_DATE,
    )
    parser.add_argument("--force-refetch", action="store_true")
    parser.add_argument("--force-refresh-master", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    config.data.start_date = args.start_date
    config.data.end_date = args.end_date
    if args.raw_dir is not None:
        config.data.raw_dir = args.raw_dir
    if args.processed_dir is not None:
        config.data.processed_dir = args.processed_dir
    filter_config = load_universe_filters(args.filters_config)
    setup_logging(config.logging.level, config.logging.log_dir)

    print(f"data period: {config.data.start_date} .. {config.data.end_date}")

    master = load_jpx_master(args.jpx_master_path, force_refresh=args.force_refresh_master)
    build_result = apply_static_filters(
        master,
        segments=config.universe.segments,
        exclude_etf=config.universe.exclude_etf,
        exclude_reit=config.universe.exclude_reit,
    )
    tickers = build_result.included["code"].tolist()

    print(f"JPX master candidates: {len(master)}")
    for segment, count in master["market_segment"].value_counts().items():
        print(f"  {segment}: {count}")
    print(f"Static filter included: {len(build_result.included)}")
    print(f"Static filter excluded: {len(build_result.excluded)}")

    save_universe_snapshot(build_result.included, args.snapshot_name, args.snapshot_dir)

    if args.limit is not None:
        tickers = tickers[: args.limit]
        print(f"--limit applied: processing first {len(tickers)} tickers")

    summary = run_universe_ingest(
        config, filter_config, tickers, args.manifest,
        force_refetch=args.force_refetch,
    )

    print(f"candidates this run: {summary.ticker_count}")
    print(f"attempted: {summary.attempted}, skipped (cached): {summary.skipped_cached}")
    print(
        f"fetch success/partial/failed: "
        f"{summary.fetch_success}/{summary.fetch_partial}/{summary.fetch_failed}"
    )
    print(f"processed (final universe): {summary.processed_count}")
    print(f"excluded by liquidity: {summary.excluded_by_liquidity}")
    print(f"TOPIX Proxy available: {summary.topix_available}")
    print(f"duration: {summary.duration_seconds:.1f}s")
    if summary.failed_tickers:
        print(
            f"failed tickers ({len(summary.failed_tickers)}):",
            ", ".join(summary.failed_tickers[:30]),
        )


if __name__ == "__main__":
    main()
