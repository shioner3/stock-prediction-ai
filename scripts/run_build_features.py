"""CLI entry point for the Phase 2/3 feature-building pipeline.

Usage:
    python scripts/run_build_features.py
    python scripts/run_build_features.py --tickers 7203 6758
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging
from config.loader import load_app_config
from pipeline.build_features import run_build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2/3 feature-building pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Only build features for these tickers (default: everything in data/processed/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    setup_logging(config.logging.level, config.logging.log_dir)

    summary = run_build_features(config, tickers=args.tickers)

    print(f"tickers: {summary.ticker_count}")
    print(f"success: {summary.success_count}")
    print(f"failed: {len(summary.failed_tickers)}")
    if summary.failed_tickers:
        print("failed tickers:", ", ".join(summary.failed_tickers))
    print(f"market_data_available (Relative Strength): {summary.market_data_available}")
    print(f"duration: {summary.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
