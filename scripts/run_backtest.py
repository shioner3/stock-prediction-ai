"""CLI entry point for the Phase 4 backtest.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --tickers 7203 6758
    python scripts/run_backtest.py --save-trades data/backtest/trades.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.metrics import compute_metrics_by_signal
from common.logging_setup import setup_logging
from config.loader import load_app_config
from pipeline.run_backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 backtest")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Only backtest these tickers (default: everything in data/signals/)",
    )
    parser.add_argument(
        "--save-trades", type=Path, default=None,
        help="Optional path to save the combined Trade Record table as Parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    setup_logging(config.logging.level, config.logging.log_dir)

    summary = run_backtest(config, tickers=args.tickers)

    print(f"tickers: {summary.ticker_count}")
    print(f"success: {summary.success_count}")
    print(f"failed: {len(summary.failed_tickers)}")
    if summary.failed_tickers:
        print("failed tickers:", ", ".join(summary.failed_tickers))
    print(f"trades: {len(summary.trades)}")
    print(f"skipped signals: {len(summary.skipped)}")
    print(f"duration: {summary.duration_seconds:.1f}s")

    if args.save_trades and not summary.trades.empty:
        args.save_trades.parent.mkdir(parents=True, exist_ok=True)
        summary.trades.to_parquet(args.save_trades, engine="pyarrow", index=False)
        print(f"trades saved to: {args.save_trades}")

    print()
    print("--- metrics by signal ---")
    metrics_by_signal = compute_metrics_by_signal(summary.trades)
    for signal_name in sorted(metrics_by_signal):
        m = metrics_by_signal[signal_name]
        print(
            f"{signal_name}: n={m.n_trades} win_rate={m.win_rate:.1%} "
            f"avg_return={m.average_return:.2%} median_return={m.median_return:.2%} "
            f"total_return={m.total_return:.2%} profit_factor={m.profit_factor} "
            f"expectancy={m.expectancy:.2%}"
        )
    if not metrics_by_signal:
        print("(no trades)")


if __name__ == "__main__":
    main()
