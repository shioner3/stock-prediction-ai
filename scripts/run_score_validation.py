"""CLI entry point for Phase 5 Score Validation (bucket analysis).

Usage:
    python scripts/run_score_validation.py
    python scripts/run_score_validation.py --bucket-method quantile
    python scripts/run_score_validation.py --forward-window 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging
from config.loader import load_app_config
from pipeline.run_score_validation import run_score_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 Score Validation")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument(
        "--bucket-method", choices=["fixed", "quantile"], default="fixed",
        help="Fixed 0-100 buckets, or population-quantile buckets",
    )
    parser.add_argument(
        "--forward-window", type=int, default=None,
        help="Only print this forward window in days (default: all of 1/3/5/7/10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    setup_logging(config.logging.level, config.logging.log_dir)

    results = run_score_validation(
        config, tickers=args.tickers, bucket_method=args.bucket_method
    )

    if not results:
        print("(no scored signals found - run scripts/run_build_scores.py first)")
        return

    for r in results:
        if args.forward_window is not None and r.forward_window != args.forward_window:
            continue
        print(
            f"\n{r.direction} {r.signal_name} | forward={r.forward_window}d "
            f"| bucket_method={r.bucket_method} | n={r.n} "
            f"| monotonic={r.monotonic} | monotonicity_corr={r.monotonicity_corr}"
        )
        for bucket, m in r.bucket_metrics.items():
            if m.n_trades == 0:
                print(f"  {bucket}: n=0")
                continue
            print(
                f"  {bucket}: n={m.n_trades} mean_return={m.average_return:.2%} "
                f"median_return={m.median_return:.2%} win_rate={m.win_rate:.1%} "
                f"profit_factor={m.profit_factor} expectancy={m.expectancy:.2%}"
            )


if __name__ == "__main__":
    main()
