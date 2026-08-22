"""CLI entry point for Phase 6 Walk Forward + Statistical Validation.

Usage:
    python scripts/run_walk_forward.py
    python scripts/run_walk_forward.py --tickers 7203 6758

Output is a report per Signal; the ACCEPT_CANDIDATE/REJECT/
INSUFFICIENT_EVIDENCE label is a SUGGESTION for human review, never an
automatic action - see backtest/decision.py.
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
from config.loader import load_app_config  # noqa: E402
from pipeline.run_walk_forward import WalkForwardReport, run_walk_forward  # noqa: E402


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


def _save_report(report: WalkForwardReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(report)
    path.write_text(
        json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 Walk Forward validation")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument(
        "--save-report", type=Path, default=None,
        help="Optional path to save the full report as JSON (includes config_hash/data_hash)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    setup_logging(config.logging.level, config.logging.log_dir)

    report = run_walk_forward(config, tickers=args.tickers)

    print(f"tickers: {report.tickers}")
    print(f"config_hash: {report.config_hash}")
    print(f"data_hash: {report.data_hash}")
    print(f"windows: {len(report.windows)}")
    for w in report.windows:
        print(
            f"  window {w.index}: TRAIN {w.train_start}..{w.train_end} | "
            f"VAL {w.validation_start}..{w.validation_end} | "
            f"OOS {w.oos_start}..{w.oos_end}"
            f"{' (truncated)' if w.oos_truncated else ''}"
        )

    print("\n--- Signal results (OOS, base cost tier) ---")
    for r in report.signal_results:
        base = r.oos_metrics_by_cost_tier.get("base")
        high = r.oos_metrics_by_cost_tier.get("high")
        exp_ci = r.bootstrap.get("expectancy")
        print(
            f"\n{r.direction} {r.signal_name}: decision={r.decision.value}"
        )
        if base is not None:
            print(
                f"  OOS(base): n={base.n_trades} win_rate={base.win_rate} "
                f"PF={base.profit_factor} expectancy={base.expectancy}"
            )
        if high is not None:
            print(f"  OOS(high cost): expectancy={high.expectancy}")
        if exp_ci is not None:
            print(
                f"  bootstrap expectancy 95% CI: [{exp_ci.ci_low:.4f}, {exp_ci.ci_high:.4f}]"
                if exp_ci.n_observations > 0
                else "  bootstrap: n/a (no OOS trades)"
            )
        if r.permutation is not None:
            print(
                f"  permutation p-value: {r.permutation.p_value} "
                f"(n_signal={r.permutation.n_signal}, n_population={r.permutation.n_population})"
            )
        print(
            f"  window consistency: PF>1 in "
            f"{r.windows_with_positive_pf}/{r.windows_evaluated} windows"
        )
        if r.regime_metrics:
            regimes = ", ".join(
                f"{name}:n={m.n_trades},exp={m.expectancy}" for name, m in r.regime_metrics.items()
            )
            print(f"  by regime: {regimes}")
        c = r.concentration
        if c.n_trades > 0:
            print(
                f"  concentration: {c.n_tickers} tickers, top1/5/10 trade share="
                f"{c.top1_ticker_trade_share:.2f}/{c.top5_ticker_trade_share:.2f}/"
                f"{c.top10_ticker_trade_share:.2f}, top1/5/10 return share="
                f"{c.top1_ticker_return_share}/{c.top5_ticker_return_share}/"
                f"{c.top10_ticker_return_share}"
            )

    if report.fdr_results:
        print("\n--- Multiple testing: Benjamini-Hochberg FDR (informational only) ---")
        for key, fdr in sorted(report.fdr_results.items(), key=lambda kv: kv[1].rank):
            print(
                f"  {key}: raw_p={fdr.raw_p_value:.4f} adjusted_p={fdr.adjusted_p_value:.4f} "
                f"rank={fdr.rank}/{fdr.n_tests} significant={fdr.significant}"
            )

    if args.save_report:
        _save_report(report, args.save_report)
        print(f"\nfull report saved to: {args.save_report}")


if __name__ == "__main__":
    main()
