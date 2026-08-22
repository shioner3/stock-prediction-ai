"""Phase 12 CLI entry point: Signal Ensemble Validation over the SAME
data/phase7/ dataset already fetched (no new fetch needed). See
pipeline/run_phase12_ensemble.py's module docstring for the full design.

Usage:
    python scripts/run_phase12_ensemble.py
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
from forward_test.manifest import load_manifest_raw  # noqa: E402
from pipeline.run_phase8_analysis import ConfigMismatchError  # noqa: E402
from pipeline.run_phase12_ensemble import Phase12Report, run_phase12_ensemble  # noqa: E402
from pipeline.universe_ingest import load_manifest  # noqa: E402


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


def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 12 Signal Ensemble Validation")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--phase7-manifest", type=Path,
        default=Path("data/phase7/_universe_fetch_manifest.json"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/phase7/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/phase7/processed"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/phase7/features"))
    parser.add_argument("--signals-dir", type=Path, default=Path("data/phase7/signals"))
    parser.add_argument("--scores-dir", type=Path, default=Path("data/phase7/scores"))
    parser.add_argument(
        "--phase6-5-report", type=Path,
        default=Path("data/walk_forward/phase6_5_full_universe_report.json"),
    )
    parser.add_argument(
        "--phase7-report", type=Path,
        default=Path("data/walk_forward/phase7_report.json"),
    )
    parser.add_argument(
        "--strategy-manifest", type=Path,
        default=Path("data/forward_test/manifest.json"),
    )
    parser.add_argument(
        "--save-report", type=Path, default=Path("data/walk_forward/phase12_ensemble_report.json"),
    )
    return parser.parse_args()


def _print_summary(report: Phase12Report) -> None:
    print(f"config_check.matches: {report.config_check.matches}")
    print(f"integrity_hash_matches_strategy_v1: {report.integrity_hash_matches_strategy_v1}")
    if not report.integrity_hash_matches_strategy_v1:
        print(f"  mismatches: {report.integrity_hash_mismatches}")
    print(f"tickers: {len(report.tickers)}")
    print(f"total_trading_days: {report.total_trading_days}")

    print(f"\n{'bucket':<16} {'decision':<24} {'n':>8} {'freq%':>7} {'PF_high':>8} {'p_raw':>8}")
    for b in report.long_count_buckets + report.short_count_buckets + report.net_count_buckets:
        pf_high = (
            b.cost_metrics["high"].profit_factor
            if b.cost_metrics and b.cost_metrics["high"].profit_factor is not None
            else None
        )
        p_raw = b.permutation.p_value if b.permutation else None
        pf_str = f"{pf_high:.3f}" if pf_high is not None else "n/a"
        p_str = f"{p_raw:.4f}" if p_raw is not None else "n/a"
        print(
            f"{b.label:<16} {b.decision.value:<24} {b.n_sample:>8} "
            f"{b.frequency.pct_trading_days_with_occurrence * 100:>6.1f}% {pf_str:>8} {p_str:>8}"
        )

    n_long_sufficient = sum(c.combo.sufficient_sample for c in report.long_combinations)
    n_short_sufficient = sum(c.combo.sufficient_sample for c in report.short_combinations)
    print(f"\nLONG combinations (sufficient sample): {n_long_sufficient}")
    print(f"SHORT combinations (sufficient sample): {n_short_sufficient}")
    for c in (report.long_combinations + report.short_combinations)[:20]:
        if c.combo.sufficient_sample:
            print(
                f"  {c.combo.direction}:{'+'.join(c.combo.signals)} "
                f"n={c.combo.n_occurrences} decision={c.decision.value}"
            )

    if report.top5_simulation:
        s = report.top5_simulation
        print(
            f"\nTop-5 simulation: n_trades={s.n_trades} total_return={s.total_return} "
            f"CAGR={s.cagr} Sharpe={s.sharpe} max_drawdown={s.max_drawdown}"
        )


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    config.data.raw_dir = args.raw_dir
    config.data.processed_dir = args.processed_dir
    config.data.features_dir = args.features_dir
    config.data.signals_dir = args.signals_dir
    config.data.scores_dir = args.scores_dir
    setup_logging(config.logging.level, config.logging.log_dir)

    manifest = load_manifest(args.phase7_manifest)
    entries: dict = manifest.get("tickers", {})
    tickers = sorted(t for t, e in entries.items() if e.get("included_in_universe"))
    print(f"tickers: {len(tickers)}")
    if not tickers:
        print("No tickers found - run scripts/run_universe_ingest.py / run_phase7_report.py first.")
        return

    if not args.strategy_manifest.exists():
        print(f"No Forward Test Strategy manifest at {args.strategy_manifest} - "
              "run Phase 10/11 first.")
        return
    strategy_manifest = load_manifest_raw(args.strategy_manifest)

    try:
        report = run_phase12_ensemble(
            config, tickers, args.phase6_5_report, args.phase7_report, strategy_manifest,
        )
    except ConfigMismatchError as exc:
        print(f"CONFIG_MISMATCH: {exc}")
        sys.exit(1)

    if not report.integrity_hash_matches_strategy_v1:
        print(f"INTEGRITY_HASH_MISMATCH: {report.integrity_hash_mismatches}")
        print("Results below are computed but should be treated as INVALID per spec section 34.")

    _print_summary(report)
    _save_json(report, args.save_report)
    print(f"\nsaved: {args.save_report}")


if __name__ == "__main__":
    main()
