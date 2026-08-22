"""Phase 14 CLI entry point: long_oversold_rebound Conditional Edge
Independent Validation over the SAME data/phase7/ dataset already
fetched (no new fetch needed). See
pipeline/run_phase14_validation.py's module docstring for the full
design.

Usage:
    python scripts/run_phase14_validation.py
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
from pipeline.run_phase14_validation import (  # noqa: E402
    Phase14Report,
    StrategyHashMismatchError,
    run_phase14_validation,
)
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
    parser = argparse.ArgumentParser(
        description="Run Phase 14 long_oversold_rebound Conditional Edge Independent Validation"
    )
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
        "--phase13-report", type=Path,
        default=Path("data/walk_forward/phase13_conditional_report.json"),
    )
    parser.add_argument(
        "--strategy-manifest", type=Path,
        default=Path("data/forward_test/manifest.json"),
    )
    parser.add_argument(
        "--preregistration-path", type=Path,
        default=Path("data/walk_forward/phase14_preregistration.json"),
    )
    parser.add_argument(
        "--save-report", type=Path,
        default=Path("data/walk_forward/phase14_validation_report.json"),
    )
    return parser.parse_args()


def _print_bucket_summary(title: str, buckets) -> None:
    print(f"\n--- {title} ---")
    for b in buckets:
        base = b.cost_metrics.get("base") if b.cost_metrics else None
        high = b.cost_metrics.get("high") if b.cost_metrics else None
        p = b.permutation.p_value if b.permutation else None
        n5d = next((s for s in b.forward_return_stats if s.window_days == 5), None)
        mean5d = n5d.mean_return if n5d else None
        print(
            f"  {b.label:<16} n={b.n:>6} mean5d={mean5d} "
            f"PF_base={base.profit_factor if base else None} "
            f"PF_high={high.profit_factor if high else None} "
            f"exp={base.expectancy if base else None} p={p}"
        )


def _print_summary(report: Phase14Report) -> None:
    print(f"config_check.matches: {report.config_check.matches}")
    print(f"strategy_hash_matches: {report.strategy_hash_matches}")
    print(f"n_core_condition_trades: {report.n_core_condition_trades}")

    print("\n--- Core / Control condition ---")
    print(f"  core:    n={report.core_condition_bucket.n}")
    print(f"  control: n={report.control_condition_bucket.n}")

    _print_bucket_summary("Pre-registered TOPIX 20d threshold grid", report.threshold_grid)
    _print_bucket_summary("Regime x Dose-response", report.regime_x_dose_response)
    _print_bucket_summary("Dose-response (mutually exclusive)", report.dose_response)

    print(f"\n--- BEAR Episodes ({len(report.bear_episodes)}) ---")
    for ea in report.bear_episodes:
        ep = ea.rich_metrics.episode
        print(
            f"  {ep.start_date}..{ep.end_date} major={ea.is_major} "
            f"n={ea.rich_metrics.metrics.n_trades} PF={ea.rich_metrics.metrics.profit_factor} "
            f"mfe5d={ea.mean_mfe_5d} mae5d={ea.mean_mae_5d}"
        )
    print(f"  2025-04 episode: {report.apr_2025_episode}")

    print(f"\n--- Leave-One-Episode-Out ({len(report.loeo)} major episodes) ---")
    for r in report.loeo:
        print(
            f"  excl {r.period_label}: n_removed={r.n_trades_removed} "
            f"remaining_n={r.remaining_metrics.n_trades} exp={r.remaining_metrics.expectancy} "
            f"ci_low={r.remaining_bootstrap_expectancy.ci_low}"
        )

    print(f"\n--- Leave-One-Year-Out ({len(report.loyo)} years) ---")
    for r in report.loyo:
        print(
            f"  excl {r.period_label}: n_removed={r.n_trades_removed} "
            f"remaining_n={r.remaining_metrics.n_trades} exp={r.remaining_metrics.expectancy}"
        )

    print("\n--- Event Exclusion ---")
    for label, exclusion in (
        ("2024-08", report.exclude_aug2024),
        ("2025-04", report.exclude_apr2025),
        ("both", report.exclude_both),
    ):
        print(
            f"  excl {label}: before={exclusion.n_trades_before} "
            f"after={exclusion.n_trades_after} positive={exclusion.positive} "
            f"exp={exclusion.metrics_after.expectancy}"
        )

    print(f"\n--- Timing Placebo ({len(report.timing_placebo)} offsets) ---")
    for offset_result in report.timing_placebo:
        print(
            f"  offset={offset_result.offset_days:+d}d n={offset_result.n_core_trades} "
            f"exp={offset_result.metrics.expectancy} core_positive={offset_result.core_positive}"
        )

    print("\n--- Score Independence (within core condition) ---")
    for b in report.score_independence_core.buckets:
        print(f"  {b.bucket}: n={b.n} mean5d={b.mean_return_5d} PF={b.metrics.profit_factor}")
    print(
        f"  monotonic={report.score_independence_core.monotonic} "
        f"rank_corr={report.score_independence_core.rank_correlation} "
        f"Q5-Q1={report.score_independence_core.q5_q1_spread}"
    )

    print("\n--- Forward Horizon: BEAR vs NON-BEAR ---")
    for horizon in report.forward_horizon_comparison:
        print(
            f"  {horizon.window_days}d: BEAR mean={horizon.bear.mean_return} "
            f"n={horizon.bear.n} | NON-BEAR mean={horizon.non_bear.mean_return} "
            f"n={horizon.non_bear.n}"
        )

    print("\n--- Cost Sensitivity ---")
    for cost_result in report.cost_sensitivity:
        tiers_str = " ".join(
            f"{tier}:PF={m.profit_factor}" for tier, m in cost_result.metrics_by_tier.items()
        )
        print(f"  {cost_result.label}: {tiers_str}")

    bb = report.bootstrap_battery
    print("\n--- Bootstrap Battery (core condition, expectancy) ---")
    print(f"  trade_level:    {bb.trade_level.ci_low} .. {bb.trade_level.ci_high}")
    print(f"  day_cluster:    {bb.day_cluster.ci_low} .. {bb.day_cluster.ci_high}")
    print(f"  block:          {bb.block.ci_low} .. {bb.block.ci_high}")
    print(f"  ticker_cluster: {bb.ticker_cluster.ci_low} .. {bb.ticker_cluster.ci_high}")

    print(f"\n--- Permutation ({len(report.permutation_battery)} tests) + FDR ---")
    for key, fdr in sorted(report.fdr_results.items(), key=lambda kv: kv[1].raw_p_value):
        print(f"  {key:<24} raw_p={fdr.raw_p_value:.4f} adj_p={fdr.adjusted_p_value:.4f}")

    print(f"\n=== Decision: {report.decision.primary.value} ===")
    if report.decision.secondary:
        print(f"    Secondary: {[d.value for d in report.decision.secondary]}")


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
        print(
            f"No Forward Test Strategy manifest at {args.strategy_manifest} - "
            "run scripts/run_forward_test.py's initialization first."
        )
        sys.exit(1)
    strategy_manifest = load_manifest_raw(args.strategy_manifest)

    try:
        report = run_phase14_validation(
            config, tickers, args.phase6_5_report, args.phase7_report,
            args.phase13_report, strategy_manifest,
            preregistration_path=args.preregistration_path,
        )
    except ConfigMismatchError as exc:
        print(f"CONFIG_MISMATCH: {exc}")
        sys.exit(1)
    except StrategyHashMismatchError as exc:
        print(f"{exc}")
        sys.exit(1)

    _print_summary(report)
    _save_json(report, args.save_report)
    print(f"\nsaved: {args.save_report}")
    print(f"preregistration: {args.preregistration_path}")


if __name__ == "__main__":
    main()
