"""Phase 9 CLI entry point: robustness/sensitivity checks for
long_oversold_rebound, over the SAME data/phase7/ dataset Phase 7/8
already fetched (no new fetch needed). See
pipeline/run_phase9_analysis.py's module docstring for the full design.

Usage:
    python scripts/run_phase9_analysis.py
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
from config.loader import load_app_config, load_phase9_config  # noqa: E402
from pipeline.run_phase8_analysis import ConfigMismatchError  # noqa: E402
from pipeline.run_phase9_analysis import Phase9Report, run_phase9_analysis  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Run Phase 9 robustness analysis")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--phase9-config", type=Path, default=Path("config/phase9_settings.yaml"),
    )
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
        "--jpx-master-path", type=Path, default=Path("data/reference/jpx_master_current.xls"),
    )
    parser.add_argument(
        "--phase6-5-report", type=Path,
        default=Path("data/walk_forward/phase6_5_full_universe_report.json"),
    )
    parser.add_argument(
        "--phase7-report", type=Path,
        default=Path("data/walk_forward/phase7_report.json"),
    )
    parser.add_argument("--target-direction", type=str, default="LONG")
    parser.add_argument("--target-signal-name", type=str, default="long_oversold_rebound")
    parser.add_argument(
        "--save-report", type=Path, default=Path("data/walk_forward/phase9_report.json"),
    )
    return parser.parse_args()


def _print_summary(report: Phase9Report) -> None:
    print(f"config_check.matches: {report.config_check.matches}")
    print(f"n_signal_trades_total: {report.n_signal_trades_total}")
    print(f"BEAR episodes: {len(report.bear_episodes)}")

    print("\n--- Rich BEAR episode metrics ---")
    for e in report.rich_episode_metrics:
        print(
            f"  ep{e.episode.index} {e.episode.start_date}..{e.episode.end_date}: "
            f"n={e.metrics.n_trades} tickers={e.unique_tickers} PF={e.metrics.profit_factor} "
            f"median={e.median_trade_return} max_dd={e.max_drawdown} "
            f"pnl_share={e.pnl_contribution_share}"
        )

    print("\n--- Leave-One-BEAR-Episode-Out ---")
    for lopo_ep in report.lopo_by_episode:
        print(
            f"  episode {lopo_ep.period_label}: removed={lopo_ep.n_trades_removed} "
            f"full_pf={lopo_ep.full_sample_metrics.profit_factor} "
            f"remaining_pf={lopo_ep.remaining_metrics.profit_factor} "
            f"remaining_exp_ci=[{lopo_ep.remaining_bootstrap_expectancy.ci_low:.5f}, "
            f"{lopo_ep.remaining_bootstrap_expectancy.ci_high:.5f}]"
        )

    print("\n--- Leave-One-Year-Out ---")
    for lopo_yr in report.lopo_by_year:
        print(
            f"  year {lopo_yr.period_label}: removed={lopo_yr.n_trades_removed} "
            f"full_pf={lopo_yr.full_sample_metrics.profit_factor} "
            f"remaining_pf={lopo_yr.remaining_metrics.profit_factor}"
        )

    dc = report.day_concentration_bear
    print(f"\n--- Day concentration (BEAR, n_days={dc.n_days}) ---")
    print(f"  pnl_share_by_k: {dc.pnl_share_by_k}")
    print(f"  trade_share_by_k: {dc.trade_share_by_k}")
    print(f"  gini: {dc.gini_coefficient}")

    print("\n--- Day Cluster Bootstrap (BEAR) ---")
    for dc_name, dc_result in report.day_cluster_bootstrap_bear.items():
        print(
            f"  {dc_name}: point={dc_result.point_estimate} "
            f"CI=[{dc_result.ci_low}, {dc_result.ci_high}] n_days={dc_result.n_days}"
        )

    print("\n--- Block Bootstrap (BEAR) ---")
    for bb_name, bb_result in report.block_bootstrap_bear.items():
        print(
            f"  {bb_name}: point={bb_result.point_estimate} "
            f"CI=[{bb_result.ci_low}, {bb_result.ci_high}]"
        )

    print("\n--- Timing Offset Sweep (BEAR) ---")
    for timing in report.timing_offset_sweep:
        print(
            f"  offset={timing.offset_days}: n_bear={timing.n_trades_bear} "
            f"PF={timing.metrics_bear.profit_factor} exp={timing.metrics_bear.expectancy}"
        )

    print("\n--- Sector breakdown (BEAR) ---")
    if not report.sector_breakdown_bear:
        print("  NOT_AVAILABLE")
    for sector, sector_m in report.sector_breakdown_bear.items():
        print(f"  {sector}: n={sector_m.n_trades} PF={sector_m.profit_factor}")

    print("\n--- Liquidity breakdown (BEAR) ---")
    if not report.liquidity_breakdown_bear:
        print("  NOT_AVAILABLE")
    for bucket, bucket_m in report.liquidity_breakdown_bear.items():
        print(f"  {bucket}: n={bucket_m.n_trades} PF={bucket_m.profit_factor}")

    print("\n--- Cost stress ---")
    for label, stress in (
        ("Combined", report.cost_stress_combined),
        ("BEAR", report.cost_stress_bear),
        ("Aug-2024 episode", report.cost_stress_aug2024_episode),
        ("BEAR excl. Aug-2024", report.cost_stress_bear_excl_aug2024),
    ):
        print(f"  {label}: " + ", ".join(f"{t}={m.profit_factor}" for t, m in stress.items()))

    print("\n--- Forward Holding Period profile ---")
    for horizon in report.forward_horizon_profile:
        print(
            f"  {horizon.horizon_days}d: n={horizon.n} mean={horizon.mean_return} "
            f"median={horizon.median_return} win_rate={horizon.win_rate} "
            f"CI=[{horizon.bootstrap.ci_low}, {horizon.bootstrap.ci_high}]"
        )

    print("\n--- Scenarios ---")
    for scenario in report.scenarios:
        print(
            f"  {scenario.name} ({scenario.description}): "
            f"n={scenario.metrics.n_trades} PF={scenario.metrics.profit_factor}"
        )


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    config.data.raw_dir = args.raw_dir
    config.data.processed_dir = args.processed_dir
    config.data.features_dir = args.features_dir
    config.data.signals_dir = args.signals_dir
    config.data.scores_dir = args.scores_dir
    phase9_config = load_phase9_config(args.phase9_config)
    setup_logging(config.logging.level, config.logging.log_dir)

    manifest = load_manifest(args.phase7_manifest)
    entries: dict = manifest.get("tickers", {})
    tickers = sorted(t for t, e in entries.items() if e.get("included_in_universe"))
    print(f"tickers: {len(tickers)}")
    if not tickers:
        print("No tickers found - run scripts/run_phase7_report.py first.")
        return

    try:
        report = run_phase9_analysis(
            config, phase9_config, tickers, args.phase6_5_report, args.phase7_report,
            args.jpx_master_path,
            target_direction=args.target_direction, target_signal_name=args.target_signal_name,
        )
    except ConfigMismatchError as exc:
        print(f"CONFIG_MISMATCH: {exc}")
        sys.exit(1)

    _print_summary(report)
    _save_json(report, args.save_report)
    print(f"\nsaved: {args.save_report}")


if __name__ == "__main__":
    main()
