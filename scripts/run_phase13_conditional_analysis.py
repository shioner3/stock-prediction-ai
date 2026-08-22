"""Phase 13 CLI entry point: long_oversold_rebound Conditional Analysis
over the SAME data/phase7/ dataset already fetched (no new fetch
needed). See pipeline/run_phase13_conditional_analysis.py's module
docstring for the full design.

Usage:
    python scripts/run_phase13_conditional_analysis.py
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
from pipeline.run_phase8_analysis import ConfigMismatchError  # noqa: E402
from pipeline.run_phase13_conditional_analysis import (  # noqa: E402
    Phase13Report,
    run_phase13_conditional_analysis,
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
        description="Run Phase 13 long_oversold_rebound Conditional Analysis"
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
        "--save-report", type=Path,
        default=Path("data/walk_forward/phase13_conditional_report.json"),
    )
    return parser.parse_args()


def _print_bucket_summary(title: str, buckets) -> None:
    print(f"\n--- {title} ---")
    for b in buckets:
        base = b.cost_metrics.get("base") if b.cost_metrics else None
        pf = base.profit_factor if base else None
        exp = base.expectancy if base else None
        p = b.permutation.p_value if b.permutation else None
        n5d = next((s for s in b.forward_return_stats if s.window_days == 5), None)
        mean5d = n5d.mean_return if n5d else None
        print(
            f"  {b.label:<24} n={b.n:>6} mean5d={mean5d} PF={pf} exp={exp} p={p}"
        )


def _print_summary(report: Phase13Report) -> None:
    print(f"config_check.matches: {report.config_check.matches}")
    print(f"n_signal_total: {report.n_signal_total}")
    print(f"unique_tickers: {report.unique_tickers}")
    print(f"unique_signal_dates: {report.unique_signal_dates}")

    print("\n--- Overall Forward Return ---")
    for s in report.overall_forward_return_stats:
        print(
            f"  {s.window_days}d: n={s.n} mean={s.mean_return} "
            f"median={s.median_return} win_rate={s.win_rate}"
        )

    _print_bucket_summary("Market Regime", report.regime_buckets)
    _print_bucket_summary("Market Drawdown (TOPIX 20d)", report.market_drawdown_20d_buckets)
    _print_bucket_summary("Stock Drawdown (return_20d)", report.stock_drawdown_20d_buckets)
    _print_bucket_summary("MA20 Deviation", report.ma20_deviation_buckets)
    _print_bucket_summary("Volume", report.volume_buckets)
    _print_bucket_summary("Volatility (tercile)", report.volatility_buckets)
    _print_bucket_summary("Signal Count (other 11)", report.signal_count_buckets)
    _print_bucket_summary("LONG/SHORT Consensus", report.long_short_consensus_buckets)
    _print_bucket_summary("Regime x Score", report.regime_x_score)
    _print_bucket_summary("Market Drawdown x Score", report.drawdown_x_score)

    print("\n--- Score ---")
    for b in report.score_analysis.buckets:
        print(f"  {b.bucket}: n={b.n} mean5d={b.mean_return_5d} PF={b.metrics.profit_factor}")
    print(
        f"  monotonic={report.score_analysis.monotonic} "
        f"rank_corr={report.score_analysis.rank_correlation}"
    )
    print(f"  Q5-Q1 spread={report.score_analysis.q5_q1_spread}")

    print("\n--- Event Exclusion (base cost) ---")
    print(f"  A (full): n={report.event_case_a.n_trades} PF={report.event_case_a.profit_factor}")
    print(
        f"  B (excl Aug2024): n={report.event_case_b_excl_aug2024.n_trades} "
        f"PF={report.event_case_b_excl_aug2024.profit_factor}"
    )
    print(
        f"  C (excl 2024): n={report.event_case_c_excl_2024.n_trades} "
        f"PF={report.event_case_c_excl_2024.profit_factor}"
    )
    print(
        f"  D (BEAR excl Aug2024): n={report.event_case_d_bear_excl_aug2024.n_trades} "
        f"PF={report.event_case_d_bear_excl_aug2024.profit_factor}"
    )

    print(f"\n--- BEAR Episodes ({len(report.bear_episodes)}) ---")
    for em in report.bear_episodes:
        print(
            f"  ep{em.episode.index} {em.episode.start_date}..{em.episode.end_date}: "
            f"n={em.metrics.n_trades} PF={em.metrics.profit_factor} "
            f"pnl_share={em.pnl_contribution_share}"
        )

    print(f"\nFDR-tested units: {len(report.fdr_tested_p_values)}")


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

    try:
        report = run_phase13_conditional_analysis(
            config, tickers, args.phase6_5_report, args.phase7_report,
        )
    except ConfigMismatchError as exc:
        print(f"CONFIG_MISMATCH: {exc}")
        sys.exit(1)

    _print_summary(report)
    _save_json(report, args.save_report)
    print(f"\nsaved: {args.save_report}")


if __name__ == "__main__":
    main()
