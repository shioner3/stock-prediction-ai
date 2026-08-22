"""Phase 8 CLI entry point: BEAR regime reproducibility verification for
long_oversold_rebound, over the SAME data/phase7/ dataset Phase 7 already
fetched (it spans 2022-01-04..2026-08-20 continuously - no new fetch is
needed for Phase 8). See pipeline/run_phase8_analysis.py's module
docstring for the full design rationale.

Usage:
    python scripts/run_phase8_analysis.py
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
from pipeline.run_phase8_analysis import (  # noqa: E402
    ConfigMismatchError,
    Phase8Report,
    run_phase8_analysis,
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
    parser = argparse.ArgumentParser(description="Run Phase 8 BEAR reproducibility analysis")
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
    parser.add_argument("--target-direction", type=str, default="LONG")
    parser.add_argument("--target-signal-name", type=str, default="long_oversold_rebound")
    parser.add_argument(
        "--save-report", type=Path, default=Path("data/walk_forward/phase8_report.json"),
    )
    return parser.parse_args()


def _print_summary(report: Phase8Report) -> None:
    t = report.target
    print(f"config_check.matches: {report.config_check.matches}")
    print(f"windows (combined, all): {len(report.windows)}")
    print(f"BEAR episodes (full period): {len(report.bear_episodes_all)}")
    for ep in report.bear_episodes_all:
        print(f"  episode {ep.index}: {ep.start_date}..{ep.end_date} ({ep.trading_days} days)")

    print(f"\n--- Target: {t.direction}:{t.signal_name} ---")
    base = t.combined.oos_metrics_by_cost_tier.get("base")
    if base is not None:
        print(f"Combined OOS: n={base.n_trades} PF={base.profit_factor} exp={base.expectancy}")
    print(f"Combined decision: {t.combined.decision.value}")

    print("\nBEAR x Window:")
    for rw in t.bear_by_window:
        print(f"  window {rw.window_index}: n={rw.metrics.n_trades} PF={rw.metrics.profit_factor}")

    print("\nBEAR x Year:")
    for year, ry in sorted(t.bear_by_year.items()):
        if ry.metrics is None:
            print(f"  {year}: NO_BEAR_DATA")
        else:
            print(f"  {year}: n={ry.metrics.n_trades} PF={ry.metrics.profit_factor}")

    print("\nBEAR episodes (this signal):")
    for em in t.bear_episodes:
        print(
            f"  episode {em.episode.index} ({em.episode.start_date}..{em.episode.end_date}): "
            f"n={em.metrics.n_trades} PF={em.metrics.profit_factor} "
            f"cum_return={em.cumulative_return}"
        )

    print(f"\nBEAR concentration: n_tickers={t.bear_concentration.n_tickers}")
    for k, share in sorted(t.bear_top_k_shares.items()):
        print(f"  top{k}: trade_share={share.trade_share} return_share={share.return_share}")

    print("\nBEAR cost sensitivity:")
    for tier, m in t.bear_cost_sensitivity.items():
        print(f"  {tier}: PF={m.profit_factor} exp={m.expectancy}")

    print("\nBEAR bootstrap:")
    for name, b in t.bear_bootstrap.items():
        print(f"  {name}: point={b.point_estimate} CI=[{b.ci_low}, {b.ci_high}]")

    if t.bear_permutation is not None:
        print(f"\nBEAR permutation p={t.bear_permutation.p_value}")
    else:
        print("\nBEAR permutation: n/a (no BEAR-classified signal returns)")

    print("\nLeave-One-Period-Out (by year):")
    for r in t.lopo_by_year:
        print(
            f"  {r.period_label}: full_pf={r.full_sample_pf} "
            f"loo_pf={r.leave_one_out_pf} -> {r.classification}"
        )

    print("\nLeave-One-Period-Out (by BEAR episode):")
    for r in t.lopo_by_episode:
        print(
            f"  episode {r.period_label}: full_pf={r.full_sample_pf} "
            f"loo_pf={r.leave_one_out_pf} -> {r.classification}"
        )

    p = t.placebo
    print(
        f"\nPlacebo (shift={p.shift_trading_days} trading days): "
        f"real_bear_pf={p.real_bear_metrics.profit_factor} "
        f"placebo_bear_pf={p.placebo_bear_metrics.profit_factor} "
        f"(n_real={p.real_bear_metrics.n_trades}, n_placebo={p.placebo_bear_metrics.n_trades})"
    )

    print("\nOther 11 signals (combined):")
    for s in report.other_signals:
        print(
            f"  {s.direction}:{s.signal_name}: combined_pf={s.combined_pf_base} "
            f"p6.5_pf={s.phase6_5_pf_base} p7_pf={s.phase7_pf_base} decision={s.decision_combined}"
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

    try:
        report = run_phase8_analysis(
            config, tickers, args.phase6_5_report, args.phase7_report,
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
