"""Phase 7 CLI entry point: Feature/Signal/Score build + Walk Forward
(scoped to OOS windows starting on/after 2024-07-01) + Data Integrity
report + Phase 6.5-vs-Phase 7 comparison, over a SEPARATE dataset under
data/phase7/ - see the module docstring in pipeline/run_walk_forward.py
(min_oos_start) and scripts/run_universe_ingest.py (--raw-dir/
--processed-dir) for why a fully separate dataset is fetched rather than
appending to Phase 6.5's data/raw, data/processed, data/features, etc.

Why the fetch range starts at 2022-01-04 (Phase 6.5's own anchor) rather
than 2024-07-01: a Walk Forward window whose OOS starts 2024-07-01 needs
15 months of REAL prior data (12mo TRAIN + 3mo VALIDATION) to be built
without look-ahead - see pipeline/run_walk_forward.py's min_oos_start
docstring. Reusing the same anchor also means the Universe's dynamic
liquidity filter (universe/filters.py, .head()-based) evaluates the SAME
early-2022 basis Phase 6.5 used, keeping the Universe-membership decision
mechanism identical and non-leaking rather than shifting to a
2024-07-based liquidity snapshot for no reason.

Usage:
    python scripts/run_universe_ingest.py --start-date 2022-01-04 \
        --end-date 2026-08-20 --raw-dir data/phase7/raw \
        --processed-dir data/phase7/processed \
        --manifest data/phase7/_universe_fetch_manifest.json \
        --snapshot-name phase7_full_universe \
        --snapshot-dir data/phase7/processed/universe
    python scripts/run_phase7_report.py
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from pipeline.build_features import run_build_features  # noqa: E402
from pipeline.build_scores import run_build_scores  # noqa: E402
from pipeline.build_signals import run_build_signals  # noqa: E402
from pipeline.data_integrity import build_data_integrity_report  # noqa: E402
from pipeline.phase_comparison import compare_reports  # noqa: E402
from pipeline.run_walk_forward import WalkForwardReport, run_walk_forward  # noqa: E402
from pipeline.universe_ingest import load_manifest  # noqa: E402
from universe.build import apply_static_filters  # noqa: E402
from universe.jpx_master import load_jpx_master  # noqa: E402

PHASE7_OOS_START = date(2024, 7, 1)
PHASE7_FETCH_START = date(2022, 1, 4)


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
    parser = argparse.ArgumentParser(description="Run Phase 7 independent OOS report")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--jpx-master-path", type=Path, default=Path("data/reference/jpx_master_current.xls"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/phase7/_universe_fetch_manifest.json"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/phase7/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/phase7/processed"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/phase7/features"))
    parser.add_argument("--signals-dir", type=Path, default=Path("data/phase7/signals"))
    parser.add_argument("--scores-dir", type=Path, default=Path("data/phase7/scores"))
    parser.add_argument(
        "--save-walk-forward-report", type=Path,
        default=Path("data/walk_forward/phase7_report.json"),
    )
    parser.add_argument(
        "--save-integrity-report", type=Path,
        default=Path("data/walk_forward/phase7_data_integrity_report.json"),
    )
    parser.add_argument(
        "--save-comparison", type=Path,
        default=Path("data/walk_forward/phase6_5_vs_phase7_comparison.json"),
    )
    parser.add_argument(
        "--phase6-5-report", type=Path,
        default=Path("data/walk_forward/phase6_5_full_universe_report.json"),
        help="Already-saved Phase 6.5 report, used for the comparison section only",
    )
    parser.add_argument("--fetch-start-date", type=date.fromisoformat, default=PHASE7_FETCH_START)
    parser.add_argument("--fetch-end-date", type=date.fromisoformat, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    config.data.raw_dir = args.raw_dir
    config.data.processed_dir = args.processed_dir
    config.data.features_dir = args.features_dir
    config.data.signals_dir = args.signals_dir
    config.data.scores_dir = args.scores_dir
    setup_logging(config.logging.level, config.logging.log_dir)

    fetch_end_date = args.fetch_end_date or datetime.date.today()

    manifest = load_manifest(args.manifest)
    entries: dict = manifest.get("tickers", {})
    final_universe_tickers = sorted(
        t for t, e in entries.items() if e.get("included_in_universe")
    )
    print(f"Full Universe ticker count (from manifest): {len(final_universe_tickers)}")
    if not final_universe_tickers:
        print(
            "No tickers in the Full Universe - nothing to do. "
            "Run scripts/run_universe_ingest.py with --raw-dir/--processed-dir "
            "pointed at data/phase7/ first."
        )
        return

    master = load_jpx_master(args.jpx_master_path)
    build_result = apply_static_filters(
        master,
        segments=config.universe.segments,
        exclude_etf=config.universe.exclude_etf,
        exclude_reit=config.universe.exclude_reit,
    )

    print("\n--- Step 1/4: build_features ---")
    feat_summary = run_build_features(config, tickers=final_universe_tickers)
    print(f"  success={feat_summary.success_count} failed={len(feat_summary.failed_tickers)}")

    print("--- Step 2/4: build_signals ---")
    sig_summary = run_build_signals(config, tickers=final_universe_tickers)
    print(
        f"  success={sig_summary.success_count} failed={len(sig_summary.failed_tickers)} "
        f"total_signals={sig_summary.total_signals_triggered}"
    )

    print("--- Step 3/4: build_scores ---")
    score_summary = run_build_scores(config, tickers=final_universe_tickers)
    print(f"  success={score_summary.success_count} failed={len(score_summary.failed_tickers)}")

    print(f"--- Step 4/4: run_walk_forward (min_oos_start={PHASE7_OOS_START}) ---")
    report: WalkForwardReport = run_walk_forward(
        config, tickers=final_universe_tickers, min_oos_start=PHASE7_OOS_START
    )
    print(f"  windows={len(report.windows)} signals_evaluated={len(report.signal_results)}")
    for w in report.windows:
        print(f"    window {w.index}: OOS {w.oos_start}..{w.oos_end}")
    print(f"  config_hash={report.config_hash}")
    print(f"  data_hash={report.data_hash}")

    _save_json(report, args.save_walk_forward_report)
    print(f"  saved: {args.save_walk_forward_report}")

    print("\n--- Data Integrity funnel ---")
    integrity = build_data_integrity_report(
        jpx_master_candidates=len(master),
        static_filter_included=len(build_result.included),
        static_filter_excluded=len(build_result.excluded),
        manifest_path=args.manifest,
        raw_dir=Path(config.data.raw_dir),
        start=args.fetch_start_date,
        end=fetch_end_date,
    )
    f = integrity.funnel
    print(f"  jpx_master_candidates: {f.jpx_master_candidates}")
    print(f"  static_filter_included: {f.static_filter_included}")
    print(f"  static_filter_excluded: {f.static_filter_excluded}")
    print(f"  fetch_attempted: {f.fetch_attempted}")
    print(f"  fetch_success/partial/failed: {f.fetch_success}/{f.fetch_partial}/{f.fetch_failed}")
    print(f"  price_liquidity_excluded: {f.price_liquidity_excluded}")
    print(f"  final_universe: {f.final_universe}")

    _save_json(integrity, args.save_integrity_report)
    print(f"  saved: {args.save_integrity_report}")

    if args.phase6_5_report.exists():
        print("\n--- Phase 6.5 vs Phase 7 comparison ---")
        phase6_5_report = json.loads(args.phase6_5_report.read_text(encoding="utf-8"))
        phase7_report = json.loads(args.save_walk_forward_report.read_text(encoding="utf-8"))
        comparisons = compare_reports(phase6_5_report, phase7_report)
        for c in comparisons:
            print(
                f"  {c.direction}:{c.signal_name}: case={c.case} "
                f"PF(base) {c.pf_base_p65} -> {c.pf_base_p7} "
                f"decision {c.decision_p65} -> {c.decision_p7}"
            )
        _save_json(comparisons, args.save_comparison)
        print(f"  saved: {args.save_comparison}")
    else:
        print(
            f"\n(skipping comparison - {args.phase6_5_report} not found; "
            "pass --phase6-5-report to point at the saved Phase 6.5 report)"
        )


if __name__ == "__main__":
    main()
