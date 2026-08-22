"""Phase 6.5 CLI entry point: Feature/Signal/Score build + Walk Forward +
Data Integrity report, restricted to the EXACT Full Universe ticker list
recorded in the ingestion manifest (data/raw/_universe_fetch_manifest.json)
- i.e. tickers with status in {success, partial} AND
included_in_universe=true.

Deliberately does NOT default to "everything currently in data/features/"
the way scripts/run_build_features.py etc. do on their own: this
project's data/ directories may still contain leftover files from
earlier phases (e.g. Phase 1-6's 4 hand-picked tickers, one of which -
8951 - is a REIT and would be excluded by the CURRENT static filter).
Passing an explicit ticker list is the only way to guarantee this run
covers exactly the Full Universe and nothing else.

Usage:
    python scripts/run_phase6_5_report.py
    python scripts/run_phase6_5_report.py --manifest data/raw/_universe_fetch_manifest.json
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
from pipeline.run_walk_forward import WalkForwardReport, run_walk_forward  # noqa: E402
from pipeline.universe_ingest import load_manifest  # noqa: E402
from universe.build import apply_static_filters  # noqa: E402
from universe.jpx_master import load_jpx_master  # noqa: E402

# Same Phase 6 base-case range scripts/run_universe_ingest.py used to
# fetch this data (README "OOS期間の固定") - needed here only for the
# Data Integrity report's expected-business-days/coverage_ratio
# calculation; config.data.start_date/end_date themselves are NOT
# reapplied to `config` (build_features/signals/scores/run_walk_forward
# never refetch, so they don't need it).
PHASE6_BASE_START_DATE = date(2022, 1, 4)
PHASE6_BASE_END_DATE = date(2024, 6, 28)


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
    parser = argparse.ArgumentParser(description="Run Phase 6.5 Full Universe report")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--jpx-master-path", type=Path, default=Path("data/reference/jpx_master_current.xls"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/raw/_universe_fetch_manifest.json"),
    )
    parser.add_argument(
        "--save-walk-forward-report", type=Path,
        default=Path("data/walk_forward/phase6_5_full_universe_report.json"),
    )
    parser.add_argument(
        "--save-integrity-report", type=Path,
        default=Path("data/walk_forward/phase6_5_data_integrity_report.json"),
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=PHASE6_BASE_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=PHASE6_BASE_END_DATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    setup_logging(config.logging.level, config.logging.log_dir)

    manifest = load_manifest(args.manifest)
    entries: dict = manifest.get("tickers", {})
    final_universe_tickers = sorted(
        t for t, e in entries.items() if e.get("included_in_universe")
    )
    print(f"Full Universe ticker count (from manifest): {len(final_universe_tickers)}")
    if not final_universe_tickers:
        print(
            "No tickers in the Full Universe - nothing to do. "
            "Run scripts/run_universe_ingest.py first."
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

    print("--- Step 4/4: run_walk_forward ---")
    report: WalkForwardReport = run_walk_forward(config, tickers=final_universe_tickers)
    print(f"  windows={len(report.windows)} signals_evaluated={len(report.signal_results)}")
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
        start=args.start_date,
        end=args.end_date,
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


if __name__ == "__main__":
    main()
