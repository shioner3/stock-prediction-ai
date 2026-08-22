"""Phase 11 Track B CLI entry point: independent verification of the 11
Signals not yet covered by Phase 6.5-9, over the SAME data/phase7/
dataset already fetched (no new fetch needed). See
pipeline/run_phase11_research.py's module docstring for the full design.

Usage:
    python scripts/run_phase11_research.py
    python scripts/run_phase11_research.py --target-signal-names long_breakout,short_pullback
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
from pipeline.run_phase11_research import (  # noqa: E402
    REMAINING_SIGNALS,
    Phase11ResearchReport,
    run_phase11_research,
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
        description="Run Phase 11 Track B: independent verification of the remaining 11 Signals"
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
    parser.add_argument(
        "--target-signal-names", type=str, default=None,
        help="comma-separated signal_name list to restrict to (direction inferred "
        "from REMAINING_SIGNALS); default: all 11 remaining signals",
    )
    parser.add_argument(
        "--save-report", type=Path, default=Path("data/walk_forward/phase11_research_report.json"),
    )
    return parser.parse_args()


def _print_summary(report: Phase11ResearchReport) -> None:
    print(f"config_check.matches: {report.config_check.matches}")
    print(f"tickers: {len(report.tickers)}")
    print(f"windows_evaluated: {report.windows_evaluated}")
    print(f"\n{'signal':<32} {'decision':<22} {'n_oos':>7} {'PF':>8} {'p_raw':>8} {'q_fdr':>8}")
    for s in report.signals:
        if s.combined is None:
            print(f"{s.direction}:{s.signal_name:<28} NEVER_TRIGGERED")
            continue
        base = s.combined.oos_metrics_by_cost_tier.get("base")
        n = base.n_trades if base else 0
        pf = f"{base.profit_factor:.3f}" if base and base.profit_factor is not None else "n/a"
        p_raw = f"{s.combined.permutation.p_value:.4f}" if s.combined.permutation else "n/a"
        q = f"{s.fdr.adjusted_p_value:.4f}" if s.fdr else "n/a"
        print(
            f"{s.direction}:{s.signal_name:<28} {s.decision.value if s.decision else 'n/a':<22} "
            f"{n:>7} {pf:>8} {p_raw:>8} {q:>8}"
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

    target_signals = None
    if args.target_signal_names:
        wanted_names = set(args.target_signal_names.split(","))
        target_signals = [
            (d, n) for d, n in REMAINING_SIGNALS if n in wanted_names
        ]
        if not target_signals:
            print(f"No match for --target-signal-names={args.target_signal_names!r}")
            return

    try:
        report = run_phase11_research(
            config, tickers, args.phase6_5_report, args.phase7_report,
            args.jpx_master_path, target_signals=target_signals,
        )
    except ConfigMismatchError as exc:
        print(f"CONFIG_MISMATCH: {exc}")
        sys.exit(1)

    _print_summary(report)
    _save_json(report, args.save_report)
    print(f"\nsaved: {args.save_report}")


if __name__ == "__main__":
    main()
