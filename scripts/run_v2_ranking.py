"""V2 CLI entry point: build the Swing Candidate Ranking for one date
(spec section 21).

Usage:
    python scripts/run_v2_ranking.py --date 2026-08-20
    python scripts/run_v2_ranking.py                 # latest available date

Reads V1's already-fetched Full Universe caches (data/phase7/ by
default - see v2/config/v2_settings.yaml) - never re-fetches from
providers/, never writes into any V1-owned directory. Writes V2's own
output under data/v2/ (rankings/candidates/manifests).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from v2.candidate import CandidateRecord  # noqa: E402
from v2.config.loader import load_v2_config  # noqa: E402
from v2.manifest import build_v2_manifest, save_v2_manifest  # noqa: E402
from v2.pipeline import candidate_table_for_date, run_v2_ranking  # noqa: E402
from v2.ranking.score import CATEGORY_FEATURES  # noqa: E402


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2 Swing Candidate Ranking for one date")
    parser.add_argument("--v2-config", type=Path, default=Path("v2/config/v2_settings.yaml"))
    parser.add_argument("--v1-config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD; default = latest")
    parser.add_argument("--limit-tickers", type=int, default=None, help="dev/testing only")
    return parser.parse_args()


def _print_candidates(title: str, records: list[CandidateRecord]) -> None:
    print(f"\n--- {title} ---")
    for r in records:
        print(
            f"  #{r.rank:<4} {r.ticker:<10} score={r.score:.4f} "
            f"pctl={r.score_percentile:.3f} {r.classification}"
        )


def main() -> None:
    args = parse_args()
    v1_config = load_app_config(args.v1_config)
    setup_logging(v1_config.logging.level, v1_config.logging.log_dir)

    v2_config = load_v2_config(args.v2_config)

    tickers = None
    if args.limit_tickers:
        from v2.pipeline import load_universe_tickers

        tickers = load_universe_tickers(v2_config)[: args.limit_tickers]

    ranked = run_v2_ranking(v2_config, tickers=tickers)
    if ranked.empty:
        print("No V2 ranking data produced - check source_features_dir/manifest paths.")
        sys.exit(1)

    if args.date:
        as_of_date = datetime.date.fromisoformat(args.date)
    else:
        as_of_date = ranked["date"].max()
    print(f"as_of_date: {as_of_date}")

    records = candidate_table_for_date(ranked, as_of_date)
    if not records:
        print(f"No candidates for {as_of_date} - date may be outside the cached range.")
        sys.exit(1)

    for n in v2_config.candidate_top_n:
        _print_candidates(f"TOP {n}", records[:n])

    v2_config.v2_candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = v2_config.v2_candidates_dir / f"candidates_{as_of_date}.json"
    candidates_path.write_text(
        json.dumps([dataclasses.asdict(r) for r in records], default=_json_default, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved candidates: {candidates_path}")

    feature_list = sorted({col for members in CATEGORY_FEATURES.values() for col, _ in members})
    manifest = build_v2_manifest(
        universe_size=ranked["ticker"].nunique(),
        date_range_start=ranked["date"].min(),
        date_range_end=ranked["date"].max(),
        feature_list=feature_list,
        score_weights=v2_config.score_weights.model_dump(),
        forward_windows=v2_config.forward_windows,
    )
    manifest_path = save_v2_manifest(
        manifest, v2_config.v2_manifests_dir / f"manifest_{as_of_date}.json"
    )
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
