"""V2-1 dry run: confirm the Feature/Target/Ranking Engine works end-to-
end against the existing Research/Development Dataset (spec section 16)
and compute the Score Q1-Q5 / Holding Period statistics
research/phase_v2_1_report.md reports.

This is NOT a "V2 independent OOS performance" claim (spec section 16's
own prohibition - the 2022-2026 dataset is Research/Development only,
already analyzed by V1's own Phases) and is NOT meant for repeated daily
use (see scripts/run_v2_ranking.py for that) - it is a one-time report-
data generator for Phase V2-1's own completion report.

Usage:
    python scripts/run_v2_research_dry_run.py
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from scoring.validation import assign_quantile_buckets  # noqa: E402
from v2.config.loader import load_v2_config  # noqa: E402
from v2.manifest import build_v2_manifest, save_v2_manifest  # noqa: E402
from v2.pipeline import run_v2_ranking  # noqa: E402
from v2.ranking.score import CATEGORY_FEATURES  # noqa: E402
from v2.stats import (  # noqa: E402
    MAX_PLAUSIBLE_FORWARD_RETURN,
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    exclude_implausible_returns,
)

FORWARD_RETURN_COLUMNS = {
    5: "forward_return_5d", 10: "forward_return_10d",
    15: "forward_return_15d", 20: "forward_return_20d",
}


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def main() -> None:
    v1_config = load_app_config(Path("config/settings.yaml"))
    setup_logging(v1_config.logging.level, v1_config.logging.log_dir)
    v2_config = load_v2_config()

    print("Building V2 Universe panel + cross-sectional ranking (Full Universe)...")
    ranked = run_v2_ranking(v2_config)
    print(f"ranked panel shape: {ranked.shape}")
    print(f"unique tickers: {ranked['ticker'].nunique()}")
    print(f"date range: {ranked['date'].min()} .. {ranked['date'].max()}")

    scored = ranked.dropna(subset=["total_score"]).copy()
    scored["score_bucket"] = assign_quantile_buckets(scored["total_score"])
    print(f"rows with a valid total_score: {len(scored)} / {len(ranked)}")

    report: dict[str, Any] = {
        "universe_size": int(ranked["ticker"].nunique()),
        "date_range_start": ranked["date"].min(),
        "date_range_end": ranked["date"].max(),
        "total_rows": len(ranked),
        "scored_rows": len(scored),
        "score_bucket_counts": scored["score_bucket"].value_counts().to_dict(),
        "holding_periods": {},
    }

    for window, col in FORWARD_RETURN_COLUMNS.items():
        window_scored_raw = scored.dropna(subset=[col])
        window_scored = exclude_implausible_returns(window_scored_raw, col)
        n_excluded = len(window_scored_raw) - len(window_scored)
        bucket_stats = compute_quantile_bucket_stats(window_scored, "score_bucket", col, window)
        spread = compute_q5_q1_spread(bucket_stats)
        report["holding_periods"][str(window)] = {
            "n_resolved": len(window_scored),
            "n_excluded_as_implausible": n_excluded,
            "max_plausible_forward_return": MAX_PLAUSIBLE_FORWARD_RETURN,
            "buckets": {b.bucket: dataclasses.asdict(b.stats) for b in bucket_stats},
            "q5_q1_spread": spread,
        }
        print(
            f"\n--- {window}d forward return, by Score bucket "
            f"(n_resolved={len(window_scored)}, excluded_as_implausible={n_excluded}) ---"
        )
        for b in sorted(bucket_stats, key=lambda x: x.bucket):
            s = b.stats
            print(
                f"  {b.bucket}: n={s.n} mean={s.mean_return} median={s.median_return} "
                f"win_rate={s.win_rate} PF={s.profit_factor} expectancy={s.expectancy} "
                f"std={s.std_dev} max_loss={s.max_loss}"
            )
        print(f"  Q5-Q1 spread: {spread}")

    save_path = Path("data/v2/reports/v2_1_dry_run_stats.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(report, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {save_path}")

    feature_list = sorted({col for members in CATEGORY_FEATURES.values() for col, _ in members})
    manifest = build_v2_manifest(
        universe_size=report["universe_size"],
        date_range_start=report["date_range_start"],
        date_range_end=report["date_range_end"],
        feature_list=feature_list,
        score_weights=v2_config.score_weights.model_dump(),
        forward_windows=v2_config.forward_windows,
    )
    manifest_path = save_v2_manifest(
        manifest, Path("data/v2/manifests/manifest_dry_run.json")
    )
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
