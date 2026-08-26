"""STEP 3 (spec section 39): before running any new analysis, confirm
Phase V2-2's PRIMARY (5d) window results reproduce EXACTLY from a fresh
V2-1 ranked panel built the same way this Phase's own orchestrator builds
its panel - proof that V2-3 analyzes the identical Score/Panel/Target
pipeline V2-2 already validated, not a silently-drifted variant.

Only the DETERMINISTIC pieces (no RNG) are checked - bucket means/n,
Q5-Q1 spread, monotonicity, and daily IC summary all follow purely from
(data, V2-1 code, unmodified), so an EXACT match is the right bar (this
module's own STEP 2 hash check already confirms V2-1's code/config are
byte-identical to Phase V2-2's run, which is what makes an exact match
here the correct expectation rather than an approximate one).
Bootstrap/Permutation results are not re-verified here: they are already
deterministic given a fixed seed and unchanged V1 code (which the hash
check covers), and re-running the full V2-2 battery would duplicate
~15 minutes of Full Universe compute for no additional assurance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from v2.stats import (
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    exclude_implausible_returns,
)
from v2.validation.ic import compute_daily_spearman_ic, summarize_ic
from v2.validation.monotonicity import compute_monotonicity

DEFAULT_V2_2_REPORT_PATH = Path("data/v2/reports/v2_2_validation_report.json")
PRIMARY_WINDOW_DAYS = 5
RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ReproducibilityCheckResult:
    passed: bool
    mismatches: list[str] = field(default_factory=list)
    saved_spread: float | None = None
    recomputed_spread: float | None = None


def _isclose(a: float | None, b: float | None, tol: float = RELATIVE_TOLERANCE) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verify_v2_2_reproducibility(
    ranked: pd.DataFrame,
    scored: pd.DataFrame,
    saved_report_path: Path = DEFAULT_V2_2_REPORT_PATH,
) -> ReproducibilityCheckResult:
    with open(saved_report_path, encoding="utf-8") as f:
        saved = json.load(f)
    saved_primary_light = saved["report"]["primary"]["light"]

    return_col = f"forward_return_{PRIMARY_WINDOW_DAYS}d"
    raw = scored.dropna(subset=[return_col])
    clean = exclude_implausible_returns(raw, return_col)

    bucket_stats = compute_quantile_bucket_stats(
        clean, "score_bucket", return_col, PRIMARY_WINDOW_DAYS
    )
    recomputed_spread = compute_q5_q1_spread(bucket_stats)
    saved_spread = saved_primary_light["q5_q1_spread"]

    mismatches: list[str] = []
    if not _isclose(recomputed_spread, saved_spread):
        mismatches.append(
            f"q5_q1_spread mismatch: recomputed={recomputed_spread} saved={saved_spread}"
        )

    saved_buckets = {b["bucket"]: b["stats"] for b in saved_primary_light["bucket_stats"]}
    for b in bucket_stats:
        saved_b = saved_buckets.get(b.bucket)
        if saved_b is None:
            mismatches.append(f"bucket {b.bucket} missing from saved V2-2 report")
            continue
        if b.stats.n != saved_b["n"]:
            mismatches.append(f"{b.bucket} n mismatch: recomputed={b.stats.n} saved={saved_b['n']}")
        if not _isclose(b.stats.mean_return, saved_b["mean_return"]):
            mismatches.append(
                f"{b.bucket} mean_return mismatch: recomputed={b.stats.mean_return} "
                f"saved={saved_b['mean_return']}"
            )

    bucket_means = {b.bucket: b.stats.mean_return for b in bucket_stats}
    mono = compute_monotonicity(bucket_means, PRIMARY_WINDOW_DAYS)
    saved_mono = saved_primary_light["monotonicity"]
    if not _isclose(mono.spearman, saved_mono["spearman"]):
        mismatches.append(
            f"spearman mismatch: recomputed={mono.spearman} saved={saved_mono['spearman']}"
        )

    daily_ic = compute_daily_spearman_ic(clean, "total_score", return_col)
    ic_summary = summarize_ic(daily_ic, PRIMARY_WINDOW_DAYS)
    saved_ic = saved_primary_light["ic_summary"]
    if not _isclose(ic_summary.mean_ic, saved_ic["mean_ic"]):
        mismatches.append(
            f"mean_ic mismatch: recomputed={ic_summary.mean_ic} saved={saved_ic['mean_ic']}"
        )

    n_tickers = int(ranked["ticker"].nunique())
    if n_tickers != saved["report"]["n_tickers"]:
        mismatches.append(
            f"n_tickers mismatch: recomputed={n_tickers} saved={saved['report']['n_tickers']}"
        )

    return ReproducibilityCheckResult(
        passed=len(mismatches) == 0,
        mismatches=mismatches,
        saved_spread=saved_spread,
        recomputed_spread=recomputed_spread,
    )
