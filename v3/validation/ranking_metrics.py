"""Cross-sectional ranking evaluation (spec section 8/10): Q1-Q5 bucket
stats + Q5-Q1 spread + Rank IC for ONE window's (or the FULL pooled OOS)
predictions. Reuses `scoring.validation.assign_quantile_buckets()` (V1),
`v2.stats.compute_quantile_bucket_stats()`/`compute_q5_q1_spread()` (V2),
and `v2.validation.ic.compute_daily_spearman_ic()`/`summarize_ic()`
(Phase V2-2) all UNMODIFIED - applied to a Model prediction column
instead of V2's rule-based Score, exactly mirroring `v3/models/
cross_sectional.py`'s own precedent from Phase V3-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.validation import assign_quantile_buckets
from v2.stats import QuantileBucketStats, compute_q5_q1_spread, compute_quantile_bucket_stats
from v2.validation.ic import ICSummary, compute_daily_spearman_ic, summarize_ic


@dataclass(frozen=True)
class RankingResult:
    n: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None
    ic_summary: ICSummary


def evaluate_ranking(
    predictions: pd.DataFrame,
    window_days: int,
    prediction_col: str = "prediction",
    actual_col: str = "actual",
) -> RankingResult:
    valid = predictions.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    bucket_stats = compute_quantile_bucket_stats(valid, "_bucket", actual_col, window_days)
    daily_ic = compute_daily_spearman_ic(valid, prediction_col, actual_col)
    return RankingResult(
        n=len(valid),
        bucket_stats=bucket_stats,
        q5_q1_spread=compute_q5_q1_spread(bucket_stats),
        ic_summary=summarize_ic(daily_ic, window_days),
    )
