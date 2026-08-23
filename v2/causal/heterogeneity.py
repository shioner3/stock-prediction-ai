"""Q1 Internal Heterogeneity (spec section 11): is Q1 uniform, or does
Forward Return keep worsening toward its very bottom?

Splits ONLY the rows already in score_bucket=="Q1" into 5 finer internal
sub-buckets via scoring.validation.assign_quantile_buckets() (V1,
unmodified) applied to total_score restricted to that subset - the same
generic quantile-splitting function used everywhere else in this project,
never a bespoke threshold. This module produces descriptive sub-bucket
statistics only; per spec section 11's explicit instruction, no new
threshold is adopted from this result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.validation import assign_quantile_buckets
from v2.stats import QuantileBucketStats, compute_quantile_bucket_stats

# Q1-a (lowest total_score within Q1) .. Q1-e (highest total_score within
# Q1, i.e. bordering Q2) - assign_quantile_buckets() itself always labels
# Q1..Q5; relabelled here purely for report readability so it is never
# confused with the outer Score bucket.
SUB_BUCKET_LABELS = {"Q1": "Q1-a", "Q2": "Q1-b", "Q3": "Q1-c", "Q4": "Q1-d", "Q5": "Q1-e"}


@dataclass(frozen=True)
class HeterogeneityResult:
    window_days: int
    sub_bucket_stats: list[QuantileBucketStats]


def analyze_q1_heterogeneity(
    scored: pd.DataFrame, return_col: str, window_days: int, score_bucket_col: str = "score_bucket"
) -> HeterogeneityResult:
    q1_rows = scored[scored[score_bucket_col] == "Q1"].dropna(subset=[return_col]).copy()
    q1_rows["_sub_bucket"] = assign_quantile_buckets(q1_rows["total_score"]).map(SUB_BUCKET_LABELS)
    bucket_stats = compute_quantile_bucket_stats(q1_rows, "_sub_bucket", return_col, window_days)
    # Reorder to the natural Q1-a..Q1-e sequence (compute_quantile_bucket_stats
    # groups alphabetically by default, which already matches here).
    order = {label: i for i, label in enumerate(SUB_BUCKET_LABELS.values())}
    bucket_stats = sorted(bucket_stats, key=lambda b: order.get(b.bucket, 99))
    return HeterogeneityResult(window_days=window_days, sub_bucket_stats=bucket_stats)
