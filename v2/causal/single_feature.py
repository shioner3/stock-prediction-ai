"""Single Feature -> Forward Return, and Feature Direction classification
(spec sections 8/9).

Buckets EACH raw Feature's own cross-sectional percentile (v2/causal/
feature_stats.py::compute_feature_percentiles(), itself a direct reuse of
V2-1's percentile_rank_by_day()) via scoring.validation.
assign_quantile_buckets() - the SAME V1 function that splits total_score
into Q1-Q5, applied here per-feature instead. Because each percentile
column already encodes that feature's own higher_is_better direction,
"Q5" always means "best 20% by that feature" for every feature, exactly
matching what Q5 already means for total_score - no per-feature special
casing needed.

This module explicitly does NOT define a new Signal or add anything to
V2-1's Score: it is read-only analysis over Feature percentile buckets
that already exist as V2-1 output columns (spec section 8's own
"Feature percentileを新しいScoreへ追加しない" instruction).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from scoring.validation import assign_quantile_buckets
from v2.causal.feature_stats import FEATURE_LIST, percentile_column
from v2.stats import QuantileBucketStats, compute_q5_q1_spread, compute_quantile_bucket_stats
from v2.validation.monotonicity import MonotonicityResult, compute_monotonicity


@dataclass(frozen=True)
class SingleFeatureWindowResult:
    category: str
    feature: str
    window_days: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None
    monotonicity: MonotonicityResult


def assign_feature_buckets(panel: pd.DataFrame, feature: str) -> pd.Series:
    """Q1 (worst 20% by this feature's own direction) .. Q5 (best 20%) -
    scoring.validation.assign_quantile_buckets(), unmodified, applied to
    this feature's percentile column instead of total_score.
    """
    return assign_quantile_buckets(panel[percentile_column(feature)])


def analyze_single_feature(
    scored: pd.DataFrame, category: str, feature: str, return_col: str, window_days: int
) -> SingleFeatureWindowResult:
    df = scored.dropna(subset=[return_col, percentile_column(feature)]).copy()
    df["_feature_bucket"] = assign_feature_buckets(df, feature)
    bucket_stats = compute_quantile_bucket_stats(df, "_feature_bucket", return_col, window_days)
    bucket_means = {b.bucket: b.stats.mean_return for b in bucket_stats}
    return SingleFeatureWindowResult(
        category=category,
        feature=feature,
        window_days=window_days,
        bucket_stats=bucket_stats,
        q5_q1_spread=compute_q5_q1_spread(bucket_stats),
        monotonicity=compute_monotonicity(bucket_means, window_days),
    )


def analyze_all_features(
    scored: pd.DataFrame, return_col: str, window_days: int
) -> list[SingleFeatureWindowResult]:
    return [
        analyze_single_feature(scored, category, feature, return_col, window_days)
        for category, feature, _higher_is_better in FEATURE_LIST
    ]


class FeatureDirection(str, Enum):
    """spec section 9's 4-way classification. Thresholds are fixed BEFORE
    running against this Phase's own data (pre-registered, not tuned to
    this Phase's results):
      - spearman is computed on the 5 bucket-mean points (same convention
        as v2/validation/monotonicity.py - a trivial n=5 correlation, not
        a per-row correlation).
      - The |spread| floor (0.05%) is HALF of Phase V2-2's own already-
        published primary Q5-Q1 spread magnitude (0.109%, research/
        phase_v2_2_report.md section 7) - an external, independently-
        arrived-at reference scale from a PRIOR Phase, not chosen from
        this Phase's own feature results.
    """

    POSITIVE_PREDICTIVE = "POSITIVE_PREDICTIVE"
    NEGATIVE_PREDICTIVE = "NEGATIVE_PREDICTIVE"
    NON_MONOTONIC = "NON_MONOTONIC"
    NO_EVIDENCE = "NO_EVIDENCE"


SPEARMAN_STRONG_THRESHOLD = 0.8
SPREAD_FLOOR = 0.0005  # half of V2-2's primary 5d Q5-Q1 spread magnitude (0.109%)


def classify_feature_direction(result: SingleFeatureWindowResult) -> FeatureDirection:
    spearman = result.monotonicity.spearman
    spread = result.q5_q1_spread
    if spearman is not None and spearman >= SPEARMAN_STRONG_THRESHOLD:
        return FeatureDirection.POSITIVE_PREDICTIVE
    if spearman is not None and spearman <= -SPEARMAN_STRONG_THRESHOLD:
        return FeatureDirection.NEGATIVE_PREDICTIVE
    if spread is not None and abs(spread) > SPREAD_FLOOR:
        return FeatureDirection.NON_MONOTONIC
    return FeatureDirection.NO_EVIDENCE
