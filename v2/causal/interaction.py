"""Score x Feature interaction, and pairwise Feature x Feature interaction
(spec sections 12/13): is Q1's negative relationship driven by a SINGLE
feature, or does it only show up when several features are jointly
extreme?

Builds on v2/causal/single_feature.py::assign_feature_buckets() (itself a
reuse of V1's assign_quantile_buckets() on each feature's own percentile
column) - no new bucketing rule, no new Signal condition is generated
here (spec section 13's explicit prohibition).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from v2.causal.feature_stats import percentile_column
from v2.causal.single_feature import assign_feature_buckets
from v2.stats import ReturnStats, compute_return_stats


@dataclass(frozen=True)
class ScoreFeatureCell:
    score_bucket: str
    feature_bucket: str
    stats: ReturnStats


@dataclass(frozen=True)
class ScoreFeatureCrosstab:
    feature: str
    window_days: int
    cells: list[ScoreFeatureCell]


def score_feature_crosstab(
    scored: pd.DataFrame,
    feature: str,
    return_col: str,
    window_days: int,
    score_bucket_col: str = "score_bucket",
    score_buckets: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5"),
) -> ScoreFeatureCrosstab:
    """spec section 12: within each Score bucket (default all of Q1-Q5,
    the Q1 row is the one the report focuses on), split further by this
    Feature's own Q1-Q5 bucket and compare Forward Return - reveals
    whether a Score bucket's composition on this one Feature matters.
    """
    df = scored.dropna(subset=[return_col, percentile_column(feature)]).copy()
    df["_feature_bucket"] = assign_feature_buckets(df, feature)
    cells = []
    for score_bucket in score_buckets:
        for feature_bucket in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            subset = df[
                (df[score_bucket_col] == score_bucket) & (df["_feature_bucket"] == feature_bucket)
            ]
            cells.append(
                ScoreFeatureCell(
                    score_bucket=score_bucket,
                    feature_bucket=feature_bucket,
                    stats=compute_return_stats(subset[return_col]),
                )
            )
    return ScoreFeatureCrosstab(feature=feature, window_days=window_days, cells=cells)


@dataclass(frozen=True)
class PairwiseCell:
    label: str  # "low/low", "low/high", "high/low", "high/high"
    stats: ReturnStats


@dataclass(frozen=True)
class PairwiseInteractionResult:
    feature_a: str
    feature_b: str
    window_days: int
    cells: list[PairwiseCell]


_LOW_HIGH = {"low": "Q1", "high": "Q5"}


def pairwise_feature_interaction(
    scored: pd.DataFrame, feature_a: str, feature_b: str, return_col: str, window_days: int
) -> PairwiseInteractionResult:
    """spec section 13: restricted to each feature's own top/bottom
    quintile (low=Q1, high=Q5 of that feature's percentile) - a 2x2
    grid, not the full 5x5, matching spec's own "low/low, low/high,
    high/low, high/high" framing.
    """
    df = scored.dropna(
        subset=[return_col, percentile_column(feature_a), percentile_column(feature_b)]
    ).copy()
    df["_bucket_a"] = assign_feature_buckets(df, feature_a)
    df["_bucket_b"] = assign_feature_buckets(df, feature_b)
    cells = []
    for label_a, code_a in _LOW_HIGH.items():
        for label_b, code_b in _LOW_HIGH.items():
            subset = df[(df["_bucket_a"] == code_a) & (df["_bucket_b"] == code_b)]
            cells.append(
                PairwiseCell(
                    label=f"{label_a}/{label_b}", stats=compute_return_stats(subset[return_col])
                )
            )
    return PairwiseInteractionResult(
        feature_a=feature_a, feature_b=feature_b, window_days=window_days, cells=cells
    )


# One representative feature per category (spec section 13's explicit
# allowance to restrict pairwise interaction to V2-1's "main categories"
# rather than all C(24,2) pairs) - chosen as the single member each
# category's own docstring/ordering in v2/ranking/score.py::
# CATEGORY_FEATURES treats as its clearest representative (first listed
# member most central to that category's name), fixed here before running,
# not selected after seeing which pair looked interesting.
CATEGORY_REPRESENTATIVE_FEATURE: dict[str, str] = {
    "momentum": "return_20d",
    "trend": "sma_20_slope",
    "volume": "volume_ratio_20d",
    "volatility": "volatility_20d",
    "relative_strength": "rs_20d",
    "pullback": "rsi_14",
}
