"""Feature Decomposition / Feature Contribution / Q1 vs Q5 Profile
(spec sections 6/7/10): "which Feature(s) is Q1 actually made of?"

Reuses V2-1's own CATEGORY_FEATURES dict and percentile_rank_by_day()
UNMODIFIED (v2/ranking/score.py, v2/ranking/cross_sectional.py) - every
per-feature percentile column here is computed by the SAME cross-
sectional ranking function V2-1's Score itself uses, just persisted
per-feature instead of averaged away into a category rank. This module
never changes CATEGORY_FEATURES, never adds a feature, never changes a
weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2.ranking.cross_sectional import percentile_rank_by_day
from v2.ranking.score import CATEGORY_FEATURES

# One "raw_pct_<feature>" column per (category, feature) pair - the same
# percentile_rank_by_day() V2-1's own category ranks average together,
# kept individually here instead of averaged away.
FEATURE_LIST: list[tuple[str, str, bool]] = [
    (category, column, higher_is_better)
    for category, members in CATEGORY_FEATURES.items()
    for column, higher_is_better in members
]


def percentile_column(feature: str) -> str:
    return f"pct_{feature}"


def compute_feature_percentiles(panel: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Adds one pct_<feature> column per raw Feature in CATEGORY_FEATURES,
    each in (0, 1] with 1.0 always meaning "more attractive" in that
    feature's own higher_is_better direction (matching how V2-1's Score
    itself reads every feature) - so Q5 of pct_<feature> always means
    "best 20% by that feature," consistent with what Q5 of total_score
    already means. Returns a NEW DataFrame merged onto panel by date/ticker
    (does not mutate panel).
    """
    out = panel[[date_col, "ticker"]].copy()
    for _category, column, higher_is_better in FEATURE_LIST:
        out[percentile_column(column)] = percentile_rank_by_day(
            panel, column, date_col=date_col, higher_is_better=higher_is_better
        )
    return panel.merge(out, on=[date_col, "ticker"], how="left")


@dataclass(frozen=True)
class FeatureBucketProfile:
    category: str
    feature: str
    bucket: str
    n: int
    raw_mean: float | None
    raw_median: float | None
    pct_mean: float | None  # mean of this feature's own cross-sectional percentile
    population_raw_mean: float | None
    z_score: float | None  # (bucket raw mean - population raw mean) / population raw std


def _zscore(bucket_values: pd.Series, population_values: pd.Series) -> float | None:
    pop = population_values.dropna()
    if len(pop) < 2 or pop.std() == 0:
        return None
    bucket_mean = bucket_values.dropna().mean()
    if pd.isna(bucket_mean):
        return None
    return float((bucket_mean - pop.mean()) / pop.std())


def compute_feature_bucket_profile(
    scored: pd.DataFrame, bucket_col: str = "score_bucket", bucket_label: str = "Q1"
) -> list[FeatureBucketProfile]:
    """Per-feature decomposition of one score_bucket (default Q1, spec
    section 6): raw mean/median, mean percentile rank, population raw
    mean, and Z-score of the bucket's raw mean relative to the FULL
    population's raw distribution (not just this bucket) - answers "how
    extreme is Q1 on this Feature, in the Feature's own units?"
    """
    subset = scored[scored[bucket_col] == bucket_label]
    results = []
    for category, feature, _higher_is_better in FEATURE_LIST:
        pct_col = percentile_column(feature)
        raw_values = subset[feature].dropna()
        n = len(raw_values)
        results.append(
            FeatureBucketProfile(
                category=category,
                feature=feature,
                bucket=bucket_label,
                n=n,
                raw_mean=float(raw_values.mean()) if n else None,
                raw_median=float(raw_values.median()) if n else None,
                pct_mean=float(subset[pct_col].dropna().mean())
                if subset[pct_col].notna().any()
                else None,
                population_raw_mean=float(scored[feature].dropna().mean())
                if scored[feature].notna().any()
                else None,
                z_score=_zscore(subset[feature], scored[feature]),
            )
        )
    return results


@dataclass(frozen=True)
class CategoryContribution:
    category: str
    bucket: str
    mean_category_rank: float | None
    population_mean_category_rank: float | None  # always ~0.5 by construction, kept for clarity
    deviation_from_population: float | None  # bucket mean - population mean (negative = "low" side)


def compute_category_contribution(
    scored: pd.DataFrame, bucket_col: str = "score_bucket", bucket_label: str = "Q1"
) -> list[CategoryContribution]:
    """spec section 7: for the 6 category_rank columns (V2-1's own
    CATEGORY_RANK_COLUMNS, unmodified), how far below/above the population
    mean (~0.5 by construction, since ranks are uniform percentiles) does
    `bucket_label` sit? The category with the LARGEST negative deviation
    is the one most "forming" a low bucket like Q1 - explanation only,
    weights are never touched.
    """
    subset = scored[scored[bucket_col] == bucket_label]
    results = []
    for category in CATEGORY_FEATURES:
        col = f"{category}_rank"
        bucket_mean = subset[col].dropna().mean() if subset[col].notna().any() else None
        population_mean = scored[col].dropna().mean() if scored[col].notna().any() else None
        deviation = (
            float(bucket_mean - population_mean)
            if bucket_mean is not None and population_mean is not None
            else None
        )
        results.append(
            CategoryContribution(
                category=category,
                bucket=bucket_label,
                mean_category_rank=float(bucket_mean) if bucket_mean is not None else None,
                population_mean_category_rank=float(population_mean)
                if population_mean is not None
                else None,
                deviation_from_population=deviation,
            )
        )
    return results


def compute_category_correlation_matrix(
    scored: pd.DataFrame, bucket_col: str | None = None, bucket_label: str | None = None
) -> pd.DataFrame:
    """Pearson correlation matrix of the 6 category_rank columns (spec
    section 7's "Component間相関") - computed either over the full
    population (bucket_col=None) or restricted to one bucket, to see
    whether categories move together more/less tightly inside Q1 than in
    the general population.
    """
    df = scored
    if bucket_col is not None and bucket_label is not None:
        df = df[df[bucket_col] == bucket_label]
    cols = [f"{category}_rank" for category in CATEGORY_FEATURES]
    return df[cols].corr(method="pearson")


def rank_categories_by_q1_deviation(
    contributions: list[CategoryContribution],
) -> list[CategoryContribution]:
    """Sorted ascending by deviation_from_population (most negative /
    "most Q1-forming" first) - identifies the dominant contributor(s)
    named in spec section 7 ("Q1を形成している主要component"). NaN
    deviations sort last.
    """
    return sorted(
        contributions,
        key=lambda c: c.deviation_from_population
        if c.deviation_from_population is not None
        else np.inf,
    )
