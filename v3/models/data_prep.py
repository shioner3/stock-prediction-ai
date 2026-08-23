"""Feature/Target matrix extraction from a V3 Dataset (spec section 16's
"target columnを誤ってFeatureに含めていないことを機械的に検証する").

`feature_matrix()` is the ONLY function anywhere in `v3/models/` allowed
to build the X passed to a model's `.fit()`/`.predict()` - it selects
EXACTLY `v3.features.registry.CORE_FEATURE_NAMES`, nothing else, so a
Target column (or `date`/`ticker`) can never end up in X by accident.
`assert_no_target_leakage_in_features()` re-verifies this mechanically
against whatever DataFrame a caller is about to hand to a model, not just
against the registry's own (trivially self-consistent) column lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from v3.features.registry import CORE_FEATURE_NAMES
from v3.targets.registry import TARGET_COLUMN_NAMES


def assert_no_target_leakage_in_features(feature_columns: list[str]) -> None:
    overlap = set(feature_columns) & set(TARGET_COLUMN_NAMES)
    if overlap:
        raise ValueError(f"TARGET_LEAKAGE_IN_FEATURES: {sorted(overlap)}")
    non_id_overlap = set(feature_columns) & {"date", "ticker"}
    if non_id_overlap:
        raise ValueError(f"non-feature identifier column(s) in feature matrix: {non_id_overlap}")


def feature_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """X - exactly the Feature Registry's CORE columns, in registry order.
    LightGBM handles NaN features natively (missing-value-aware split
    finding), so warmup-period NaNs are left as-is, never imputed.
    """
    assert_no_target_leakage_in_features(CORE_FEATURE_NAMES)
    return dataset[CORE_FEATURE_NAMES]


@dataclass(frozen=True)
class TrainingSet:
    X: pd.DataFrame
    y: pd.Series
    dates: pd.Series
    tickers: pd.Series


def prepare_training_set(dataset: pd.DataFrame, target_col: str) -> TrainingSet:
    """Drops rows where the TARGET is NaN (end-of-history rows with no
    t+h data yet, or a zero-denominator ex-post/ex-ante ratio target) -
    features may still contain NaN (see feature_matrix()'s docstring).
    """
    valid = dataset.dropna(subset=[target_col])
    return TrainingSet(
        X=feature_matrix(valid),
        y=valid[target_col],
        dates=valid["date"],
        tickers=valid["ticker"],
    )


def binary_target(y: pd.Series) -> pd.Series:
    """Model B's target (spec section 6): future_return > 0."""
    return (y > 0).astype(int)
