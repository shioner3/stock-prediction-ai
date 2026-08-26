"""Feature/Target matrix extraction from a V3 Dataset (spec section 16's
"target columnを誤ってFeatureに含めていないことを機械的に検証する").

`feature_matrix()` is the ONLY function anywhere in `v3/models/` allowed
to build the X passed to a model's `.fit()`/`.predict()` - it selects
EXACTLY `v3.features.registry.CORE_FEATURE_NAMES`, nothing else, so a
Target column (or `date`/`ticker`) can never end up in X by accident.
`assert_no_target_leakage_in_features()` re-verifies this mechanically
against whatever DataFrame a caller is about to hand to a model, not just
against the registry's own (trivially self-consistent) column lists.

`prepare_training_set()` also excludes implausible target VALUES (reusing
`v2.stats.exclude_implausible_returns()`/`MAX_PLAUSIBLE_FORWARD_RETURN`,
V1/V2's own already-established data-integrity bound, unmodified) from
BOTH the training and OOS sides - discovered necessary during Phase V3-3's
real Full Universe run (research/phase_v3_3_report.md "Bugs Discovered"):
a single known raw-data artifact (the same class of issue V2-1 already
documented for ticker 8303 - an unadjusted-looking price producing a
~19.8-million-percent 5-day return) sat inside 3 of 6 WFO windows'
TRAINING data, causing LightGBM to learn wildly unbounded extrapolations
for those windows (predictions in the millions instead of the expected
+/-0.2 range). `v3/validation/data_integrity.py::clean_predictions()`
already filtered the OOS/evaluation side; this is the missing training-
side half of the same fix - a data-integrity safeguard, not a Target
formula change (target_raw_5d's own definition in `v3/targets/compute.py`
is untouched).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from v2.stats import MAX_PLAUSIBLE_FORWARD_RETURN, exclude_implausible_returns
from v3.features.registry import CORE_FEATURE_NAMES
from v3.targets.registry import TARGET_COLUMN_NAMES

logger = logging.getLogger(__name__)


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
    t+h data yet, or a zero-denominator ex-post/ex-ante ratio target) OR
    implausible (|target| > MAX_PLAUSIBLE_FORWARD_RETURN, a known raw-data
    artifact class - see this module's own docstring) - features may
    still contain NaN (see feature_matrix()'s docstring).
    """
    valid = dataset.dropna(subset=[target_col])
    n_before = len(valid)
    valid = exclude_implausible_returns(valid, target_col, MAX_PLAUSIBLE_FORWARD_RETURN)
    n_excluded = n_before - len(valid)
    if n_excluded > 0:
        logger.info(
            "V3 data integrity: excluded %d/%d implausible %s rows (training-set prep)",
            n_excluded, n_before, target_col,
        )
    return TrainingSet(
        X=feature_matrix(valid),
        y=valid[target_col],
        dates=valid["date"],
        tickers=valid["ticker"],
    )


def binary_target(y: pd.Series) -> pd.Series:
    """Model B's target (spec section 6): future_return > 0."""
    return (y > 0).astype(int)
