"""Concentration analysis (spec section 25): does the Model's Q5 bucket's
aggregate performance depend on a small number of tickers/days? Reuses
Phase V2-2's `v2/validation/concentration.py::compute_concentration()`
(frozen, unmodified) - the SAME ticker/day contribution-share primitive,
applied to a Model prediction's Q5 bucket instead of V2's rule-based
Score.
"""

from __future__ import annotations

import pandas as pd

from scoring.validation import assign_quantile_buckets
from v2.validation.concentration import ConcentrationResult, compute_concentration


def compute_prediction_concentration(
    predictions: pd.DataFrame, window_days: int, prediction_col: str = "prediction",
    actual_col: str = "actual", bucket: str = "Q5",
) -> ConcentrationResult:
    valid = predictions.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    return compute_concentration(
        valid, actual_col, window_days, bucket=bucket, bucket_col="_bucket"
    )
