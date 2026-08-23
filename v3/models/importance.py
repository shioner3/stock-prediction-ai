"""Feature importance extraction (spec section 15) - Gain, Split, and SHAP
(via LightGBM's own native `pred_contrib=True` prediction mode, which
gives exact SHAP values without adding a separate `shap` package
dependency - see LightGBM's documentation on `predict_contrib`).

This module ONLY reports importance; nothing here removes, reorders, or
otherwise changes the Feature Registry (`v3/features/registry.py` stays
completely fixed - spec section 15's explicit "importanceが低いFeature
を削除などは絶対にしない").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm.sklearn import LGBMClassifier


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    gain: float
    split: float


def extract_gain_and_split_importance(
    model: LGBMRegressor | LGBMClassifier, feature_names: list[str]
) -> list[FeatureImportance]:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    return [
        FeatureImportance(feature=name, gain=float(g), split=float(s))
        for name, g, s in zip(feature_names, gain, split, strict=True)
    ]


def compute_shap_values(
    model: LGBMRegressor | LGBMClassifier, X: pd.DataFrame
) -> pd.DataFrame:
    """Exact SHAP values via LightGBM's native pred_contrib mode - one
    column per feature (same order as X.columns) plus a final
    "_expected_value" (bias) column, one row per input row.
    """
    contributions = np.asarray(model.predict(X, pred_contrib=True))
    columns = [*X.columns, "_expected_value"]
    return pd.DataFrame(contributions, columns=columns, index=X.index)


def mean_absolute_shap_by_feature(shap_values: pd.DataFrame) -> pd.Series:
    feature_columns = [c for c in shap_values.columns if c != "_expected_value"]
    return shap_values[feature_columns].abs().mean().sort_values(ascending=False)
