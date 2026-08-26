"""Loads a persisted Frozen Model artifact and predicts on a NEW Feature
panel - deliberately does NOT go through `v3.models.regression.
predict_regression()`'s `LGBMRegressor` wrapper (that requires the
in-memory sklearn object this process never has after a restart); a
`lgb.Booster` loaded from the saved text file predicts identically
(LightGBM's own documented guarantee - `Booster.predict()` is a pure
function of the saved tree structure and the input matrix).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v3.features.registry import CORE_FEATURE_NAMES
from v3.frozen.manifest import FrozenModelSpec, load_model_artifact
from v3.models.data_prep import assert_no_target_leakage_in_features


def predict_with_frozen_model(spec: FrozenModelSpec, feature_panel: pd.DataFrame) -> np.ndarray:
    """feature_panel: one row per ticker, must contain every
    CORE_FEATURE_NAMES column (missing-value-aware, NaN features are
    handled natively by LightGBM exactly as at training time).
    """
    assert_no_target_leakage_in_features(CORE_FEATURE_NAMES)
    booster = load_model_artifact(Path(spec.artifact_path))
    x = feature_panel[CORE_FEATURE_NAMES]
    return np.asarray(booster.predict(x))


def predict_with_all_frozen_models(
    specs: list[FrozenModelSpec], feature_panel: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {spec.model_id: predict_with_frozen_model(spec, feature_panel) for spec in specs}
