"""Model A: LightGBM Regression (spec section 5) - Feature(t) -> Future
Return(t+h). Target column is switchable via config (spec section 5's
"target columnをconfigから切り替えられる構造にする"); the model itself
has no opinion about which Horizon/Variant it was trained against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from v3.models.config import BASELINE_PARAMS, LightGBMBaselineParams


def fit_regression_model(
    X: pd.DataFrame, y: pd.Series, params: LightGBMBaselineParams = BASELINE_PARAMS
) -> LGBMRegressor:
    model = LGBMRegressor(**params.as_kwargs(), verbosity=-1)
    model.fit(X, y)
    return model


def predict_regression(model: LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))
