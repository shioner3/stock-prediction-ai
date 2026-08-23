"""Model B: LightGBM Binary Classification (spec section 6). Target:
future_return_h > 0. Output: predicted probability of a positive return -
spec section 6's explicit caution applies verbatim: this Phase does NOT
assume "higher probability -> more profitable" - that is a Calibration
question deferred to a later Phase (spec section 6/24).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from v3.models.config import BASELINE_PARAMS, LightGBMBaselineParams


def fit_classification_model(
    X: pd.DataFrame, y: pd.Series, params: LightGBMBaselineParams = BASELINE_PARAMS
) -> LGBMClassifier:
    model = LGBMClassifier(**params.as_kwargs(), verbosity=-1)
    model.fit(X, y)
    return model


def predict_positive_probability(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """P(future_return_h > 0) - the positive class's probability column."""
    proba = np.asarray(model.predict_proba(X))
    positive_class_index = list(model.classes_).index(1)
    return np.asarray(proba[:, positive_class_index])
