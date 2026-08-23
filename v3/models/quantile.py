"""Model C: LightGBM Quantile Regression (spec section 7) - q in
{0.1, 0.5, 0.9}, giving Downside/Median/Upside forward-return estimates.
LightGBM's sklearn API trains exactly ONE quantile per model instance
(`objective="quantile", alpha=q`), so this fits QUANTILES.length
independent models rather than one multi-output model - the standard way
to get multiple quantiles from gradient-boosted trees, not a shortcut.

spec section 7 is explicit that V3-2 does NOT build a final Risk-adjusted
Score from these - `fit_quantile_models()`/`predict_quantiles()` only
produce the three raw per-quantile predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from v3.models.config import BASELINE_PARAMS, QUANTILES, LightGBMBaselineParams


def fit_quantile_models(
    X: pd.DataFrame,
    y: pd.Series,
    quantiles: tuple[float, ...] = QUANTILES,
    params: LightGBMBaselineParams = BASELINE_PARAMS,
) -> dict[float, LGBMRegressor]:
    models: dict[float, LGBMRegressor] = {}
    for q in quantiles:
        model = LGBMRegressor(objective="quantile", alpha=q, **params.as_kwargs(), verbosity=-1)
        model.fit(X, y)
        models[q] = model
    return models


def predict_quantiles(models: dict[float, LGBMRegressor], X: pd.DataFrame) -> pd.DataFrame:
    predictions = {f"q{q}": np.asarray(model.predict(X)) for q, model in models.items()}
    return pd.DataFrame(predictions, index=X.index)
