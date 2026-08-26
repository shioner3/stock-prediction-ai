from __future__ import annotations

import numpy as np
import pandas as pd

from v3.models.importance import (
    compute_shap_values,
    extract_gain_and_split_importance,
    mean_absolute_shap_by_feature,
)
from v3.models.regression import fit_regression_model


def _synthetic_Xy(n: int = 300, seed: int = 2):
    rng = np.random.default_rng(seed)
    feature_names = [f"f{i}" for i in range(6)]
    X = pd.DataFrame({name: rng.normal(size=n) for name in feature_names})
    y = X["f0"] * 0.05 + rng.normal(0, 0.01, size=n)  # f0 should dominate importance
    return X, y, feature_names


def test_gain_and_split_importance_cover_every_feature() -> None:
    X, y, feature_names = _synthetic_Xy()
    model = fit_regression_model(X, y)
    importance = extract_gain_and_split_importance(model, feature_names)
    assert {i.feature for i in importance} == set(feature_names)
    assert all(i.gain >= 0 for i in importance)
    assert all(i.split >= 0 for i in importance)


def test_dominant_feature_has_highest_gain() -> None:
    X, y, feature_names = _synthetic_Xy()
    model = fit_regression_model(X, y)
    importance = extract_gain_and_split_importance(model, feature_names)
    top = max(importance, key=lambda i: i.gain)
    assert top.feature == "f0"


def test_shap_values_sum_to_prediction_minus_expected_value() -> None:
    X, y, _names = _synthetic_Xy()
    model = fit_regression_model(X, y)
    shap_values = compute_shap_values(model, X)
    predictions = model.predict(X)
    feature_cols = [c for c in shap_values.columns if c != "_expected_value"]
    reconstructed = shap_values[feature_cols].sum(axis=1) + shap_values["_expected_value"]
    assert np.allclose(reconstructed.to_numpy(), predictions, atol=1e-6)


def test_mean_absolute_shap_ranks_dominant_feature_first() -> None:
    X, y, _names = _synthetic_Xy()
    model = fit_regression_model(X, y)
    shap_values = compute_shap_values(model, X)
    ranked = mean_absolute_shap_by_feature(shap_values)
    assert ranked.index[0] == "f0"
