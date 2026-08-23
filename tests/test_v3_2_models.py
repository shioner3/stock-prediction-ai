from __future__ import annotations

import numpy as np
import pandas as pd

from v3.features.registry import CORE_FEATURE_NAMES
from v3.models.classification import fit_classification_model, predict_positive_probability
from v3.models.quantile import fit_quantile_models, predict_quantiles
from v3.models.regression import fit_regression_model, predict_regression


def _synthetic_Xy(n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {name: rng.normal(size=n) for name in CORE_FEATURE_NAMES[:10]}
    )
    y = X.iloc[:, 0] * 0.02 + rng.normal(0, 0.01, size=n)
    return X, y


def test_model_a_fits_and_predicts_expected_shape() -> None:
    X, y = _synthetic_Xy()
    model = fit_regression_model(X, y)
    preds = predict_regression(model, X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_model_a_is_reproducible_given_same_seed() -> None:
    X, y = _synthetic_Xy()
    model1 = fit_regression_model(X, y)
    model2 = fit_regression_model(X, y)
    preds1 = predict_regression(model1, X)
    preds2 = predict_regression(model2, X)
    assert np.array_equal(preds1, preds2)


def test_model_b_fits_and_predicts_probabilities_in_0_1() -> None:
    X, y = _synthetic_Xy()
    binary_y = (y > 0).astype(int)
    model = fit_classification_model(X, binary_y)
    proba = predict_positive_probability(model, X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_b_is_reproducible_given_same_seed() -> None:
    X, y = _synthetic_Xy()
    binary_y = (y > 0).astype(int)
    model1 = fit_classification_model(X, binary_y)
    model2 = fit_classification_model(X, binary_y)
    proba1 = predict_positive_probability(model1, X)
    proba2 = predict_positive_probability(model2, X)
    assert np.array_equal(proba1, proba2)


def test_model_c_fits_three_quantiles_and_predicts_monotonic_by_row() -> None:
    X, y = _synthetic_Xy()
    models = fit_quantile_models(X, y)
    assert set(models.keys()) == {0.1, 0.5, 0.9}
    predictions = predict_quantiles(models, X)
    assert list(predictions.columns) == ["q0.1", "q0.5", "q0.9"]
    assert (predictions["q0.1"] <= predictions["q0.5"] + 1e-9).mean() > 0.9
    assert (predictions["q0.5"] <= predictions["q0.9"] + 1e-9).mean() > 0.9


def test_model_c_is_reproducible_given_same_seed() -> None:
    X, y = _synthetic_Xy()
    models1 = fit_quantile_models(X, y)
    models2 = fit_quantile_models(X, y)
    pred1 = predict_quantiles(models1, X)
    pred2 = predict_quantiles(models2, X)
    assert pred1.equals(pred2)
