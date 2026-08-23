"""Per-window train/predict (spec section 7): for one WalkForwardWindow,
slice TRAIN ([train_start, train_end)) and OOS ([oos_start, oos_end])
directly from the already-built Full Universe Dataset - the VALIDATION
segment in between is used for neither (this Phase's EMBARGO, see
`v3/validation/__init__.py`). Trains exactly ONE model per call using
`v3/models/*` (frozen Phase V3-2 model code, unmodified) and returns OOS
predictions tagged with the window index for later pooling/aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.walk_forward import WalkForwardWindow
from v3.models.classification import fit_classification_model, predict_positive_probability
from v3.models.data_prep import binary_target, prepare_training_set
from v3.models.quantile import fit_quantile_models, predict_quantiles
from v3.models.regression import fit_regression_model, predict_regression
from v3.validation.data_integrity import clean_predictions


def slice_window(
    dataset: pd.DataFrame, window: WalkForwardWindow
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = dataset[
        (dataset["date"] >= window.train_start) & (dataset["date"] < window.train_end)
    ]
    oos = dataset[
        (dataset["date"] >= window.oos_start) & (dataset["date"] <= window.oos_end)
    ]
    return train, oos


@dataclass(frozen=True)
class WindowPrediction:
    window_index: int
    target_col: str
    model_type: str  # "A" | "B" | "C"
    predictions: pd.DataFrame  # date, ticker, actual, prediction[, q0.1/q0.5/q0.9 for C]
    train_rows: int
    oos_rows: int
    n_tickers: int


def run_model_a_on_window(
    dataset: pd.DataFrame, window: WalkForwardWindow, target_col: str
) -> WindowPrediction:
    train, oos = slice_window(dataset, window)
    train_set = prepare_training_set(train, target_col)
    oos_set = prepare_training_set(oos, target_col)
    model = fit_regression_model(train_set.X, train_set.y)
    prediction = predict_regression(model, oos_set.X)

    predictions = pd.DataFrame(
        {
            "date": oos_set.dates.to_numpy(), "ticker": oos_set.tickers.to_numpy(),
            "actual": oos_set.y.to_numpy(), "prediction": prediction,
        }
    )
    predictions = clean_predictions(
        predictions, label=f"(window {window.index}, Model A, {target_col})"
    )
    return WindowPrediction(
        window_index=window.index, target_col=target_col, model_type="A", predictions=predictions,
        train_rows=len(train_set.X), oos_rows=len(oos_set.X),
        n_tickers=predictions["ticker"].nunique(),
    )


def run_model_b_on_window(
    dataset: pd.DataFrame, window: WalkForwardWindow, target_col: str
) -> WindowPrediction:
    train, oos = slice_window(dataset, window)
    train_set = prepare_training_set(train, target_col)
    oos_set = prepare_training_set(oos, target_col)
    train_y = binary_target(train_set.y)
    oos_y_binary = binary_target(oos_set.y)

    model = fit_classification_model(train_set.X, train_y)
    prediction = predict_positive_probability(model, oos_set.X)

    predictions = pd.DataFrame(
        {
            "date": oos_set.dates.to_numpy(), "ticker": oos_set.tickers.to_numpy(),
            "actual": oos_set.y.to_numpy(), "actual_binary": oos_y_binary.to_numpy(),
            "prediction": prediction,
        }
    )
    predictions = clean_predictions(
        predictions, label=f"(window {window.index}, Model B, {target_col})"
    )
    return WindowPrediction(
        window_index=window.index, target_col=target_col, model_type="B", predictions=predictions,
        train_rows=len(train_set.X), oos_rows=len(oos_set.X),
        n_tickers=predictions["ticker"].nunique(),
    )


def run_model_c_on_window(
    dataset: pd.DataFrame, window: WalkForwardWindow, target_col: str
) -> WindowPrediction:
    train, oos = slice_window(dataset, window)
    train_set = prepare_training_set(train, target_col)
    oos_set = prepare_training_set(oos, target_col)

    models = fit_quantile_models(train_set.X, train_set.y)
    quantile_predictions = predict_quantiles(models, oos_set.X)

    predictions = pd.DataFrame(
        {
            "date": oos_set.dates.to_numpy(), "ticker": oos_set.tickers.to_numpy(),
            "actual": oos_set.y.to_numpy(), "prediction": quantile_predictions["q0.5"].to_numpy(),
            "q0.1": quantile_predictions["q0.1"].to_numpy(),
            "q0.9": quantile_predictions["q0.9"].to_numpy(),
        }
    )
    predictions = clean_predictions(
        predictions, label=f"(window {window.index}, Model C, {target_col})"
    )
    return WindowPrediction(
        window_index=window.index, target_col=target_col, model_type="C", predictions=predictions,
        train_rows=len(train_set.X), oos_rows=len(oos_set.X),
        n_tickers=predictions["ticker"].nunique(),
    )
