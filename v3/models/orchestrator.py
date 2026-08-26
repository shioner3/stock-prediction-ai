"""Phase V3-2 orchestrator: Dataset -> time_split -> Model A/B/C ->
Prediction -> basic evaluation -> Cross-sectional check -> Random
baseline -> Feature importance, for ONE target column at a time. Used by
`scripts/train_v3_baseline.py` and by tests - NOT a Full Universe / WFO
run (spec section 18/25 - explicitly out of scope this Phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from v3.features.registry import CORE_FEATURE_NAMES
from v3.models.classification import fit_classification_model, predict_positive_probability
from v3.models.cross_sectional import (
    CrossSectionalResult,
    add_random_baseline_column,
    evaluate_cross_sectional_ranking,
)
from v3.models.data_prep import binary_target, prepare_training_set
from v3.models.evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classification,
    evaluate_regression,
)
from v3.models.importance import FeatureImportance, extract_gain_and_split_importance
from v3.models.quantile import fit_quantile_models, predict_quantiles
from v3.models.regression import fit_regression_model, predict_regression
from v3.targets.registry import target_registry_by_name


def _horizon_for_target(target_col: str) -> int:
    spec = target_registry_by_name().get(target_col)
    if spec is None:
        raise ValueError(f"unknown target column: {target_col}")
    return spec.horizon_days


@dataclass(frozen=True)
class ModelAResult:
    target_col: str
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    feature_importance: list[FeatureImportance]
    cross_sectional: CrossSectionalResult
    random_baseline_cross_sectional: CrossSectionalResult


@dataclass(frozen=True)
class ModelBResult:
    target_col: str
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    feature_importance: list[FeatureImportance]


@dataclass(frozen=True)
class ModelCResult:
    target_col: str
    quantile_predictions: pd.DataFrame = field(repr=False)


def run_model_a(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> ModelAResult:
    horizon = _horizon_for_target(target_col)
    train_set = prepare_training_set(train_df, target_col)
    test_set = prepare_training_set(test_df, target_col)

    model = fit_regression_model(train_set.X, train_set.y)
    train_pred = predict_regression(model, train_set.X)
    test_pred = predict_regression(model, test_set.X)

    test_scored = pd.DataFrame(
        {"date": test_set.dates.to_numpy(), "ticker": test_set.tickers.to_numpy(),
         target_col: test_set.y.to_numpy(), "_prediction": test_pred}
    )
    cross_sectional = evaluate_cross_sectional_ranking(
        test_scored, "_prediction", target_col, window_days=horizon
    )
    random_scored = add_random_baseline_column(test_scored)
    random_cross_sectional = evaluate_cross_sectional_ranking(
        random_scored, "_random_baseline", target_col, window_days=horizon
    )

    return ModelAResult(
        target_col=target_col,
        train_metrics=evaluate_regression(train_set.y.to_numpy(), train_pred),
        test_metrics=evaluate_regression(test_set.y.to_numpy(), test_pred),
        feature_importance=extract_gain_and_split_importance(model, CORE_FEATURE_NAMES),
        cross_sectional=cross_sectional,
        random_baseline_cross_sectional=random_cross_sectional,
    )


def run_model_b(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> ModelBResult:
    train_set = prepare_training_set(train_df, target_col)
    test_set = prepare_training_set(test_df, target_col)
    train_y = binary_target(train_set.y)
    test_y = binary_target(test_set.y)

    model = fit_classification_model(train_set.X, train_y)
    train_prob = predict_positive_probability(model, train_set.X)
    test_prob = predict_positive_probability(model, test_set.X)

    return ModelBResult(
        target_col=target_col,
        train_metrics=evaluate_classification(train_y.to_numpy(), train_prob),
        test_metrics=evaluate_classification(test_y.to_numpy(), test_prob),
        feature_importance=extract_gain_and_split_importance(model, CORE_FEATURE_NAMES),
    )


def run_model_c(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> ModelCResult:
    train_set = prepare_training_set(train_df, target_col)
    test_set = prepare_training_set(test_df, target_col)

    models = fit_quantile_models(train_set.X, train_set.y)
    predictions = predict_quantiles(models, test_set.X)
    predictions = predictions.assign(
        date=test_set.dates.to_numpy(), ticker=test_set.tickers.to_numpy(),
        actual=test_set.y.to_numpy(),
    )
    return ModelCResult(target_col=target_col, quantile_predictions=predictions)
