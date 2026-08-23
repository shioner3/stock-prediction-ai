from __future__ import annotations

import numpy as np

from v3.models.evaluate import evaluate_classification, evaluate_regression


def test_regression_metrics_perfect_prediction() -> None:
    y = np.array([0.01, -0.02, 0.03, 0.0])
    metrics = evaluate_regression(y, y)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == 1.0
    assert metrics.pearson == 1.0


def test_regression_metrics_reports_negative_r2_for_bad_predictions() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.normal(0, 0.02, size=200)
    y_pred = rng.normal(0, 0.02, size=200)  # unrelated to y_true
    metrics = evaluate_regression(y_true, y_pred)
    assert metrics.n == 200
    assert metrics.mae > 0


def test_classification_metrics_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.99, 0.98])
    metrics = evaluate_classification(y_true, y_prob)
    assert metrics.roc_auc == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.positive_rate == 0.5


def test_classification_metrics_brier_score_bounded() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.3, 0.7, 0.4, 0.6, 0.55])
    metrics = evaluate_classification(y_true, y_prob)
    assert 0.0 <= metrics.brier_score <= 1.0
