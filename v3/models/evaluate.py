"""Basic evaluation metrics (spec section 11) - regression and
classification. This module ONLY computes and returns numbers; nothing
here concludes "the model works" or "the model is useless" - that
interpretation is explicitly out of scope for Phase V3-2 (spec
section 24).

Pearson/Spearman correlation are computed via plain numpy (Pearson =
np.corrcoef; Spearman = Pearson on ranks), matching this project's
existing pattern (`v2/validation/ic.py`) rather than reaching for
`scipy.stats` even though scipy is now transitively installed (a
LightGBM dependency) - kept consistent with the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    r2: float | None
    pearson: float | None
    spearman: float | None


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return RegressionMetrics(
        n=n, mae=mae, rmse=rmse, r2=r2,
        pearson=_pearson(y_true, y_pred), spearman=_spearman(y_true, y_pred),
    )


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    roc_auc: float | None
    log_loss: float | None
    brier_score: float
    accuracy: float
    positive_rate: float


def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    y_pred = (y_prob >= 0.5).astype(int)

    unique_classes = np.unique(y_true)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(unique_classes) == 2 else None
    logloss = (
        float(log_loss(y_true, y_prob, labels=[0, 1])) if len(unique_classes) == 2 else None
    )

    return ClassificationMetrics(
        n=n,
        roc_auc=roc_auc,
        log_loss=logloss,
        brier_score=float(brier_score_loss(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        positive_rate=float(y_true.mean()),
    )
