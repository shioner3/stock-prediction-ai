from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v3.features.registry import CORE_FEATURE_NAMES
from v3.models.model_manifest import build_model_manifest, compute_model_hash, save_model_manifest
from v3.models.regression import fit_regression_model


def _synthetic_Xy(n: int = 300, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({name: rng.normal(size=n) for name in CORE_FEATURE_NAMES[:8]})
    y = X.iloc[:, 0] * 0.02 + rng.normal(0, 0.01, size=n)
    return X, y


def test_model_hash_is_deterministic_given_same_data_and_seed() -> None:
    X, y = _synthetic_Xy()
    model1 = fit_regression_model(X, y)
    model2 = fit_regression_model(X, y)
    assert compute_model_hash(model1) == compute_model_hash(model2)


def test_model_hash_changes_with_different_training_data() -> None:
    X, y = _synthetic_Xy(seed=1)
    X2, y2 = _synthetic_Xy(seed=2)
    model1 = fit_regression_model(X, y)
    model2 = fit_regression_model(X2, y2)
    assert compute_model_hash(model1) != compute_model_hash(model2)


def test_build_model_manifest_has_no_leftover_model_hash_none() -> None:
    X, y = _synthetic_Xy()
    model = fit_regression_model(X, y)
    train_df = pd.DataFrame({"date": ["2024-01-01"] * len(X)})
    test_df = pd.DataFrame({"date": ["2024-06-01"] * 10})
    manifest = build_model_manifest(
        "regression", "target_raw_5d", model, dataset_hash="abc123", train_df=train_df,
        test_df=test_df,
    )
    assert manifest.model_hash
    assert manifest.dataset_hash == "abc123"
    assert manifest.train_rows == len(X)


def test_save_model_manifest_writes_valid_json(tmp_path: Path) -> None:
    X, y = _synthetic_Xy()
    model = fit_regression_model(X, y)
    train_df = pd.DataFrame({"date": ["2024-01-01"] * len(X)})
    test_df = pd.DataFrame({"date": ["2024-06-01"] * 10})
    manifest = build_model_manifest(
        "regression", "target_raw_5d", model, dataset_hash=None, train_df=train_df, test_df=test_df
    )
    path = tmp_path / "manifest.json"
    save_model_manifest(manifest, path)
    assert path.exists()
    assert "model_hash" in path.read_text(encoding="utf-8")
