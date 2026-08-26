from __future__ import annotations

from pathlib import Path

from v3_2_test_helpers import build_small_train_test

from v3.models.orchestrator import run_model_a, run_model_b, run_model_c

TARGET_COL = "target_raw_5d"


def test_run_model_a_end_to_end_structure(tmp_path: Path) -> None:
    train, test = build_small_train_test(tmp_path, n_tickers=10, n_days=500)
    result = run_model_a(train, test, TARGET_COL)
    assert result.target_col == TARGET_COL
    assert result.train_metrics.n > 0
    assert result.test_metrics.n > 0
    assert len(result.feature_importance) > 0
    assert len(result.cross_sectional.bucket_stats) <= 5
    assert len(result.random_baseline_cross_sectional.bucket_stats) <= 5


def test_run_model_a_is_reproducible(tmp_path: Path) -> None:
    train, test = build_small_train_test(tmp_path, n_tickers=8, n_days=450)
    r1 = run_model_a(train, test, TARGET_COL)
    r2 = run_model_a(train, test, TARGET_COL)
    assert r1.test_metrics == r2.test_metrics


def test_run_model_b_end_to_end_structure(tmp_path: Path) -> None:
    train, test = build_small_train_test(tmp_path, n_tickers=10, n_days=500)
    result = run_model_b(train, test, TARGET_COL)
    assert 0.0 <= result.test_metrics.brier_score <= 1.0
    assert 0.0 <= result.test_metrics.accuracy <= 1.0
    assert len(result.feature_importance) > 0


def test_run_model_c_end_to_end_structure(tmp_path: Path) -> None:
    train, test = build_small_train_test(tmp_path, n_tickers=10, n_days=500)
    result = run_model_c(train, test, TARGET_COL)
    assert {"q0.1", "q0.5", "q0.9"} <= set(result.quantile_predictions.columns)
    assert len(result.quantile_predictions) > 0


def test_can_switch_target_column_via_config(tmp_path: Path) -> None:
    """spec section 5: target column must be switchable, not hardcoded."""
    train, test = build_small_train_test(tmp_path, n_tickers=8, n_days=450)
    for target_col in (
        "target_raw_10d", "target_topix_relative_5d", "target_vol_adjusted_5d",
        "target_risk_adjusted_5d",
    ):
        result = run_model_a(train, test, target_col)
        assert result.target_col == target_col
        assert result.test_metrics.n > 0
