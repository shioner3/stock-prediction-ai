from __future__ import annotations

import numpy as np
import pandas as pd

from scoring.validation import (
    FIXED_BUCKET_LABELS,
    QUANTILE_BUCKET_LABELS,
    assign_fixed_buckets,
    assign_quantile_buckets,
    bucket_ratio,
    bucket_spread,
    check_monotonicity,
    compute_bucket_metrics,
    extract_bucket_returns,
    monotonicity_correlation,
)


def test_assign_fixed_buckets_boundaries() -> None:
    scores = pd.Series([0, 19, 20, 39, 40, 59, 60, 79, 80, 100])
    buckets = assign_fixed_buckets(scores)
    assert list(buckets.astype(str)) == [
        "0-19", "0-19", "20-39", "20-39", "40-59",
        "40-59", "60-79", "60-79", "80-100", "80-100",
    ]


def test_assign_quantile_buckets_gives_roughly_equal_groups() -> None:
    scores = pd.Series(list(range(100)))
    buckets = assign_quantile_buckets(scores, n_buckets=5)
    counts = buckets.value_counts()
    assert len(counts) == 5
    assert counts.min() >= 15  # roughly 20 each, allow slack


def test_assign_quantile_buckets_handles_degenerate_distribution() -> None:
    scores = pd.Series([50] * 20)  # all identical - qcut can't form 5 distinct bins
    buckets = assign_quantile_buckets(scores, n_buckets=5)
    assert len(buckets) == 20
    assert not buckets.isna().any()


def _scored_df(returns: list[float], scores: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({"total_score": scores, "return_1d": returns})
    df["bucket"] = assign_fixed_buckets(df["total_score"])
    return df


def test_compute_bucket_metrics_groups_correctly() -> None:
    df = _scored_df(
        returns=[0.01, 0.02, -0.01, 0.05, 0.06, -0.02],
        scores=[10, 15, 25, 65, 70, 85],
    )
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    assert metrics["0-19"].n_trades == 2
    assert metrics["20-39"].n_trades == 1
    assert metrics["60-79"].n_trades == 2
    assert metrics["80-100"].n_trades == 1


def test_compute_bucket_metrics_drops_nan_returns() -> None:
    df = pd.DataFrame({"total_score": [10, 15, 20], "return_1d": [0.01, np.nan, 0.02]})
    df["bucket"] = assign_fixed_buckets(df["total_score"])
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    assert metrics["0-19"].n_trades == 1  # the NaN row was dropped


def test_compute_bucket_metrics_empty_input() -> None:
    df = pd.DataFrame({"total_score": [], "return_1d": []})
    df["bucket"] = assign_fixed_buckets(df["total_score"])
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    assert metrics == {}


# --- Monotonicity ------------------------------------------------------------


def test_monotonicity_true_for_increasing_returns() -> None:
    df = _scored_df(
        returns=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        scores=[5, 5, 25, 25, 45, 45, 65, 65, 90, 90],
    )
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    assert check_monotonicity(metrics, FIXED_BUCKET_LABELS) is True


def test_monotonicity_false_for_decreasing_returns() -> None:
    df = _scored_df(
        returns=[0.10, 0.09, 0.05, 0.04, 0.01],
        scores=[5, 5, 45, 45, 90],
    )
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    assert check_monotonicity(metrics, FIXED_BUCKET_LABELS) is False


def test_monotonicity_none_when_insufficient_buckets() -> None:
    df = _scored_df(returns=[0.01, 0.02], scores=[5, 8])  # only bucket "0-19" populated
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    assert check_monotonicity(metrics, FIXED_BUCKET_LABELS) is None


def test_monotonicity_correlation_perfect_positive() -> None:
    df = _scored_df(
        returns=[0.01, 0.03, 0.05, 0.07, 0.09],
        scores=[5, 25, 45, 65, 90],
    )
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    corr = monotonicity_correlation(metrics, FIXED_BUCKET_LABELS)
    assert corr is not None
    assert np.isclose(corr, 1.0, atol=1e-6)


def test_monotonicity_correlation_none_when_constant() -> None:
    df = _scored_df(returns=[0.02, 0.02, 0.02, 0.02], scores=[5, 25, 45, 65])
    metrics = compute_bucket_metrics(df, "bucket", "return_1d")
    corr = monotonicity_correlation(metrics, FIXED_BUCKET_LABELS)
    assert corr is None


# --- Q5-Q1 spread / ratio / raw-return extraction (Phase 6.5 section 21) -----


def _quantile_scored_df(returns: list[float], scores: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({"total_score": scores, "return_1d": returns})
    df["qbucket"] = assign_quantile_buckets(df["total_score"], n_buckets=5)
    return df


def test_bucket_spread_hand_computed() -> None:
    # 20 rows, evenly spaced scores 0..95 in steps of 5 -> 5 quantile
    # buckets of 4 rows each, Q1 = lowest scores, Q5 = highest.
    scores = list(range(0, 100, 5))
    returns = [s / 1000 for s in scores]  # perfectly monotonic by construction
    df = _quantile_scored_df(returns, scores)
    metrics = compute_bucket_metrics(df, "qbucket", "return_1d")

    spread = bucket_spread(metrics, "Q5", "Q1")
    assert spread is not None
    assert np.isclose(spread, metrics["Q5"].average_return - metrics["Q1"].average_return)
    assert spread > 0


def test_bucket_spread_none_when_bucket_missing() -> None:
    # All-identical scores -> qcut cannot form distinct bins -> everything
    # falls into "Q1" (assign_quantile_buckets' degenerate-distribution
    # fallback) -> "Q5" has no trades at all.
    df = _quantile_scored_df(returns=[0.01, 0.02], scores=[50, 50])
    metrics = compute_bucket_metrics(df, "qbucket", "return_1d")
    assert bucket_spread(metrics, "Q5", "Q1") is None


def test_bucket_ratio_hand_computed() -> None:
    scores = list(range(0, 100, 5))
    returns = [s / 1000 for s in scores]
    df = _quantile_scored_df(returns, scores)
    metrics = compute_bucket_metrics(df, "qbucket", "return_1d")

    ratio = bucket_ratio(metrics, "Q5", "Q1")
    assert ratio is not None
    assert np.isclose(ratio, metrics["Q5"].average_return / metrics["Q1"].average_return)


def test_bucket_ratio_none_when_low_bucket_return_is_zero() -> None:
    df = pd.DataFrame({"total_score": [1, 1, 90, 90], "return_1d": [0.0, 0.0, 0.05, 0.05]})
    df["qbucket"] = ["Q1", "Q1", "Q5", "Q5"]
    metrics = compute_bucket_metrics(df, "qbucket", "return_1d")
    assert bucket_ratio(metrics, "Q5", "Q1") is None


def test_extract_bucket_returns_filters_to_one_bucket_and_drops_nan() -> None:
    df = pd.DataFrame(
        {
            "total_score": [1, 1, 90, 90],
            "return_1d": [0.01, np.nan, 0.05, 0.06],
        }
    )
    df["qbucket"] = ["Q1", "Q1", "Q5", "Q5"]
    q5_returns = extract_bucket_returns(df, "qbucket", "return_1d", "Q5")
    assert sorted(q5_returns) == [0.05, 0.06]
    q1_returns = extract_bucket_returns(df, "qbucket", "return_1d", "Q1")
    assert list(q1_returns) == [0.01]  # the NaN row was dropped


def test_extract_bucket_returns_empty_when_bucket_absent() -> None:
    df = pd.DataFrame({"total_score": [1, 2], "return_1d": [0.01, 0.02]})
    df["qbucket"] = ["Q1", "Q1"]
    result = extract_bucket_returns(df, "qbucket", "return_1d", "Q5")
    assert len(result) == 0


def test_quantile_bucket_labels_constant() -> None:
    assert QUANTILE_BUCKET_LABELS == ["Q1", "Q2", "Q3", "Q4", "Q5"]
