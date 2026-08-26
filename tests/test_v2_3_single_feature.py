from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.causal.feature_stats import percentile_column
from v2.causal.single_feature import (
    FeatureDirection,
    analyze_single_feature,
    assign_feature_buckets,
    classify_feature_direction,
)


def _synthetic_panel(n_days: int = 40, n_tickers: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        raw = rng.normal(size=n_tickers)
        # A monotonic relationship: higher percentile -> higher return.
        pct = pd.Series(raw).rank(pct=True).to_numpy()
        ret = (pct - 0.5) * 0.02 + rng.normal(0, 0.001, size=n_tickers)
        for i in range(n_tickers):
            rows.append((d, f"T{i}", pct[i], ret[i]))
    df = pd.DataFrame(rows, columns=["date", "ticker", percentile_column("return_20d"), "ret"])
    return df


def test_assign_feature_buckets_produces_five_quantiles() -> None:
    panel = _synthetic_panel()
    buckets = assign_feature_buckets(panel, "return_20d")
    assert set(buckets.unique()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}


def test_analyze_single_feature_detects_positive_monotonic_relationship() -> None:
    panel = _synthetic_panel()
    result = analyze_single_feature(panel, "momentum", "return_20d", "ret", window_days=5)
    assert result.q5_q1_spread is not None and result.q5_q1_spread > 0
    assert result.monotonicity.spearman is not None and result.monotonicity.spearman > 0.9


def test_classify_feature_direction_positive_predictive() -> None:
    panel = _synthetic_panel()
    result = analyze_single_feature(panel, "momentum", "return_20d", "ret", window_days=5)
    assert classify_feature_direction(result) == FeatureDirection.POSITIVE_PREDICTIVE


def test_classify_feature_direction_no_evidence_for_flat_relationship() -> None:
    rng = np.random.default_rng(7)
    n_days, n_tickers = 40, 30
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        raw = rng.normal(size=n_tickers)
        pct = pd.Series(raw).rank(pct=True).to_numpy()
        ret = rng.normal(0, 0.001, size=n_tickers)  # no relationship to pct at all
        for i in range(n_tickers):
            rows.append((d, f"T{i}", pct[i], ret[i]))
    panel = pd.DataFrame(rows, columns=["date", "ticker", percentile_column("return_20d"), "ret"])
    result = analyze_single_feature(panel, "momentum", "return_20d", "ret", window_days=5)
    assert classify_feature_direction(result) == FeatureDirection.NO_EVIDENCE


def test_classify_feature_direction_negative_predictive() -> None:
    rng = np.random.default_rng(3)
    n_days, n_tickers = 40, 30
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        raw = rng.normal(size=n_tickers)
        pct = pd.Series(raw).rank(pct=True).to_numpy()
        ret = -(pct - 0.5) * 0.02 + rng.normal(0, 0.001, size=n_tickers)  # inverted
        for i in range(n_tickers):
            rows.append((d, f"T{i}", pct[i], ret[i]))
    panel = pd.DataFrame(rows, columns=["date", "ticker", percentile_column("return_20d"), "ret"])
    result = analyze_single_feature(panel, "momentum", "return_20d", "ret", window_days=5)
    assert classify_feature_direction(result) == FeatureDirection.NEGATIVE_PREDICTIVE
