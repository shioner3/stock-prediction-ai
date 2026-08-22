from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from features.trend import compute_ma_distance_features, compute_trend_features


def test_sma_matches_manual_rolling_mean() -> None:
    df = make_synthetic_ohlcv(60, seed=3)
    out = compute_trend_features(df)
    expected = df["close"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(out["sma_20"], expected, check_names=False)


def test_sma_warmup_is_exactly_window_minus_one_nans() -> None:
    df = make_synthetic_ohlcv(60, seed=3)
    out = compute_trend_features(df)
    assert out["sma_10"].iloc[:9].isna().all()
    assert out["sma_10"].iloc[9:].notna().all()


def test_ema_warmup_matches_sma_convention() -> None:
    df = make_synthetic_ohlcv(60, seed=3)
    out = compute_trend_features(df)
    assert out["ema_20"].iloc[:19].isna().all()
    assert out["ema_20"].iloc[19:].notna().all()


def test_constant_price_gives_ma_equal_to_price() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = 1234.5
    out = compute_trend_features(df)
    assert np.allclose(out["sma_20"].dropna(), 1234.5)
    assert np.allclose(out["ema_20"].dropna(), 1234.5)
    assert np.allclose(out["sma_20_slope"].dropna(), 0.0)


def test_ma_distance_zero_when_close_equals_sma() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = 500.0
    out = compute_ma_distance_features(df)
    assert np.allclose(out["close_to_sma_20"].dropna(), 0.0)


def test_ma_distance_positive_when_close_above_average() -> None:
    n = 40
    close = np.concatenate([np.full(30, 100.0), np.full(10, 200.0)])
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = close
    out = compute_ma_distance_features(df)
    assert out["close_to_sma_20"].iloc[-1] > 0
