from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from features.indicators import compute_macd_features, compute_rsi_features


def test_rsi_warmup() -> None:
    df = make_synthetic_ohlcv(60, seed=7)
    out = compute_rsi_features(df)
    assert out["rsi_14"].iloc[:14].isna().all()
    assert out["rsi_14"].iloc[14:].notna().all()
    assert out["rsi_7"].iloc[:7].isna().all()
    assert out["rsi_7"].iloc[7:].notna().all()


def test_rsi_bounded_between_0_and_100() -> None:
    df = make_synthetic_ohlcv(200, seed=7)
    out = compute_rsi_features(df)
    assert (out["rsi_14"].dropna() >= 0).all()
    assert (out["rsi_14"].dropna() <= 100).all()


def test_strictly_increasing_price_gives_rsi_100() -> None:
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100.0, 500.0, n)  # strictly increasing, never flat
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.full(n, 10_000.0),
        }
    )
    out = compute_rsi_features(df)
    assert np.allclose(out["rsi_14"].dropna(), 100.0)


def test_strictly_decreasing_price_gives_rsi_0() -> None:
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(500.0, 100.0, n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.full(n, 10_000.0),
        }
    )
    out = compute_rsi_features(df)
    assert np.allclose(out["rsi_14"].dropna(), 0.0)


def test_flat_price_gives_rsi_50() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = 250.0
    out = compute_rsi_features(df)
    assert np.allclose(out["rsi_14"].dropna(), 50.0)


def test_macd_warmup() -> None:
    df = make_synthetic_ohlcv(80, seed=8)
    out = compute_macd_features(df)
    assert out["macd"].iloc[:25].isna().all()
    assert out["macd"].iloc[25:].notna().all()
    assert out["macd_signal"].iloc[:33].isna().all()
    assert out["macd_signal"].iloc[33:].notna().all()


def test_macd_hist_equals_macd_minus_signal() -> None:
    df = make_synthetic_ohlcv(80, seed=8)
    out = compute_macd_features(df)
    manual = out["macd"] - out["macd_signal"]
    pd.testing.assert_series_equal(
        out["macd_hist"].dropna(), manual.dropna(), check_names=False
    )


def test_flat_price_gives_zero_macd() -> None:
    n = 60
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = 300.0
    out = compute_macd_features(df)
    assert np.allclose(out["macd"].dropna(), 0.0, atol=1e-9)
    assert np.allclose(out["macd_signal"].dropna(), 0.0, atol=1e-9)
    assert np.allclose(out["macd_hist"].dropna(), 0.0, atol=1e-9)
