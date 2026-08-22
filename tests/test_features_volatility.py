from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from features.volatility import compute_volatility_features, true_range


def test_true_range_matches_hand_computed_values() -> None:
    # Row 0 has no prior close -> TR falls back to High-Low.
    # Row 1: prevClose=100 -> H-L=12, |H-prevC|=10, |L-prevC|=2 -> TR=12
    # Row 2: prevClose=105 -> H-L=18, |H-prevC|=3,  |L-prevC|=15 -> TR=18
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * 3,
            "date": pd.bdate_range("2024-01-01", periods=3),
            "open": [100.0, 105.0, 95.0],
            "high": [105.0, 110.0, 108.0],
            "low": [95.0, 98.0, 90.0],
            "close": [100.0, 105.0, 95.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )
    tr = true_range(df)
    assert np.isclose(tr.iloc[0], 10.0)  # 105-95
    assert np.isclose(tr.iloc[1], 12.0)
    assert np.isclose(tr.iloc[2], 18.0)


def test_atr_warmup() -> None:
    df = make_synthetic_ohlcv(60, seed=5)
    out = compute_volatility_features(df)
    assert out["atr"].iloc[:14].isna().all()
    assert out["atr"].iloc[14:].notna().all()


def test_atr_is_positive_for_moving_prices() -> None:
    df = make_synthetic_ohlcv(60, seed=5)
    out = compute_volatility_features(df)
    assert (out["atr"].dropna() > 0).all()


def test_zero_range_series_gives_zero_atr() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=1)
    df["open"] = df["high"] = df["low"] = df["close"] = 100.0
    out = compute_volatility_features(df)
    assert np.allclose(out["atr"].dropna(), 0.0)
    assert np.allclose(out["volatility_20d"].dropna(), 0.0)


def test_atr_pct_is_atr_over_close() -> None:
    df = make_synthetic_ohlcv(60, seed=5)
    out = compute_volatility_features(df)
    manual = out["atr"] / df["close"]
    pd.testing.assert_series_equal(
        out["atr_pct"].dropna(), manual.dropna(), check_names=False
    )
