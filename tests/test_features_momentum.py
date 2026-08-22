from __future__ import annotations

import numpy as np
from conftest import make_synthetic_ohlcv

from features.momentum import compute_momentum_features


def test_return_matches_manual_formula() -> None:
    df = make_synthetic_ohlcv(40, seed=4)
    out = compute_momentum_features(df)
    manual = df["close"] / df["close"].shift(5) - 1
    assert np.allclose(out["return_5d"].dropna(), manual.dropna())


def test_return_warmup() -> None:
    df = make_synthetic_ohlcv(40, seed=4)
    out = compute_momentum_features(df)
    assert out["return_10d"].iloc[:10].isna().all()
    assert out["return_10d"].iloc[10:].notna().all()


def test_flat_price_gives_zero_return() -> None:
    n = 30
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = 777.0
    out = compute_momentum_features(df)
    assert np.allclose(out["return_1d"].dropna(), 0.0)
    assert np.allclose(out["return_20d"].dropna(), 0.0)


def test_doubling_price_gives_return_of_one() -> None:
    n = 25
    df = make_synthetic_ohlcv(n, seed=1)
    df["close"] = 100.0
    df.loc[df.index[-1], "close"] = 200.0
    out = compute_momentum_features(df)
    assert np.isclose(out["return_1d"].iloc[-1], 1.0)
