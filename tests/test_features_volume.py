from __future__ import annotations

import numpy as np
from conftest import make_synthetic_ohlcv

from features.volume import compute_volume_features


def test_constant_volume_gives_ratio_one_and_zscore_nan() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=6)
    df["volume"] = 50_000.0
    out = compute_volume_features(df)

    assert np.allclose(out["volume_ratio_5d"].dropna(), 1.0)
    assert np.allclose(out["volume_ratio_20d"].dropna(), 1.0)
    # rolling std of a constant series is 0 -> safe_divide must yield NaN,
    # never +/-inf (see features/_utils.py::safe_divide).
    zscore_tail = out["volume_zscore"].iloc[25:]
    assert zscore_tail.isna().all()
    assert not np.isinf(out["volume_zscore"].dropna()).any()


def test_volume_spike_gives_ratio_above_one() -> None:
    n = 30
    df = make_synthetic_ohlcv(n, seed=6)
    df["volume"] = 10_000.0
    df.loc[df.index[-1], "volume"] = 100_000.0
    out = compute_volume_features(df)
    assert out["volume_ratio_5d"].iloc[-1] > 1.0
    assert out["volume_zscore"].iloc[-1] > 0.0


def test_volume_ratio_warmup() -> None:
    df = make_synthetic_ohlcv(40, seed=6)
    out = compute_volume_features(df)
    assert out["volume_ratio_20d"].iloc[:19].isna().all()
    assert out["volume_ratio_20d"].iloc[19:].notna().all()


def test_no_infinite_values_ever() -> None:
    n = 50
    df = make_synthetic_ohlcv(n, seed=6)
    df.loc[df.index[10:20], "volume"] = 0.0
    out = compute_volume_features(df)
    for col in out.columns:
        assert not np.isinf(out[col].dropna()).any(), col
