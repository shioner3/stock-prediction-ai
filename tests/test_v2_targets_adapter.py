from __future__ import annotations

import pandas as pd
from conftest import make_synthetic_ohlcv

from v2.targets_adapter import FORWARD_WINDOWS, build_v2_forward_targets


def test_all_forward_windows_present() -> None:
    ohlcv = make_synthetic_ohlcv(100, seed=1, ticker="T0")
    targets = build_v2_forward_targets(ohlcv)
    for n in (5, 10, 15, 20):
        assert f"forward_return_{n}d" in targets.columns
    assert set(FORWARD_WINDOWS) == {1, 3, 5, 7, 10, 15, 20}


def test_last_n_rows_are_nan_boundary_condition() -> None:
    ohlcv = make_synthetic_ohlcv(100, seed=1, ticker="T0")
    targets = build_v2_forward_targets(ohlcv)
    assert targets["forward_return_20d"].iloc[-20:].isna().all()
    assert targets["forward_return_20d"].iloc[:-20].notna().all()
    assert targets["forward_return_5d"].iloc[-5:].isna().all()
    assert pd.notna(targets["forward_return_5d"].iloc[-6])


def test_forward_return_formula() -> None:
    ohlcv = make_synthetic_ohlcv(50, seed=1, ticker="T0")
    targets = build_v2_forward_targets(ohlcv)
    close = ohlcv["close"]
    expected_5d = close.shift(-5) / close - 1
    import numpy as np

    assert np.allclose(
        targets["forward_return_5d"].to_numpy(dtype=float),
        expected_5d.to_numpy(dtype=float),
        equal_nan=True,
    )
