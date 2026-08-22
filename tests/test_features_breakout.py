from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from features.breakout import compute_breakout_features


def test_todays_close_excluded_from_its_own_rolling_high() -> None:
    """The central anti-lookahead requirement for this feature category:
    if today's Close is a new all-time high, high_20d[t] must NOT be
    that value - it must be the highest Close over the PRIOR 20 sessions
    only.
    """
    n = 40
    df = make_synthetic_ohlcv(n, seed=9)
    df["close"] = 100.0
    spike_idx = 30
    df.loc[df.index[spike_idx], "close"] = 999_999.0  # today's own new high

    out = compute_breakout_features(df)

    assert out["high_20d"].iloc[spike_idx] != 999_999.0
    assert np.isclose(out["high_20d"].iloc[spike_idx], 100.0)
    # The day AFTER the spike, though, the prior-20d high must reflect it.
    assert np.isclose(out["high_20d"].iloc[spike_idx + 1], 999_999.0)


def test_high_nd_matches_manual_shifted_rolling_max() -> None:
    df = make_synthetic_ohlcv(60, seed=9)
    out = compute_breakout_features(df)
    manual = df["close"].shift(1).rolling(20, min_periods=20).max()
    pd.testing.assert_series_equal(out["high_20d"], manual, check_names=False)


def test_breakout_warmup() -> None:
    df = make_synthetic_ohlcv(60, seed=9)
    out = compute_breakout_features(df)
    # high_5d needs 5 prior days + the shift -> first valid at index 5.
    assert out["high_5d"].iloc[:5].isna().all()
    assert out["high_5d"].iloc[5:].notna().all()


def test_todays_close_excluded_from_its_own_rolling_low() -> None:
    """Mirror of the high_Nd test above, for SHORT Breakdown's low_Nd."""
    n = 40
    df = make_synthetic_ohlcv(n, seed=9)
    df["close"] = 100.0
    dip_idx = 30
    df.loc[df.index[dip_idx], "close"] = 0.01  # today's own new low

    out = compute_breakout_features(df)

    assert out["low_20d"].iloc[dip_idx] != 0.01
    assert np.isclose(out["low_20d"].iloc[dip_idx], 100.0)
    assert np.isclose(out["low_20d"].iloc[dip_idx + 1], 0.01)


def test_low_nd_matches_manual_shifted_rolling_min() -> None:
    df = make_synthetic_ohlcv(60, seed=9)
    out = compute_breakout_features(df)
    manual = df["close"].shift(1).rolling(20, min_periods=20).min()
    pd.testing.assert_series_equal(out["low_20d"], manual, check_names=False)
