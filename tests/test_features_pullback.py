from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from features.pullback import compute_pullback_features


def test_distance_from_recent_high_is_zero_when_today_is_the_high() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=10)
    df["close"] = np.linspace(100.0, 200.0, n)  # strictly increasing -> today always the high
    out = compute_pullback_features(df, recent_high_window=20)
    tail = out.iloc[25:]
    assert np.allclose(tail["distance_from_recent_high"], 0.0)
    assert np.allclose(tail["pullback_depth"], 0.0)


def test_distance_from_recent_high_never_positive() -> None:
    df = make_synthetic_ohlcv(80, seed=11)
    out = compute_pullback_features(df, recent_high_window=20)
    assert (out["distance_from_recent_high"].dropna() <= 1e-12).all()
    assert (out["pullback_depth"].dropna() >= -1e-12).all()


def test_pullback_depth_is_negative_of_distance_from_recent_high() -> None:
    df = make_synthetic_ohlcv(80, seed=11)
    out = compute_pullback_features(df, recent_high_window=20)
    manual = -out["distance_from_recent_high"]
    pd.testing.assert_series_equal(
        out["pullback_depth"].dropna(), manual.dropna(), check_names=False
    )


def test_recent_high_window_is_configurable() -> None:
    df = make_synthetic_ohlcv(80, seed=11)
    out_10 = compute_pullback_features(df, recent_high_window=10)
    out_40 = compute_pullback_features(df, recent_high_window=40)
    # Different windows -> first valid row differs, and (for random-walk
    # data) values differ too once both are valid.
    assert out_10["distance_from_recent_high"].iloc[:9].isna().all()
    assert out_40["distance_from_recent_high"].iloc[:39].isna().all()
    assert not out_10["distance_from_recent_high"].equals(out_40["distance_from_recent_high"])


def test_consecutive_down_days_matches_hand_computed_sequence() -> None:
    # index:  0    1    2    3   4   5    6    7   8   9   10
    close = [100, 105, 103, 101, 99, 102, 101, 100, 99, 98, 97]
    # expected consecutive down-day counts ending at each index
    # (row 0 undefined: no prior close):
    expected = [np.nan, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5]

    n = len(close)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": close, "high": close, "low": close, "close": close,
            "volume": [1000.0] * n,
        }
    )
    out = compute_pullback_features(df, recent_high_window=5)
    result = out["consecutive_down_days"].tolist()

    assert np.isnan(result[0])
    assert result[1:] == expected[1:]


def test_distance_from_sma_matches_manual_formula() -> None:
    df = make_synthetic_ohlcv(40, seed=12)
    out = compute_pullback_features(df, recent_high_window=10)
    manual = df["close"] / df["close"].rolling(5, min_periods=5).mean() - 1
    pd.testing.assert_series_equal(
        out["distance_from_sma5"].dropna(), manual.dropna(), check_names=False
    )


# --- distance_from_recent_low / bounce_depth (SHORT Pullback's mirror) -----


def test_distance_from_recent_low_is_zero_when_today_is_the_low() -> None:
    n = 40
    df = make_synthetic_ohlcv(n, seed=10)
    df["close"] = np.linspace(200.0, 100.0, n)  # strictly decreasing -> today always the low
    out = compute_pullback_features(df, recent_high_window=20)
    tail = out.iloc[25:]
    assert np.allclose(tail["distance_from_recent_low"], 0.0)
    assert np.allclose(tail["bounce_depth"], 0.0)


def test_distance_from_recent_low_never_negative() -> None:
    df = make_synthetic_ohlcv(80, seed=11)
    out = compute_pullback_features(df, recent_high_window=20)
    assert (out["distance_from_recent_low"].dropna() >= -1e-12).all()
    assert (out["bounce_depth"].dropna() >= -1e-12).all()


def test_bounce_depth_equals_distance_from_recent_low() -> None:
    df = make_synthetic_ohlcv(80, seed=11)
    out = compute_pullback_features(df, recent_high_window=20)
    pd.testing.assert_series_equal(
        out["bounce_depth"].dropna(), out["distance_from_recent_low"].dropna(),
        check_names=False,
    )
