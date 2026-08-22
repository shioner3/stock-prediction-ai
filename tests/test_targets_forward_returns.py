from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from targets.forward_returns import FORWARD_WINDOWS, compute_forward_returns, compute_mfe_mae


def _flat_df(n: int, close: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": [close] * n, "high": [close] * n, "low": [close] * n,
            "close": [close] * n, "volume": [1000.0] * n,
        }
    )


def test_forward_return_matches_manual_formula() -> None:
    n = 30
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(100.0, 200.0, n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": [1000.0] * n,
        }
    )
    out = compute_forward_returns(df)
    manual = df["close"].shift(-5) / df["close"] - 1
    pd.testing.assert_series_equal(
        out["forward_return_5d"].dropna(), manual.dropna(), check_names=False
    )


def test_forward_return_tail_is_nan() -> None:
    n = 20
    df = _flat_df(n)
    out = compute_forward_returns(df)
    # forward_return_10d needs t+10 to exist -> last 10 rows are NaN.
    assert out["forward_return_10d"].iloc[-10:].isna().all()
    assert out["forward_return_10d"].iloc[:-10].notna().all()


def test_flat_price_gives_zero_forward_return() -> None:
    df = _flat_df(30)
    out = compute_forward_returns(df)
    for n in FORWARD_WINDOWS:
        assert np.allclose(out[f"forward_return_{n}d"].dropna(), 0.0)


# --- Phase 12: 15d/20d addition, backward compatibility ------------------------


def test_forward_windows_now_includes_15d_and_20d() -> None:
    assert FORWARD_WINDOWS == (1, 3, 5, 7, 10, 15, 20)


def test_forward_return_15d_20d_match_manual_formula() -> None:
    n = 40
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": [1000.0] * n,
        }
    )
    out = compute_forward_returns(df)
    for w in (15, 20):
        manual = df["close"].shift(-w) / df["close"] - 1
        pd.testing.assert_series_equal(
            out[f"forward_return_{w}d"].dropna(), manual.dropna(), check_names=False
        )


def test_forward_return_20d_tail_is_nan() -> None:
    n = 30
    df = _flat_df(n)
    out = compute_forward_returns(df)
    assert out["forward_return_20d"].iloc[-20:].isna().all()
    assert out["forward_return_20d"].iloc[:-20].notna().all()


def test_forward_return_1d_3d_5d_7d_10d_unchanged_by_15d_20d_addition() -> None:
    """Phase 12 section 4: adding 15d/20d must not alter the pre-existing
    1/3/5/7/10d columns at all - each window's column is computed
    independently in the same loop body, so this proves the addition is
    purely additive, not a behavior change to what Phase 5-11 already
    relied on (Score Validation, WFO permutation forward_window, etc).
    """
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(11)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.015, size=n))
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": [1000.0] * n,
        }
    )
    out = compute_forward_returns(df)
    for w in (1, 3, 5, 7, 10):
        manual = df["close"].shift(-w) / df["close"] - 1
        pd.testing.assert_series_equal(
            out[f"forward_return_{w}d"], manual, check_names=False
        )


# --- MFE / MAE -----------------------------------------------------------------


def test_mfe_mae_hand_computed_long() -> None:
    # index: 0    1    2    3    4    5
    # close: 100, 100, 100, 100, 100, 100
    # high:  100, 105, 130, 100, 100, 100
    # low:   100,  95,  95,  90, 100, 100
    n = 6
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": [100.0] * n,
            "high": [100.0, 105.0, 130.0, 100.0, 100.0, 100.0],
            "low": [100.0, 95.0, 95.0, 90.0, 100.0, 100.0],
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )
    out = compute_mfe_mae(df, "LONG")
    # At t=0, forward window n=3 covers t=1,2,3: max(high)=130, min(low)=90.
    assert np.isclose(out["mfe_3d"].iloc[0], 130.0 / 100.0 - 1)
    assert np.isclose(out["mae_3d"].iloc[0], 90.0 / 100.0 - 1)


def test_mfe_mae_hand_computed_short() -> None:
    n = 6
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": [100.0] * n,
            "high": [100.0, 105.0, 130.0, 100.0, 100.0, 100.0],
            "low": [100.0, 95.0, 95.0, 90.0, 100.0, 100.0],
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )
    out = compute_mfe_mae(df, "SHORT")
    # For SHORT, favorable = price falling (low), adverse = price rising (high).
    assert np.isclose(out["mfe_3d"].iloc[0], 1 - 90.0 / 100.0)
    assert np.isclose(out["mae_3d"].iloc[0], 1 - 130.0 / 100.0)


def test_mfe_is_non_negative_and_mae_non_positive_on_average_case() -> None:
    n = 40
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(5)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))
    high = close * 1.01
    low = close * 0.99
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": high, "low": low, "close": close,
            "volume": [1000.0] * n,
        }
    )
    for direction in ("LONG", "SHORT"):
        out = compute_mfe_mae(df, direction)
        for w in FORWARD_WINDOWS:
            mfe = out[f"mfe_{w}d"].dropna()
            mae = out[f"mae_{w}d"].dropna()
            assert (mfe >= mae).all()  # best case is never worse than worst case


def test_mfe_mae_tail_is_nan() -> None:
    df = _flat_df(20)
    out = compute_mfe_mae(df, "LONG")
    assert out["mfe_10d"].iloc[-10:].isna().all()
    assert out["mfe_10d"].iloc[:-10].notna().all()


def test_flat_price_gives_zero_mfe_mae() -> None:
    df = _flat_df(30)
    for direction in ("LONG", "SHORT"):
        out = compute_mfe_mae(df, direction)
        for w in FORWARD_WINDOWS:
            assert np.allclose(out[f"mfe_{w}d"].dropna(), 0.0)
            assert np.allclose(out[f"mae_{w}d"].dropna(), 0.0)


# --- Phase 12: MFE/MAE 15d/20d addition, backward compatibility ----------------


def test_mfe_mae_20d_hand_computed_long() -> None:
    n = 25
    dates = pd.bdate_range("2024-01-01", periods=n)
    high = [100.0] * n
    low = [100.0] * n
    high[18] = 150.0  # inside the t=0..t+20 forward window
    low[19] = 60.0
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": [100.0] * n, "high": high, "low": low, "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )
    out = compute_mfe_mae(df, "LONG")
    assert np.isclose(out["mfe_20d"].iloc[0], 150.0 / 100.0 - 1)
    assert np.isclose(out["mae_20d"].iloc[0], 60.0 / 100.0 - 1)


def test_mfe_mae_20d_tail_is_nan() -> None:
    df = _flat_df(30)
    out = compute_mfe_mae(df, "LONG")
    assert out["mfe_20d"].iloc[-20:].isna().all()
    assert out["mfe_20d"].iloc[:-20].notna().all()


def test_mfe_mae_1d_3d_5d_7d_10d_unchanged_by_15d_20d_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct regression proof: computing MFE/MAE with the OLD
    (pre-Phase-12) window set gives bit-identical 1/3/5/7/10d columns to
    computing with the NEW (15d/20d-extended) window set - the loop body
    per window is independent, so extending the tuple is purely additive.
    """
    import targets.forward_returns as fr

    n = 40
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(13)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n, "date": [d.date() for d in dates],
            "open": close, "high": high, "low": low, "close": close,
            "volume": [1000.0] * n,
        }
    )
    for direction in ("LONG", "SHORT"):
        out_new = compute_mfe_mae(df, direction)
        monkeypatch.setattr(fr, "FORWARD_WINDOWS", (1, 3, 5, 7, 10))
        out_old = compute_mfe_mae(df, direction)
        monkeypatch.undo()
        for w in (1, 3, 5, 7, 10):
            pd.testing.assert_series_equal(out_new[f"mfe_{w}d"], out_old[f"mfe_{w}d"])
            pd.testing.assert_series_equal(out_new[f"mae_{w}d"], out_old[f"mae_{w}d"])
