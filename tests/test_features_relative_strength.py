from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic_ohlcv

from features.momentum import compute_return
from features.relative_strength import RS_WINDOWS, compute_relative_strength_features


def _constant_rate_ohlcv(
    n: int, total_return_over_20d: float, ticker: str, base: float = 100.0
) -> pd.DataFrame:
    """A price series with a constant daily growth rate chosen so that
    the trailing 20-day return is exactly `total_return_over_20d` at
    every row once warmed up - makes the "stock +10% / market +5%"
    style test cases exact rather than approximate.
    """
    daily_rate = (1 + total_return_over_20d) ** (1 / 20) - 1
    close = base * (1 + daily_rate) ** np.arange(n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.full(n, 10_000.0),
        }
    )


# --- Mathematical property tests (section "16. Mathematical tests") --------


def test_case1_stock_up_10pct_market_up_5pct_gives_rs_5pct() -> None:
    n = 60
    stock = _constant_rate_ohlcv(n, 0.10, "TEST")
    market = _constant_rate_ohlcv(n, 0.05, "TOPIX")
    out = compute_relative_strength_features(stock, market)
    tail = out["rs_20d"].iloc[30:]
    assert np.allclose(tail, 0.05, atol=1e-9)


def test_case2_stock_up_5pct_market_up_10pct_gives_rs_negative_5pct() -> None:
    n = 60
    stock = _constant_rate_ohlcv(n, 0.05, "TEST")
    market = _constant_rate_ohlcv(n, 0.10, "TOPIX")
    out = compute_relative_strength_features(stock, market)
    tail = out["rs_20d"].iloc[30:]
    assert np.allclose(tail, -0.05, atol=1e-9)


def test_case3_identical_returns_give_zero_rs() -> None:
    n = 60
    stock = _constant_rate_ohlcv(n, 0.07, "TEST")
    market = _constant_rate_ohlcv(n, 0.07, "TOPIX")
    out = compute_relative_strength_features(stock, market)
    tail = out["rs_20d"].iloc[30:]
    assert np.allclose(tail, 0.0, atol=1e-9)


def test_case4_zero_market_return_gives_rs_equal_to_stock_return_no_inf() -> None:
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    stock = make_synthetic_ohlcv(n, seed=30, ticker="TEST")
    market = pd.DataFrame(
        {
            "ticker": ["TOPIX"] * n,
            "date": [d.date() for d in dates],
            "open": np.full(n, 300.0), "high": np.full(n, 300.0),
            "low": np.full(n, 300.0), "close": np.full(n, 300.0),
            "volume": np.full(n, 10_000.0),
        }
    )
    out = compute_relative_strength_features(stock, market)
    stock_return_20d = compute_return(stock["close"], 20)

    assert not np.isinf(out["rs_20d"].dropna()).any()
    pd.testing.assert_series_equal(
        out["rs_20d"].dropna(), stock_return_20d.dropna(), check_names=False
    )


# --- Formula consistency with features/momentum.py -------------------------


def test_rs_uses_the_same_return_formula_as_momentum() -> None:
    n = 80
    stock = make_synthetic_ohlcv(n, seed=31, ticker="TEST")
    market = make_synthetic_ohlcv(n, seed=32, ticker="TOPIX")
    out = compute_relative_strength_features(stock, market)

    for window in RS_WINDOWS:
        stock_return = compute_return(stock["close"], window)
        market_by_date = pd.Series(
            market["close"].to_numpy(), index=pd.to_datetime(market["date"])
        ).sort_index()
        market_return = compute_return(market_by_date, window).reindex(
            pd.to_datetime(stock["date"])
        )
        manual_rs = stock_return.to_numpy() - market_return.to_numpy()
        assert np.allclose(
            out[f"rs_{window}d"].dropna(), manual_rs[~np.isnan(manual_rs)], atol=1e-12
        )


# --- Market unavailable / None fallback -------------------------------------


def test_market_df_none_gives_all_nan_rs_never_zero() -> None:
    stock = make_synthetic_ohlcv(60, seed=33, ticker="TEST")
    out = compute_relative_strength_features(stock, None)
    for window in RS_WINDOWS:
        assert out[f"rs_{window}d"].isna().all()


def test_empty_market_df_also_gives_all_nan() -> None:
    stock = make_synthetic_ohlcv(60, seed=33, ticker="TEST")
    empty_market = stock.iloc[0:0]
    out = compute_relative_strength_features(stock, empty_market)
    for window in RS_WINDOWS:
        assert out[f"rs_{window}d"].isna().all()


# --- Date alignment (section "7/8. 市場と個別株の日付不一致") --------------


def test_missing_market_dates_give_nan_not_forward_filled() -> None:
    n = 60
    stock = make_synthetic_ohlcv(n, seed=34, ticker="TEST")
    market = make_synthetic_ohlcv(n, seed=35, ticker="TOPIX")

    # Simulate a 5-day trading halt in the market benchmark: drop rows
    # 40..44 entirely (not just NaN them - genuinely absent dates).
    gap_dates = set(market["date"].iloc[40:45])
    market_with_gap = market[~market["date"].isin(gap_dates)].reset_index(drop=True)

    out = compute_relative_strength_features(stock, market_with_gap)

    stock_dates_in_gap = stock["date"].isin(gap_dates)
    assert stock_dates_in_gap.any()
    assert out.loc[stock_dates_in_gap, "rs_5d"].isna().all()

    # And critically: the NaN must not have been filled from the last
    # available (pre-gap) market value - reindex() must never receive a
    # `method=` argument that would do that.
    market_value_before_gap = market.loc[market["date"] == market["date"].iloc[39], "close"]
    assert len(market_value_before_gap) == 1  # sanity check on the fixture


def test_extra_market_dates_outside_stock_range_are_ignored() -> None:
    n = 60
    stock = make_synthetic_ohlcv(n, seed=36, ticker="TEST")
    market_full = make_synthetic_ohlcv(n + 20, seed=37, ticker="TOPIX")  # longer history

    out_full = compute_relative_strength_features(stock, market_full)
    out_trimmed = compute_relative_strength_features(stock, market_full.iloc[:n])

    for window in RS_WINDOWS:
        pd.testing.assert_series_equal(
            out_full[f"rs_{window}d"], out_trimmed[f"rs_{window}d"], check_names=False
        )


# --- Warmup ------------------------------------------------------------------


def test_rs_warmup_with_full_date_overlap() -> None:
    n = 100
    stock = make_synthetic_ohlcv(n, seed=38, ticker="TEST")
    market = make_synthetic_ohlcv(n, seed=39, ticker="TOPIX")
    out = compute_relative_strength_features(stock, market)

    assert out["rs_20d"].iloc[:20].isna().all()
    assert out["rs_20d"].iloc[20:].notna().all()


# --- No-lookahead: row-order independence (section "Test D") ---------------


def test_market_row_order_does_not_affect_aligned_result() -> None:
    n = 80
    stock = make_synthetic_ohlcv(n, seed=40, ticker="TEST")
    market = make_synthetic_ohlcv(n, seed=41, ticker="TOPIX")
    market_shuffled = market.sample(frac=1, random_state=7).reset_index(drop=True)

    out_ordered = compute_relative_strength_features(stock, market)
    out_shuffled = compute_relative_strength_features(stock, market_shuffled)

    for window in RS_WINDOWS:
        pd.testing.assert_series_equal(
            out_ordered[f"rs_{window}d"], out_shuffled[f"rs_{window}d"], check_names=False
        )


def test_stock_row_order_is_not_relied_upon_by_market_alignment() -> None:
    """Regression guard: date alignment must key off stock_df["date"]
    values, not stock_df's positional index - duplicating a date-sorted
    frame with a shuffled positional index must give the same result.
    """
    n = 60
    stock = make_synthetic_ohlcv(n, seed=42, ticker="TEST")
    market = make_synthetic_ohlcv(n, seed=43, ticker="TOPIX")

    stock_reindexed = stock.copy()
    stock_reindexed.index = stock_reindexed.index + 1000  # non-default positional index

    out_default = compute_relative_strength_features(stock, market)
    out_reindexed = compute_relative_strength_features(stock_reindexed, market)

    assert out_default["rs_20d"].to_numpy()[20:] == pytest.approx(
        out_reindexed["rs_20d"].to_numpy()[20:]
    )
