from __future__ import annotations

import numpy as np
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from targets.forward_returns import compute_forward_returns
from v3.targets.compute import compute_v3_targets
from v3.targets.registry import HORIZONS, TARGET_COLUMN_NAMES


def _stock_and_market(n_days: int = 200):
    market = make_synthetic_ohlcv(n_days, seed=999, ticker="TOPIX")
    stock_ohlcv = make_synthetic_ohlcv(n_days, seed=1, ticker="T0")
    stock_panel = compute_feature_panel(stock_ohlcv, market_df=market)
    return stock_panel, market


def test_all_16_target_columns_present() -> None:
    stock_panel, market = _stock_and_market()
    out = compute_v3_targets(stock_panel, market)
    for col in TARGET_COLUMN_NAMES:
        assert col in out.columns


def test_raw_target_matches_v1_forward_return() -> None:
    stock_panel, market = _stock_and_market()
    out = compute_v3_targets(stock_panel, market)
    v1_forward = compute_forward_returns(stock_panel)
    for h in HORIZONS:
        assert np.allclose(
            out[f"target_raw_{h}d"].to_numpy(dtype=float),
            v1_forward[f"forward_return_{h}d"].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_topix_relative_equals_raw_minus_market_forward_return() -> None:
    stock_panel, market = _stock_and_market()
    out = compute_v3_targets(stock_panel, market)
    market_close = market.set_index("date")["close"]
    for h in HORIZONS:
        expected = []
        for d in stock_panel["date"]:
            if d not in market_close.index:
                expected.append(np.nan)
                continue
            idx = market_close.index.get_loc(d)
            if idx + h >= len(market_close):
                expected.append(np.nan)
                continue
            expected.append(market_close.iloc[idx + h] / market_close.iloc[idx] - 1)
        raw = out[f"target_raw_{h}d"].to_numpy(dtype=float)
        relative = out[f"target_topix_relative_{h}d"].to_numpy(dtype=float)
        implied_market = raw - relative
        assert np.allclose(implied_market, np.array(expected, dtype=float), equal_nan=True)


def test_vol_adjusted_equals_raw_divided_by_volatility_20d() -> None:
    stock_panel, market = _stock_and_market()
    out = compute_v3_targets(stock_panel, market)
    for h in HORIZONS:
        expected = out[f"target_raw_{h}d"] / stock_panel["volatility_20d"].replace(0, np.nan)
        assert np.allclose(
            out[f"target_vol_adjusted_{h}d"].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            equal_nan=True,
        )


def test_risk_adjusted_is_finite_where_mae_nonzero() -> None:
    stock_panel, market = _stock_and_market()
    out = compute_v3_targets(stock_panel, market)
    for h in HORIZONS:
        values = out[f"target_risk_adjusted_{h}d"].dropna()
        assert len(values) > 0
        assert np.isfinite(values.to_numpy(dtype=float)).all()
