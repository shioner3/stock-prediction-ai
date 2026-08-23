"""V3 Target computation (spec section 3). Reuses `targets.forward_returns.
compute_forward_returns()`/`compute_mfe_mae()` (V1 code, unmodified) as
the base for Raw Return and the ex-post risk denominator; TOPIX-relative
and Volatility-adjusted are new, thin derivations built only from already-
computed columns (V1's forward returns + V3's own `volatility_20d`/TOPIX
panel), following `features/relative_strength.py`'s own DATE-based (never
row-position-based) alignment convention for anything that mixes a
stock's own date sequence with the market benchmark's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from targets.forward_returns import compute_forward_returns, compute_mfe_mae
from v3.targets.registry import (
    HORIZONS,
    VARIANT_RAW,
    VARIANT_RISK_ADJUSTED,
    VARIANT_TOPIX_RELATIVE,
    VARIANT_VOL_ADJUSTED,
    target_column_name,
)


def _forward_return_by_date(market_ohlcv: pd.DataFrame, horizon: int) -> pd.Series:
    """TOPIX Proxy's own forward return at each of ITS OWN dates, as a
    date-indexed Series - computed on the market's own chronological
    sequence (never the stock's row order), matching features/
    relative_strength.py's alignment discipline exactly.
    """
    market_dates = pd.to_datetime(market_ohlcv["date"])
    market_close = pd.Series(market_ohlcv["close"].to_numpy(), index=market_dates).sort_index()
    market_close = market_close[~market_close.index.duplicated(keep="last")]
    forward_return = market_close.shift(-horizon) / market_close - 1
    return forward_return


def compute_v3_targets(stock_panel: pd.DataFrame, market_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """stock_panel: one ticker's panel, needs close/high/low/date and an
    already-computed volatility_20d column (v3/features/price_features.py's
    output, or V1's own compute_feature_panel() output). market_ohlcv:
    the TOPIX Proxy's OHLCV. Returns a NEW DataFrame with all 16 target
    columns (v3/targets/registry.py::TARGET_REGISTRY) appended.
    """
    out = stock_panel.copy()
    forward = compute_forward_returns(stock_panel)
    mfe_mae = compute_mfe_mae(stock_panel, direction="LONG")
    stock_dates = pd.to_datetime(stock_panel["date"])

    for horizon in HORIZONS:
        raw = forward[f"forward_return_{horizon}d"]
        out[target_column_name(VARIANT_RAW, horizon)] = raw

        market_forward = _forward_return_by_date(market_ohlcv, horizon)
        aligned_market_forward = market_forward.reindex(stock_dates).to_numpy()
        out[target_column_name(VARIANT_TOPIX_RELATIVE, horizon)] = (
            raw.to_numpy() - aligned_market_forward
        )

        # np.nan (not pd.NA) - replacing into a float64 Series with pd.NA
        # silently upcasts it to `object` dtype (a mix of Python float and
        # pd.NA), which LightGBM's dtype check then rejects outright; NaN
        # stays within float64 and is what every other NaN in this
        # pipeline (warmup, end-of-history) already uses.
        vol_denominator = stock_panel["volatility_20d"].replace(0, np.nan)
        out[target_column_name(VARIANT_VOL_ADJUSTED, horizon)] = raw / vol_denominator

        mae = mfe_mae[f"mae_{horizon}d"]
        risk_denominator = mae.abs().replace(0, np.nan)
        out[target_column_name(VARIANT_RISK_ADJUSTED, horizon)] = raw / risk_denominator

    return out
