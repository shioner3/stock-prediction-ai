"""Combines every feature category into one panel for a single ticker's
OHLCV history:

    OHLCV -> Trend (+MA Distance) -> Momentum -> Volatility -> Volume
          -> Indicators (RSI+MACD) -> Breakout -> Pullback
          -> Relative Strength (only if a market benchmark is supplied)

Each compute_*_features() function is pure and takes only the OHLCV
DataFrame (plus, for Pullback, one config value, and for Relative
Strength, an optional second OHLCV DataFrame for the market benchmark) -
this module's only job is to call them in order and join the results. It
must never import providers/ (no vendor calls belong at this layer - the
market benchmark arrives here as a plain DataFrame that the caller
already fetched via MarketIndexProvider) or a future targets.forward_returns
module (see tests/test_no_lookahead.py's dependency test) - a feature
that needs forward-looking data does not belong in this package.
"""

from __future__ import annotations

import pandas as pd

from features.breakout import BREAKOUT_METADATA, compute_breakout_features
from features.indicators import INDICATOR_METADATA, compute_macd_features, compute_rsi_features
from features.metadata import FeatureMeta
from features.momentum import MOMENTUM_METADATA, compute_momentum_features
from features.pullback import (
    DEFAULT_RECENT_HIGH_WINDOW,
    compute_pullback_features,
    pullback_metadata,
)
from features.relative_strength import (
    DEFAULT_MARKET_LABEL,
    compute_relative_strength_features,
    relative_strength_metadata,
)
from features.trend import (
    MA_DISTANCE_METADATA,
    TREND_METADATA,
    compute_ma_distance_features,
    compute_trend_features,
)
from features.volatility import VOLATILITY_METADATA, compute_volatility_features
from features.volume import VOLUME_METADATA, compute_volume_features

REQUIRED_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def feature_metadata(
    recent_high_window: int = DEFAULT_RECENT_HIGH_WINDOW,
    market_label: str = DEFAULT_MARKET_LABEL,
) -> list[FeatureMeta]:
    """All feature metadata, for a given pullback recent_high_window and
    market-benchmark label (purely descriptive - does not affect whether
    RS is actually computed, only how its FeatureMeta.description reads).
    """
    return (
        TREND_METADATA
        + MA_DISTANCE_METADATA
        + MOMENTUM_METADATA
        + VOLATILITY_METADATA
        + VOLUME_METADATA
        + INDICATOR_METADATA
        + BREAKOUT_METADATA
        + pullback_metadata(recent_high_window)
        + relative_strength_metadata(market_label)
    )


def compute_feature_panel(
    df: pd.DataFrame,
    recent_high_window: int = DEFAULT_RECENT_HIGH_WINDOW,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """df must be a single ticker's OHLCV history, sorted ascending by
    date, with columns REQUIRED_COLUMNS (extra columns are passed through
    untouched). Returns df with every feature column appended - no rows
    are dropped for warmup; early rows simply carry NaN in whichever
    features need more history than is available yet.

    market_df is the market benchmark's OHLCV (same shape as df) used for
    Relative Strength (rs_5d/20d/60d). Pass None when the benchmark could
    not be fetched (see providers/market_index.py's topix_available) -
    every other feature category is still computed normally; only rs_*
    becomes all-NaN.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input OHLCV is missing required columns: {missing}")

    out = df.reset_index(drop=True).copy()
    feature_blocks = [
        compute_trend_features(out),
        compute_ma_distance_features(out),
        compute_momentum_features(out),
        compute_volatility_features(out),
        compute_volume_features(out),
        compute_rsi_features(out),
        compute_macd_features(out),
        compute_breakout_features(out),
        compute_pullback_features(out, recent_high_window=recent_high_window),
        compute_relative_strength_features(out, market_df),
    ]
    for block in feature_blocks:
        out = out.join(block)
    return out
