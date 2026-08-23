"""V3 Market/Index context features (spec section 5). TOPIX-level
indicators reuse V1's own `compute_momentum_features()`/
`compute_volatility_features()`/`compute_pullback_features()` (features/
momentum.py, features/volatility.py, features/pullback.py, all
unmodified) applied to the market benchmark's OWN OHLCV, then joined by
date onto every stock's panel - the same TOPIX Proxy (1306.T) V1's
Relative Strength feature and V2's Regime classification already use, not
a new market data source.

Market breadth / advancing / declining ratio are NEW - no existing V1/V2
primitive computes a daily cross-sectional breadth statistic. Computed
once per date across the FULL stacked Universe panel (like v3/features/
cross_sectional.py, and unlike every per-ticker feature above).
"""

from __future__ import annotations

import pandas as pd

from features.momentum import compute_momentum_features
from features.pullback import compute_pullback_features
from features.volatility import compute_volatility_features

MARKET_COLUMN_PREFIX = "topix_"


def compute_market_context_panel(market_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """market_ohlcv: the market benchmark's OHLCV (V1's existing TOPIX
    Proxy fetch). Returns date + topix_return_{1,3,5,10,20,60}d +
    topix_volatility_20d + topix_drawdown_20d, ready to left-merge by
    date onto every ticker's own panel.
    """
    momentum = compute_momentum_features(market_ohlcv)
    volatility = compute_volatility_features(market_ohlcv)
    drawdown = compute_pullback_features(market_ohlcv, recent_high_window=20)

    out = pd.DataFrame({"date": market_ohlcv["date"].to_numpy()})
    for col in momentum.columns:
        out[f"{MARKET_COLUMN_PREFIX}{col}"] = momentum[col].to_numpy()
    out[f"{MARKET_COLUMN_PREFIX}volatility_20d"] = volatility["volatility_20d"].to_numpy()
    out[f"{MARKET_COLUMN_PREFIX}drawdown_20d"] = drawdown["distance_from_recent_high"].to_numpy()
    return out


def compute_market_breadth(
    universe_panel: pd.DataFrame, date_col: str = "date", return_col: str = "return_1d"
) -> pd.DataFrame:
    """Cross-sectional daily breadth over the FULL Universe panel - must be
    computed AFTER every ticker's panel is stacked. advancing_ratio/
    declining_ratio are the raw fractions; market_breadth is their
    difference (a net breadth measure, distinct from either ratio alone).
    """
    valid = universe_panel.dropna(subset=[return_col])
    grouped = valid.groupby(date_col)[return_col]
    advancing_ratio = grouped.apply(lambda s: float((s > 0).mean()))
    declining_ratio = grouped.apply(lambda s: float((s < 0).mean()))

    out = pd.DataFrame(
        {
            date_col: advancing_ratio.index,
            "advancing_ratio": advancing_ratio.to_numpy(),
            "declining_ratio": declining_ratio.to_numpy(),
        }
    )
    out["market_breadth"] = out["advancing_ratio"] - out["declining_ratio"]
    return out
