"""V2 Feature Engineering adapter (spec section 6).

Does NOT reimplement Momentum/Trend/Volatility/Volume/Relative Strength/
Pullback from scratch - it calls features.pipeline.compute_feature_panel()
(unmodified V1 code) and adds a small number of V2-specific DERIVED
ratios (MA-vs-MA crosses) that V1's panel doesn't already expose,
computed by calling V1's own features._utils.sma() utility on the same
Close series - never a new independent rolling-mean formula.

Every added column here is a pure function of columns compute_feature_panel()
already produced (or of the same backward-only sma()/safe_divide() V1
already uses), so it carries the same no-lookahead guarantee by
construction - see tests/test_v2_leakage.py's future-shock test.
"""

from __future__ import annotations

import pandas as pd

from features._utils import safe_divide, sma
from features.pipeline import compute_feature_panel

# V1's compute_feature_panel() has no 60-day SMA distance/cross column
# (features/trend.py::MA_DISTANCE_WINDOWS stops at 50/75, no 60) - spec
# section 6.A explicitly asks for price_vs_ma60/ma20_vs_ma60, so a single
# additional 60-day SMA is computed here via V1's own sma() utility
# (reused, not reimplemented) rather than substituting the nearest
# existing window.
MA60_WINDOW = 60

# V1's pullback.py only exposes a 20-day "distance from recent high"
# (its recent_high_window default) - spec section 6.B also asks for a
# 60-day version. Computed the same way pullback.py computes its own
# 20-day column (today's Close included in the rolling max, backward-only
# rolling().max(), safe_divide()) - not a new formula, just the same
# formula at a different window.
RECENT_HIGH_60D_WINDOW = 60


def add_v2_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: an already-computed V1 Feature panel (either freshly built
    by compute_feature_panel() or loaded from V1's existing Parquet cache
    via storage.parquet_store.load_feature_panel() - both are the exact
    same shape/columns, since V1's cache was built by this same
    unmodified function). Adds ONLY the 4 V2-specific derived columns;
    never recomputes anything compute_feature_panel() already produced.
    Split out from build_v2_feature_panel() so v2/pipeline.py can reuse
    V1's ALREADY-CACHED Full Universe feature panels (fast path) instead
    of recomputing Trend/Momentum/.../Relative Strength from raw OHLCV
    for all ~2,880 tickers.
    """
    close = panel["close"]
    sma_60 = sma(close, MA60_WINDOW)
    recent_high_60d = close.rolling(
        RECENT_HIGH_60D_WINDOW, min_periods=RECENT_HIGH_60D_WINDOW
    ).max()
    return panel.assign(
        price_vs_ma60=safe_divide(close, sma_60) - 1,
        ma5_vs_ma20=safe_divide(panel["sma_5"], panel["sma_20"]) - 1,
        ma20_vs_ma60=safe_divide(panel["sma_20"], sma_60) - 1,
        distance_from_60d_high=safe_divide(close - recent_high_60d, recent_high_60d),
    )


def build_v2_feature_panel(
    ohlcv: pd.DataFrame, market_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """ohlcv: one ticker's OHLCV (REQUIRED_COLUMNS per features/pipeline.py).
    market_df: the market benchmark's OHLCV, passed straight through to
    compute_feature_panel() for Relative Strength - same contract as V1.
    Computes V1's Feature panel FRESH (compute_feature_panel(), unmodified)
    - the full/canonical path, used directly by tests (small synthetic
    inputs) and available for any ticker/date not covered by V1's
    existing cache. See add_v2_derived_features() for the cache-reuse
    fast path v2/pipeline.py uses for the real Full Universe dataset.
    """
    panel = compute_feature_panel(ohlcv, market_df=market_df)
    return add_v2_derived_features(panel)
