"""V3-specific derived price/volume features NOT already produced by V1's
`features.pipeline.compute_feature_panel()` (spec section 5). Every column
here is built PURELY from V1's own reusable, already-tested utility
functions (`features._utils.sma/safe_divide/slope`, `features.momentum.
compute_return`, `features.indicators._wilder_rsi`, `features.pullback.
compute_pullback_features`) applied at windows/periods V1's fixed constant
tuples don't already cover - never a new, independently-derived formula.
Mirrors `v2/features_adapter.py`'s own precedent exactly (a SMALL set of
derived columns via V1's utilities, not a reimplementation).

Every function here only reads columns already present in `panel` (V1's
already-point-in-time-correct output) and only ever calls backward-looking
operations (rolling(), ewm(), shift(+n)) - never shift(-n) - so no new
lookahead surface is introduced (see v3/leakage/availability_check.py's
mechanical verification of this).
"""

from __future__ import annotations

import pandas as pd

from features._utils import safe_divide, slope, sma
from features.indicators import _wilder_rsi
from features.momentum import compute_return
from features.pullback import compute_pullback_features

NEW_RSI_PERIODS = (5, 20)  # V1's RSI_PERIODS = (7, 14) has neither
NEW_SMA_WINDOWS = (60, 120)  # V1's SMA_WINDOWS = (5,10,20,25,50,75,100,200) has neither
# V1's MA_DISTANCE_WINDOWS = (5,20,25,50,75,200) is missing 10/60/120; 10
# reuses V1's own already-computed sma_10, 60/120 use NEW_SMA_WINDOWS above.
EXTRA_CLOSE_TO_SMA_WINDOWS = (10, 60, 120)
# 20d reuses V1's own distance_from_recent_high/pullback_depth (computed
# with the default recent_high_window=20) directly - only 60d/120d are new.
DRAWDOWN_HIGH_WINDOWS = (60, 120)
TURNOVER_RATIO_WINDOW = 20
VOLATILITY_CHANGE_WINDOW = 5


def add_v3_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: a V1 Feature panel (features.pipeline.compute_feature_panel()'s
    output - already has close/volume/sma_5.../volatility_20d/distance_from_recent_high/
    pullback_depth/etc.). Returns a NEW DataFrame with V3's extra columns
    appended; never mutates panel.
    """
    close = panel["close"]
    volume = panel["volume"]
    out = panel.copy()

    out["return_120d"] = compute_return(close, 120)

    new_smas: dict[int, pd.Series] = {}
    for window in NEW_SMA_WINDOWS:
        new_smas[window] = sma(close, window)
        out[f"sma_{window}"] = new_smas[window]

    for window in EXTRA_CLOSE_TO_SMA_WINDOWS:
        existing_col = f"sma_{window}"
        ma = panel[existing_col] if existing_col in panel.columns else new_smas[window]
        out[f"close_to_sma_{window}"] = safe_divide(close, ma) - 1

    out["ma5_to_ma20"] = safe_divide(panel["sma_5"], panel["sma_20"]) - 1
    out["ma20_to_ma60"] = safe_divide(panel["sma_20"], new_smas[60]) - 1
    out["ma60_to_ma120"] = safe_divide(new_smas[60], new_smas[120]) - 1

    for period in NEW_RSI_PERIODS:
        out[f"rsi_{period}"] = _wilder_rsi(close, period)

    turnover = close * volume
    out["turnover"] = turnover
    out["turnover_ratio"] = safe_divide(turnover, sma(turnover, TURNOVER_RATIO_WINDOW))

    out["distance_from_20d_high"] = panel["distance_from_recent_high"]
    out["drawdown_from_20d_high"] = panel["pullback_depth"]
    for window in DRAWDOWN_HIGH_WINDOWS:
        pullback_at_window = compute_pullback_features(panel, recent_high_window=window)
        out[f"distance_from_{window}d_high"] = pullback_at_window["distance_from_recent_high"]
        out[f"drawdown_from_{window}d_high"] = pullback_at_window["pullback_depth"]

    out["volatility_change"] = slope(panel["volatility_20d"], VOLATILITY_CHANGE_WINDOW)

    return out
