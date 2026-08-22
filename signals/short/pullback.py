"""SHORT Pullback ("戻り売り"): a bounce within an established downtrend -
not too shallow, not too strong a reversal. Mirror of LONG Pullback,
using bounce_depth (magnitude of rise from the recent low) in place of
pullback_depth.

    SMA_fast[t] < SMA_slow[t]                      (downtrend intact)
    AND Close[t] < SMA_fast[t]                      (price still below the fast MA)
    AND min_depth <= bounce_depth[t] <= max_depth   (a real but bounded bounce)

bounce_depth was added to features/pullback.py in Phase 4 - see that
module's docstring.
See research/signal_notes/short_pullback.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ShortPullbackSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "short_pullback"


def meta(config: ShortPullbackSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.SHORT,
        description=(
            f"SMA{config.sma_fast} < SMA{config.sma_slow} (downtrend), Close below "
            f"SMA{config.sma_fast}, and bounce_depth in "
            f"[{config.min_depth}, {config.max_depth}]"
        ),
        required_columns=(
            "close",
            f"sma_{config.sma_fast}",
            f"sma_{config.sma_slow}",
            "bounce_depth",
        ),
    )


def compute_signal(panel: pd.DataFrame, config: ShortPullbackSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    sma_fast = panel[f"sma_{config.sma_fast}"]
    sma_slow = panel[f"sma_{config.sma_slow}"]
    depth = panel["bounce_depth"]

    triggered = (
        (sma_fast < sma_slow)
        & (panel["close"] < sma_fast)
        & (depth >= config.min_depth)
        & (depth <= config.max_depth)
    )
    return triggered.fillna(False)
