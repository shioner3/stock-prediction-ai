"""LONG Pullback: a dip within an established uptrend - not too shallow
(noise), not too deep (a falling knife, not a pullback).

    SMA_fast[t] > SMA_slow[t]                         (uptrend intact)
    AND Close[t] > SMA_fast[t]                         (price still above the fast MA)
    AND min_depth <= pullback_depth[t] <= max_depth    (a real but bounded dip)

pullback_depth is features/pullback.py's Phase 2 feature (magnitude of
decline from the recent high) - not recomputed here.
See research/signal_notes/long_pullback.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import LongPullbackSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "long_pullback"


def meta(config: LongPullbackSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.LONG,
        description=(
            f"SMA{config.sma_fast} > SMA{config.sma_slow} (uptrend), Close above "
            f"SMA{config.sma_fast}, and pullback_depth in "
            f"[{config.min_depth}, {config.max_depth}]"
        ),
        required_columns=(
            "close",
            f"sma_{config.sma_fast}",
            f"sma_{config.sma_slow}",
            "pullback_depth",
        ),
    )


def compute_signal(panel: pd.DataFrame, config: LongPullbackSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    sma_fast = panel[f"sma_{config.sma_fast}"]
    sma_slow = panel[f"sma_{config.sma_slow}"]
    depth = panel["pullback_depth"]

    triggered = (
        (sma_fast > sma_slow)
        & (panel["close"] > sma_fast)
        & (depth >= config.min_depth)
        & (depth <= config.max_depth)
    )
    return triggered.fillna(False)
