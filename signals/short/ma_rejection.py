"""SHORT MA Rejection: price closes back below the fast moving average
after closing at/above it the previous session, within an established
downtrend. Mirror of LONG MA Rebound.

    SMA_fast[t] < SMA_slow[t]              (downtrend intact)
    AND Close[t-1] >= SMA_fast[t-1]        (was at/above the MA yesterday)
    AND Close[t]   <  SMA_fast[t]          (closed back below it today - rejected)

See research/signal_notes/short_ma_rejection.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ShortMaRejectionSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "short_ma_rejection"


def meta(config: ShortMaRejectionSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.SHORT,
        description=(
            f"SMA{config.sma_fast} < SMA{config.sma_slow}, Close was >= SMA{config.sma_fast} "
            f"yesterday and < SMA{config.sma_fast} today"
        ),
        required_columns=("close", f"sma_{config.sma_fast}", f"sma_{config.sma_slow}"),
    )


def compute_signal(panel: pd.DataFrame, config: ShortMaRejectionSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    sma_fast = panel[f"sma_{config.sma_fast}"]
    sma_slow = panel[f"sma_{config.sma_slow}"]
    close = panel["close"]

    was_at_or_above = close.shift(1) >= sma_fast.shift(1)
    now_below = close < sma_fast
    downtrend = sma_fast < sma_slow

    triggered = downtrend & was_at_or_above & now_below
    return triggered.fillna(False)
