"""LONG MA Rebound: price closes back above the fast moving average after
closing at/below it the previous session, within an established uptrend.

    SMA_fast[t] > SMA_slow[t]              (uptrend intact)
    AND Close[t-1] <= SMA_fast[t-1]        (was at/below the MA yesterday)
    AND Close[t]   >  SMA_fast[t]          (closed back above it today)

"Rebound" is defined strictly as this yesterday-below / today-above
transition - not merely "price is above its MA" (that would match every
day of an uptrend, not just the rebound day).
See research/signal_notes/long_ma_rebound.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import LongMaReboundSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "long_ma_rebound"


def meta(config: LongMaReboundSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.LONG,
        description=(
            f"SMA{config.sma_fast} > SMA{config.sma_slow}, Close was <= SMA{config.sma_fast} "
            f"yesterday and > SMA{config.sma_fast} today"
        ),
        required_columns=("close", f"sma_{config.sma_fast}", f"sma_{config.sma_slow}"),
    )


def compute_signal(panel: pd.DataFrame, config: LongMaReboundSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    sma_fast = panel[f"sma_{config.sma_fast}"]
    sma_slow = panel[f"sma_{config.sma_slow}"]
    close = panel["close"]

    was_at_or_below = close.shift(1) <= sma_fast.shift(1)
    now_above = close > sma_fast
    uptrend = sma_fast > sma_slow

    triggered = uptrend & was_at_or_below & now_above
    return triggered.fillna(False)
