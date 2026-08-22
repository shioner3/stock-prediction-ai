"""LONG Volume Breakout: a single-day price+volume spike - a DIFFERENT
hypothesis from LONG Breakout (which requires clearing a multi-day price
high). This Signal has no high_Nd comparison at all; it can trigger on a
day that is nowhere near a 20-day high, as long as the day's own move and
volume are both unusually large.

    return_1d[t] > return_1d_min
    AND volume_ratio_20d[t] > volume_ratio_min

See research/signal_notes/long_volume_breakout.md for the hypothesis and
how it's meant to differ from LONG Breakout.
"""

from __future__ import annotations

import pandas as pd

from config.loader import LongVolumeBreakoutSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "long_volume_breakout"


def meta(config: LongVolumeBreakoutSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.LONG,
        description=(
            f"return_1d > {config.return_1d_min} and "
            f"volume_ratio_20d > {config.volume_ratio_min}"
        ),
        required_columns=("return_1d", "volume_ratio_20d"),
    )


def compute_signal(panel: pd.DataFrame, config: LongVolumeBreakoutSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)

    triggered = (panel["return_1d"] > config.return_1d_min) & (
        panel["volume_ratio_20d"] > config.volume_ratio_min
    )
    return triggered.fillna(False)
