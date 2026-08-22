"""LONG Breakout: price breaks above its N-day high on above-average volume.

    Close[t] > high_Nd[t]              (high_Nd already excludes today - Phase 2)
    AND volume_ratio_20d[t] > volume_multiple

Reuses features/breakout.py's high_Nd and features/volume.py's
volume_ratio_20d directly - no rolling calculation is reimplemented here.
See research/signal_notes/long_breakout.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import LongBreakoutSignalConfig
from features.breakout import HIGH_LOW_WINDOWS
from signals.base import Direction, SignalMeta, require_columns

NAME = "long_breakout"


def _high_column(lookback: int) -> str:
    if lookback not in HIGH_LOW_WINDOWS:
        raise ValueError(
            f"breakout lookback={lookback} has no matching feature column - "
            f"must be one of {HIGH_LOW_WINDOWS} (features/breakout.py)"
        )
    return f"high_{lookback}d"


def meta(config: LongBreakoutSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.LONG,
        description=(
            f"Close breaks above the prior {config.lookback}-day high on volume "
            f"> {config.volume_multiple}x its 20-day average"
        ),
        required_columns=("close", _high_column(config.lookback), "volume_ratio_20d"),
    )


def compute_signal(panel: pd.DataFrame, config: LongBreakoutSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    high_col = _high_column(config.lookback)

    triggered = (panel["close"] > panel[high_col]) & (
        panel["volume_ratio_20d"] > config.volume_multiple
    )
    return triggered.fillna(False)
