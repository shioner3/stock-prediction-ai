"""SHORT Volume Breakdown: mirror of LONG Volume Breakout - a single-day
price drop + volume spike, independent of any multi-day low comparison
(that's SHORT Breakdown's hypothesis, not this one's).

    return_1d[t] < return_1d_max
    AND volume_ratio_20d[t] > volume_ratio_min

See research/signal_notes/short_volume_breakdown.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ShortVolumeBreakdownSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "short_volume_breakdown"


def meta(config: ShortVolumeBreakdownSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.SHORT,
        description=(
            f"return_1d < {config.return_1d_max} and "
            f"volume_ratio_20d > {config.volume_ratio_min}"
        ),
        required_columns=("return_1d", "volume_ratio_20d"),
    )


def compute_signal(panel: pd.DataFrame, config: ShortVolumeBreakdownSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)

    triggered = (panel["return_1d"] < config.return_1d_max) & (
        panel["volume_ratio_20d"] > config.volume_ratio_min
    )
    return triggered.fillna(False)
