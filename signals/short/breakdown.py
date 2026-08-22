"""SHORT Breakdown: mirror of LONG Breakout. price breaks below its N-day
low on above-average volume.

    Close[t] < low_Nd[t]               (low_Nd already excludes today - Phase 2/4)
    AND volume_ratio_20d[t] > volume_multiple

low_Nd was added to features/breakout.py in Phase 4 specifically to keep
this Signal reading from the Feature layer instead of recomputing a
rolling min inline - see that module's docstring.
See research/signal_notes/short_breakdown.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ShortBreakdownSignalConfig
from features.breakout import HIGH_LOW_WINDOWS
from signals.base import Direction, SignalMeta, require_columns

NAME = "short_breakdown"


def _low_column(lookback: int) -> str:
    if lookback not in HIGH_LOW_WINDOWS:
        raise ValueError(
            f"breakdown lookback={lookback} has no matching feature column - "
            f"must be one of {HIGH_LOW_WINDOWS} (features/breakout.py)"
        )
    return f"low_{lookback}d"


def meta(config: ShortBreakdownSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.SHORT,
        description=(
            f"Close breaks below the prior {config.lookback}-day low on volume "
            f"> {config.volume_multiple}x its 20-day average"
        ),
        required_columns=("close", _low_column(config.lookback), "volume_ratio_20d"),
    )


def compute_signal(panel: pd.DataFrame, config: ShortBreakdownSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    low_col = _low_column(config.lookback)

    triggered = (panel["close"] < panel[low_col]) & (
        panel["volume_ratio_20d"] > config.volume_multiple
    )
    return triggered.fillna(False)
