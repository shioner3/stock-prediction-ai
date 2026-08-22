"""SHORT Momentum Continuation: mirror of LONG Momentum Continuation -
downward momentum that is still in progress.

    return_5d[t]  < return_5d_max
    AND return_20d[t] < return_20d_max
    AND Close[t] < SMA_period[t]

See research/signal_notes/short_momentum_continuation.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ShortMomentumContinuationSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "short_momentum_continuation"


def meta(config: ShortMomentumContinuationSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.SHORT,
        description=(
            f"return_5d < {config.return_5d_max}, return_20d < {config.return_20d_max}, "
            f"Close < SMA{config.sma_period}"
        ),
        required_columns=("close", "return_5d", "return_20d", f"sma_{config.sma_period}"),
    )


def compute_signal(
    panel: pd.DataFrame, config: ShortMomentumContinuationSignalConfig
) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)

    triggered = (
        (panel["return_5d"] < config.return_5d_max)
        & (panel["return_20d"] < config.return_20d_max)
        & (panel["close"] < panel[f"sma_{config.sma_period}"])
    )
    return triggered.fillna(False)
