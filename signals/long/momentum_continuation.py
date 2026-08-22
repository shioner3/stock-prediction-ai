"""LONG Momentum Continuation: short-term upward momentum that is still
in progress, not a single-day spike - the hypothesis this Signal tests is
"the trend continues," not "the stock just jumped."

    return_5d[t]  > return_5d_min
    AND return_20d[t] > return_20d_min
    AND Close[t] > SMA_period[t]

See research/signal_notes/long_momentum_continuation.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import LongMomentumContinuationSignalConfig
from signals.base import Direction, SignalMeta, require_columns

NAME = "long_momentum_continuation"


def meta(config: LongMomentumContinuationSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.LONG,
        description=(
            f"return_5d > {config.return_5d_min}, return_20d > {config.return_20d_min}, "
            f"Close > SMA{config.sma_period}"
        ),
        required_columns=("close", "return_5d", "return_20d", f"sma_{config.sma_period}"),
    )


def compute_signal(panel: pd.DataFrame, config: LongMomentumContinuationSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)

    triggered = (
        (panel["return_5d"] > config.return_5d_min)
        & (panel["return_20d"] > config.return_20d_min)
        & (panel["close"] > panel[f"sma_{config.sma_period}"])
    )
    return triggered.fillna(False)
