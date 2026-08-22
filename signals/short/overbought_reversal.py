"""SHORT Overbought Reversal: mirror of LONG Oversold Rebound - RSI is
high and today is a down day. A reversal CANDIDATE, not a claim that
"high RSI means sell."

    RSI_period[t] > rsi_min
    AND Close[t] < Close[t-1]

See research/signal_notes/short_overbought_reversal.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ShortOverboughtReversalSignalConfig
from features.indicators import RSI_PERIODS
from signals.base import Direction, SignalMeta, require_columns

NAME = "short_overbought_reversal"


def _rsi_column(period: int) -> str:
    if period not in RSI_PERIODS:
        raise ValueError(
            f"overbought_reversal rsi_period={period} has no matching feature column - "
            f"must be one of {RSI_PERIODS} (features/indicators.py)"
        )
    return f"rsi_{period}"


def meta(config: ShortOverboughtReversalSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.SHORT,
        description=f"RSI{config.rsi_period} > {config.rsi_min} and Close < previous Close",
        required_columns=("close", _rsi_column(config.rsi_period)),
    )


def compute_signal(panel: pd.DataFrame, config: ShortOverboughtReversalSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    rsi_col = _rsi_column(config.rsi_period)

    triggered = (panel[rsi_col] > config.rsi_min) & (panel["close"] < panel["close"].shift(1))
    return triggered.fillna(False)
