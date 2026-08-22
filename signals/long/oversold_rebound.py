"""LONG Oversold Rebound: RSI is low and today is an up day - a rebound
CANDIDATE, not a claim that "low RSI means buy" (that judgment is exactly
what Phase 6's backtest is for; this module only encodes the testable
condition).

    RSI_period[t] < rsi_max
    AND Close[t] > Close[t-1]

See research/signal_notes/long_oversold_rebound.md for the hypothesis.
"""

from __future__ import annotations

import pandas as pd

from config.loader import LongOversoldReboundSignalConfig
from features.indicators import RSI_PERIODS
from signals.base import Direction, SignalMeta, require_columns

NAME = "long_oversold_rebound"


def _rsi_column(period: int) -> str:
    if period not in RSI_PERIODS:
        raise ValueError(
            f"oversold_rebound rsi_period={period} has no matching feature column - "
            f"must be one of {RSI_PERIODS} (features/indicators.py)"
        )
    return f"rsi_{period}"


def meta(config: LongOversoldReboundSignalConfig) -> SignalMeta:
    return SignalMeta(
        name=NAME,
        direction=Direction.LONG,
        description=f"RSI{config.rsi_period} < {config.rsi_max} and Close > previous Close",
        required_columns=("close", _rsi_column(config.rsi_period)),
    )


def compute_signal(panel: pd.DataFrame, config: LongOversoldReboundSignalConfig) -> pd.Series:
    m = meta(config)
    require_columns(panel, m)
    rsi_col = _rsi_column(config.rsi_period)

    triggered = (panel[rsi_col] < config.rsi_max) & (panel["close"] > panel["close"].shift(1))
    return triggered.fillna(False)
