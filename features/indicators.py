"""RSI (Wilder) and MACD.

RSI uses the classic Wilder smoothing (not the ewm-approximation many
libraries substitute for it): the first average gain/loss is a simple
mean of the first `period` gains/losses, and every value after that is
the Wilder recursive smooth - see _wilder_rsi().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._utils import ema, mask_warmup
from features.metadata import FeatureMeta

RSI_PERIODS = (7, 14)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).to_numpy()
    loss = (-delta.clip(upper=0.0)).to_numpy()
    n = len(close)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > period:
        avg_gain[period] = np.nanmean(gain[1 : period + 1])
        avg_loss[period] = np.nanmean(loss[1 : period + 1])
        for t in range(period + 1, n):
            avg_gain[t] = (avg_gain[t - 1] * (period - 1) + gain[t]) / period
            avg_loss[t] = (avg_loss[t - 1] * (period - 1) + loss[t]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    # avg_loss == 0: no down days in the window.
    #   avg_gain == 0 too -> price hasn't moved at all -> neutral, 50.
    #   avg_gain > 0     -> pure up-trend -> 100.
    # avg_gain == 0 and avg_loss > 0 -> pure down-trend -> 0.
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    rsi = np.where((avg_gain == 0) & (avg_loss > 0), 0.0, rsi)
    return pd.Series(rsi, index=close.index)


def compute_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for period in RSI_PERIODS:
        out[f"rsi_{period}"] = _wilder_rsi(close, period)
    return out


def compute_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema_fast = ema(close, MACD_FAST)
    ema_slow = ema(close, MACD_SLOW)
    # ema_slow already carries NaN through row MACD_SLOW-2 (see ema()'s
    # own warmup mask), so macd inherits that warmup by simple arithmetic
    # propagation - no extra masking needed here.
    macd = ema_fast - ema_slow

    signal_raw = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    # macd itself only becomes valid at row MACD_SLOW-1; the signal line
    # then needs MACD_SIGNAL-1 more valid macd observations to be
    # considered warmed up under this project's convention (ewm() would
    # otherwise happily return values immediately once macd exists).
    signal_warmup = MACD_SLOW + MACD_SIGNAL - 1
    macd_signal = mask_warmup(signal_raw, signal_warmup)
    macd_hist = macd - macd_signal

    return pd.DataFrame(
        {"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist}, index=df.index
    )


INDICATOR_METADATA: list[FeatureMeta] = (
    [
        FeatureMeta(
            name=f"rsi_{p}",
            description=f"{p}-period Wilder RSI",
            formula="100 - 100/(1+RS), RS = Wilder-smoothed avg gain / avg loss",
            lookback=p,
            required_columns=("close",),
            warmup_period=p + 1,
            directionality="bullish_high",
        )
        for p in RSI_PERIODS
    ]
    + [
        FeatureMeta(
            name="macd",
            description=f"MACD line: EMA{MACD_FAST} - EMA{MACD_SLOW}",
            formula=f"EMA(Close,{MACD_FAST}) - EMA(Close,{MACD_SLOW})",
            lookback=MACD_SLOW,
            required_columns=("close",),
            warmup_period=MACD_SLOW,
            directionality="bullish_high",
        ),
        FeatureMeta(
            name="macd_signal",
            description=f"{MACD_SIGNAL}-period EMA of the MACD line",
            formula=f"EWM(MACD, span={MACD_SIGNAL}, adjust=False)",
            lookback=MACD_SLOW + MACD_SIGNAL - 1,
            required_columns=("close",),
            warmup_period=MACD_SLOW + MACD_SIGNAL - 1,
            directionality="bullish_high",
        ),
        FeatureMeta(
            name="macd_hist",
            description="MACD line minus its signal line",
            formula="MACD - macd_signal",
            lookback=MACD_SLOW + MACD_SIGNAL - 1,
            required_columns=("close",),
            warmup_period=MACD_SLOW + MACD_SIGNAL - 1,
            directionality="bullish_high",
        ),
    ]
)
