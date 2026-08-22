"""ATR (Wilder-smoothed) and rolling return volatility.

True Range formula (Wilder):
    TR[t] = max(High[t]-Low[t], |High[t]-Close[t-1]|, |Low[t]-Close[t-1]|)

For the first row there is no Close[t-1]; pandas' max(skipna=True) then
falls back to just High[0]-Low[0], which is the standard convention (not
a lookahead concern - it only uses day 0's own High/Low).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._utils import safe_divide
from features.metadata import FeatureMeta

ATR_PERIOD = 14
VOLATILITY_WINDOWS = (5, 10, 20)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _wilder_atr(tr: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: seed with the simple mean of the first `period`
    True Range values, then recursively smooth. First valid value at row
    index `period` (0-indexed) - see ATR_PERIOD's FeatureMeta below.
    """
    n = len(tr)
    atr = np.full(n, np.nan)
    tr_values = tr.to_numpy()
    if n > period:
        atr[period] = np.nanmean(tr_values[1 : period + 1])
        for t in range(period + 1, n):
            atr[t] = (atr[t - 1] * (period - 1) + tr_values[t]) / period
    return pd.Series(atr, index=tr.index)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    tr = true_range(df)
    atr = _wilder_atr(tr, ATR_PERIOD)

    out = pd.DataFrame(index=df.index)
    out["atr"] = atr
    out["atr_pct"] = safe_divide(atr, close)

    daily_return = safe_divide(close - close.shift(1), close.shift(1))
    for window in VOLATILITY_WINDOWS:
        out[f"volatility_{window}d"] = daily_return.rolling(window, min_periods=window).std()

    return out


VOLATILITY_METADATA: list[FeatureMeta] = [
    FeatureMeta(
        name="atr",
        description=f"Wilder-smoothed Average True Range, period={ATR_PERIOD}",
        formula="Wilder-smoothed True Range (seed = mean of first N TR values)",
        lookback=ATR_PERIOD + 1,
        required_columns=("high", "low", "close"),
        warmup_period=ATR_PERIOD + 1,
        directionality="neutral",
    ),
    FeatureMeta(
        name="atr_pct",
        description="ATR as a fraction of Close",
        formula="ATR / Close",
        lookback=ATR_PERIOD + 1,
        required_columns=("high", "low", "close"),
        warmup_period=ATR_PERIOD + 1,
        directionality="neutral",
    ),
] + [
    FeatureMeta(
        name=f"volatility_{w}d",
        description=f"{w}-day rolling std. dev. of the 1-day simple return",
        formula=f"std(return_1d[t-{w - 1}..t])",
        lookback=w + 1,
        required_columns=("close",),
        warmup_period=w + 1,
        directionality="neutral",
    )
    for w in VOLATILITY_WINDOWS
]
