"""Trend features: moving averages, their slopes, and Close's distance
from each moving average (MA Distance).

MA Distance lives in this module rather than a separate file because it
reuses the exact same SMA values as the Trend category - splitting it out
would mean either recomputing SMAs twice or threading intermediate
columns between modules, neither of which is worth it for one extra file.
"""

from __future__ import annotations

import pandas as pd

from features._utils import ema, safe_divide, slope, sma
from features.metadata import FeatureMeta

SMA_WINDOWS = (5, 10, 20, 25, 50, 75, 100, 200)
EMA_SPANS = (5, 20, 50)

# Slope = fractional change of the MA over this many trading days (see
# features/_utils.py::slope). Fixed at 5 for every MA slope so results
# are comparable across different MA windows - a MA's own period is not
# used as its slope window (e.g. SMA200 slope is NOT measured over 200
# days; it is the 5-day fractional change of the SMA200 value).
SLOPE_WINDOW = 5
SLOPE_SMA_WINDOWS = (5, 20, 50, 200)

MA_DISTANCE_WINDOWS = (5, 20, 25, 50, 75, 200)


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    smas: dict[int, pd.Series] = {}
    for window in SMA_WINDOWS:
        s = sma(close, window)
        smas[window] = s
        out[f"sma_{window}"] = s

    for span in EMA_SPANS:
        out[f"ema_{span}"] = ema(close, span)

    for window in SLOPE_SMA_WINDOWS:
        out[f"sma_{window}_slope"] = slope(smas[window], SLOPE_WINDOW)

    return out


def compute_ma_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Close's distance from each SMA, as a fraction: Close/SMA - 1."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for window in MA_DISTANCE_WINDOWS:
        out[f"close_to_sma_{window}"] = safe_divide(close, sma(close, window)) - 1
    return out


TREND_METADATA: list[FeatureMeta] = (
    [
        FeatureMeta(
            name=f"sma_{w}",
            description=f"{w}-day simple moving average of Close",
            formula=f"mean(Close[t-{w - 1}..t])",
            lookback=w,
            required_columns=("close",),
            warmup_period=w,
            directionality="neutral",
        )
        for w in SMA_WINDOWS
    ]
    + [
        FeatureMeta(
            name=f"ema_{s}",
            description=f"{s}-day exponential moving average of Close",
            formula=f"EWM(Close, span={s}, adjust=False)",
            lookback=s,
            required_columns=("close",),
            warmup_period=s,
            directionality="neutral",
        )
        for s in EMA_SPANS
    ]
    + [
        FeatureMeta(
            name=f"sma_{w}_slope",
            description=f"{SLOPE_WINDOW}-day fractional slope of SMA{w}",
            formula=(
                f"(SMA{w}[t]-SMA{w}[t-{SLOPE_WINDOW}])"
                f"/SMA{w}[t-{SLOPE_WINDOW}]/{SLOPE_WINDOW}"
            ),
            lookback=w + SLOPE_WINDOW,
            required_columns=("close",),
            warmup_period=w + SLOPE_WINDOW,
            directionality="bullish_high",
        )
        for w in SLOPE_SMA_WINDOWS
    ]
)

MA_DISTANCE_METADATA: list[FeatureMeta] = [
    FeatureMeta(
        name=f"close_to_sma_{w}",
        description=f"Close's distance from SMA{w}, as a fraction",
        formula=f"Close/SMA{w} - 1",
        lookback=w,
        required_columns=("close",),
        warmup_period=w,
        directionality="bullish_high",
    )
    for w in MA_DISTANCE_WINDOWS
]
