"""Volume-based features: ratio to the recent average, z-score, and trend."""

from __future__ import annotations

import pandas as pd

from features._utils import safe_divide, slope, sma
from features.metadata import FeatureMeta

VOLUME_RATIO_WINDOWS = (5, 20)
VOLUME_ZSCORE_WINDOW = 20
VOLUME_TREND_MA_WINDOW = 5
VOLUME_TREND_SLOPE_WINDOW = 5


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    volume = df["volume"]
    out = pd.DataFrame(index=df.index)

    for window in VOLUME_RATIO_WINDOWS:
        out[f"volume_ratio_{window}d"] = safe_divide(volume, sma(volume, window))

    vol_mean = sma(volume, VOLUME_ZSCORE_WINDOW)
    vol_std = volume.rolling(VOLUME_ZSCORE_WINDOW, min_periods=VOLUME_ZSCORE_WINDOW).std()
    # safe_divide -> NaN (not inf) when vol_std is 0, e.g. a constant-volume run.
    out["volume_zscore"] = safe_divide(volume - vol_mean, vol_std)

    vol_sma_short = sma(volume, VOLUME_TREND_MA_WINDOW)
    out["volume_trend"] = slope(vol_sma_short, VOLUME_TREND_SLOPE_WINDOW)

    return out


VOLUME_METADATA: list[FeatureMeta] = (
    [
        FeatureMeta(
            name=f"volume_ratio_{w}d",
            description=f"Today's Volume relative to its {w}-day moving average",
            formula=f"Volume / SMA(Volume, {w})",
            lookback=w,
            required_columns=("volume",),
            warmup_period=w,
            directionality="bullish_high",
        )
        for w in VOLUME_RATIO_WINDOWS
    ]
    + [
        FeatureMeta(
            name="volume_zscore",
            description=f"Volume z-score against its {VOLUME_ZSCORE_WINDOW}-day mean/std",
            formula=(
                f"(Volume - SMA(Volume,{VOLUME_ZSCORE_WINDOW}))"
                f"/std(Volume,{VOLUME_ZSCORE_WINDOW})"
            ),
            lookback=VOLUME_ZSCORE_WINDOW,
            required_columns=("volume",),
            warmup_period=VOLUME_ZSCORE_WINDOW,
            directionality="bullish_high",
        ),
        FeatureMeta(
            name="volume_trend",
            description=(
                f"{VOLUME_TREND_SLOPE_WINDOW}-day fractional slope of the "
                f"{VOLUME_TREND_MA_WINDOW}-day volume moving average"
            ),
            formula=(
                f"slope(SMA(Volume,{VOLUME_TREND_MA_WINDOW}), "
                f"window={VOLUME_TREND_SLOPE_WINDOW})"
            ),
            lookback=VOLUME_TREND_MA_WINDOW + VOLUME_TREND_SLOPE_WINDOW,
            required_columns=("volume",),
            warmup_period=VOLUME_TREND_MA_WINDOW + VOLUME_TREND_SLOPE_WINDOW,
            directionality="bullish_high",
        ),
    ]
)
