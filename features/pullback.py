"""Pullback features: distance from the recent high/low, distance from
short moving averages, and the length of the current losing streak.

distance_from_recent_low / bounce_depth were added in Phase 4: LONG's
Pullback signal (buy the dip) reads pullback_depth (distance below the
recent high); SHORT's mirror-image Pullback signal (sell the bounce in a
downtrend) needed the symmetric "distance above the recent low" - adding
it here keeps that signal reading from the Feature layer instead of
recomputing a rolling min inline (see features/breakout.py's low_Nd for
the same reasoning, and README's Phase 4 notes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features._utils import consecutive_count, safe_divide, sma
from features.metadata import FeatureMeta

# recent_high_window is configurable (config/settings.yaml: features.pullback.recent_high_window)
# because "recent" is a judgment call the user should be able to tune; this is the fallback
# used when compute_pullback_features() is called without an explicit override. The same
# window governs both the recent-high and recent-low references below (kept as a single
# parameter rather than two, to avoid config bloat - a 20-day high and 20-day low are the
# same lookback in every practical use of this feature).
DEFAULT_RECENT_HIGH_WINDOW = 20

DISTANCE_SMA_WINDOWS = (5, 20)


def compute_pullback_features(
    df: pd.DataFrame, recent_high_window: int = DEFAULT_RECENT_HIGH_WINDOW
) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # Unlike breakout.py's high_Nd/low_Nd, recent_high/recent_low INCLUDE
    # today - "distance from the recent high" should read as 0 when today
    # IS the high, not be forced non-zero by artificially excluding
    # today's own price. This is still no-lookahead-safe: rolling(window)
    # at row t only ever reads rows <= t.
    recent_high = close.rolling(recent_high_window, min_periods=recent_high_window).max()
    out["distance_from_recent_high"] = safe_divide(close - recent_high, recent_high)
    out["pullback_depth"] = -out["distance_from_recent_high"]

    recent_low = close.rolling(recent_high_window, min_periods=recent_high_window).min()
    out["distance_from_recent_low"] = safe_divide(close - recent_low, recent_low)
    out["bounce_depth"] = out["distance_from_recent_low"]

    for window in DISTANCE_SMA_WINDOWS:
        out[f"distance_from_sma{window}"] = safe_divide(close, sma(close, window)) - 1

    is_down_day = close < close.shift(1)
    consecutive = consecutive_count(is_down_day)
    # Row 0 has no prior close, so "down or not" is undefined there, not 0.
    if len(consecutive) > 0:
        consecutive.iloc[0] = np.nan
    out["consecutive_down_days"] = consecutive

    return out


def pullback_metadata(recent_high_window: int = DEFAULT_RECENT_HIGH_WINDOW) -> list[FeatureMeta]:
    return (
        [
            FeatureMeta(
                name="distance_from_recent_high",
                description=(
                    f"Close's distance from its {recent_high_window}-day rolling high "
                    "(today included), as a fraction; always <= 0"
                ),
                formula=f"Close/max(Close[t-{recent_high_window - 1}..t]) - 1",
                lookback=recent_high_window,
                required_columns=("close",),
                warmup_period=recent_high_window,
                directionality="bullish_high",
            ),
            FeatureMeta(
                name="pullback_depth",
                description="Magnitude of the pullback from the recent high; always >= 0",
                formula="-distance_from_recent_high",
                lookback=recent_high_window,
                required_columns=("close",),
                warmup_period=recent_high_window,
                directionality="bullish_low",
            ),
            FeatureMeta(
                name="distance_from_recent_low",
                description=(
                    f"Close's distance above its {recent_high_window}-day rolling low "
                    "(today included), as a fraction; always >= 0"
                ),
                formula=f"Close/min(Close[t-{recent_high_window - 1}..t]) - 1",
                lookback=recent_high_window,
                required_columns=("close",),
                warmup_period=recent_high_window,
                directionality="bullish_low",
            ),
            FeatureMeta(
                name="bounce_depth",
                description="Magnitude of the bounce up from the recent low; always >= 0 "
                "(SHORT Pullback's mirror of pullback_depth)",
                formula="distance_from_recent_low",
                lookback=recent_high_window,
                required_columns=("close",),
                warmup_period=recent_high_window,
                directionality="bullish_high",
            ),
        ]
        + [
            FeatureMeta(
                name=f"distance_from_sma{w}",
                description=f"Close's distance from SMA{w}, as a fraction (same formula "
                f"as trend.py's close_to_sma_{w}, kept as its own named column here "
                "since Pullback signals should be able to reference it without a "
                "dependency on the Trend feature group)",
                formula=f"Close/SMA{w} - 1",
                lookback=w,
                required_columns=("close",),
                warmup_period=w,
                directionality="bullish_high",
            )
            for w in DISTANCE_SMA_WINDOWS
        ]
        + [
            FeatureMeta(
                name="consecutive_down_days",
                description="Number of consecutive prior sessions (ending today) with "
                "Close < previous Close",
                formula="running count of Close[d] < Close[d-1], reset on any up/flat day",
                lookback=1,
                required_columns=("close",),
                warmup_period=2,
                directionality="bullish_low",
            )
        ]
    )


PULLBACK_METADATA: list[FeatureMeta] = pullback_metadata()
