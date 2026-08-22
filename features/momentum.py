"""Return-based momentum features: Close[t]/Close[t-n] - 1."""

from __future__ import annotations

import pandas as pd

from features._utils import safe_divide
from features.metadata import FeatureMeta

RETURN_WINDOWS = (1, 3, 5, 10, 20, 60)


def compute_return(close: pd.Series, window: int) -> pd.Series:
    """Close[t]/Close[t-window] - 1. The single formula for "return" used
    everywhere in this project - features/relative_strength.py (Phase 3)
    calls this directly for both legs of Relative Strength rather than
    re-deriving its own return calculation.
    """
    prior = close.shift(window)
    return safe_divide(close - prior, prior)


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for window in RETURN_WINDOWS:
        out[f"return_{window}d"] = compute_return(close, window)
    return out


MOMENTUM_METADATA: list[FeatureMeta] = [
    FeatureMeta(
        name=f"return_{w}d",
        description=f"{w}-day simple return of Close",
        formula=f"Close[t]/Close[t-{w}] - 1",
        lookback=w,
        required_columns=("close",),
        warmup_period=w + 1,
        directionality="bullish_high",
    )
    for w in RETURN_WINDOWS
]
