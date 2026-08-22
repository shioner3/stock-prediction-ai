"""Breakout features: the highest/lowest Close over the prior N sessions,
STRICTLY EXCLUDING today.

    high_Nd[t] = max(Close[t-N], ..., Close[t-1])
    low_Nd[t]  = min(Close[t-N], ..., Close[t-1])

This is why every window here is built as `close.shift(1).rolling(N)`
rather than `close.rolling(N)`: shifting by 1 *before* taking the rolling
max/min removes today's own Close from the window, so a later Signal
layer can safely test `Close[t] > high_Nd[t]` (breakout) or
`Close[t] < low_Nd[t]` (breakdown) without the day's own close having
leaked into the threshold it's being compared against. Never rewrite
this as "rolling(N) then compare" - that would include today's Close in
its own comparison threshold.

low_Nd was added in Phase 4 (LONG Breakout already had high_Nd since
Phase 2; SHORT Breakdown needed the symmetric feature, and inventing it
inline inside signals/short/breakdown.py would have meant reimplementing
a rolling calculation outside the Feature layer - see README's Phase 4
notes). It's a pure structural mirror of high_Nd, same warmup, same
shift(1)-before-rolling requirement.
"""

from __future__ import annotations

import pandas as pd

from features.metadata import FeatureMeta

HIGH_LOW_WINDOWS = (5, 10, 20, 60, 120)


def compute_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    shifted_close = df["close"].shift(1)
    out = pd.DataFrame(index=df.index)
    for window in HIGH_LOW_WINDOWS:
        rolling = shifted_close.rolling(window, min_periods=window)
        out[f"high_{window}d"] = rolling.max()
        out[f"low_{window}d"] = rolling.min()
    return out


BREAKOUT_METADATA: list[FeatureMeta] = [
    FeatureMeta(
        name=f"high_{w}d",
        description=f"Highest Close over the {w} sessions before today (today excluded)",
        formula=f"max(Close[t-{w}..t-1])",
        lookback=w,
        required_columns=("close",),
        warmup_period=w + 1,
        directionality="neutral",
    )
    for w in HIGH_LOW_WINDOWS
] + [
    FeatureMeta(
        name=f"low_{w}d",
        description=f"Lowest Close over the {w} sessions before today (today excluded)",
        formula=f"min(Close[t-{w}..t-1])",
        lookback=w,
        required_columns=("close",),
        warmup_period=w + 1,
        directionality="neutral",
    )
    for w in HIGH_LOW_WINDOWS
]
