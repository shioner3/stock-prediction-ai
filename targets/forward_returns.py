"""Forward Returns, MFE, and MAE - RESEARCH-ONLY targets used exclusively
by Score Validation (scoring/validation.py).

This module must NEVER be imported by features/, signals/, or scoring/'s
scorer.py/pipeline.py - see tests/test_target_leakage.py, which asserts
this by static AST inspection, not just by convention. The dependency
direction is:

    OHLCV -> targets/forward_returns.py -> Score Validation (reads Score
                                            Records + these targets, but
                                            never feeds them back)

Forward Return[t,n] = Close[t+n] / Close[t] - 1

This is DELIBERATELY DIFFERENT from the Backtest Engine's convention
(backtest/engine.py: Entry = Open[t+1], Exit = Close after hold_days).
Forward Return here measures raw price drift from the Signal date's own
Close - a quick, backtest-independent way to ask "does Score correlate
with what happens next", not a trading rule. Never conflate the two.
"""

from __future__ import annotations

import pandas as pd

FORWARD_WINDOWS = (1, 3, 5, 7, 10, 15, 20)


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """df: a single ticker's OHLCV/Feature panel (needs "close"). Returns
    forward_return_{n}d for n in FORWARD_WINDOWS. The LAST n rows of any
    dataset are always NaN here (no t+n data exists yet) - this is
    expected, not a bug; it is the mirror image of a warmup NaN.
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for n in FORWARD_WINDOWS:
        out[f"forward_return_{n}d"] = close.shift(-n) / close - 1
    return out


def _forward_extreme(series: pd.Series, n: int, how: str) -> pd.Series:
    """max/min of series[t+1 .. t+n] for every t, computed by reversing
    the series, taking a normal (backward) rolling extreme, then
    reversing back - the standard trick for a forward-looking rolling
    window in pandas (which has no native forward rolling()).
    """
    reversed_series = series.iloc[::-1]
    shifted = reversed_series.shift(1)
    rolled = shifted.rolling(n, min_periods=n).max() if how == "max" else shifted.rolling(
        n, min_periods=n
    ).min()
    return rolled.iloc[::-1]


def compute_mfe_mae(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Maximum Favorable/Adverse Excursion over each forward window,
    using High/Low within (t, t+n] (not just Close) for a more realistic
    best/worst-case measure than forward_return alone gives.

    Sign convention (same for both directions): MFE is the best case
    relative to Signal-date Close and is typically >= 0; MAE is the
    worst case and is typically <= 0. Both are simple returns relative
    to Close[t], NOT relative to the Backtest Engine's actual Entry
    price (Open[t+1]) - see this module's docstring.

        LONG:  MFE = max(High[t+1..t+n])/Close[t] - 1
               MAE = min(Low[t+1..t+n])/Close[t]  - 1
        SHORT: MFE = 1 - min(Low[t+1..t+n])/Close[t]
               MAE = 1 - max(High[t+1..t+n])/Close[t]
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    out = pd.DataFrame(index=df.index)

    for n in FORWARD_WINDOWS:
        forward_high = _forward_extreme(high, n, "max")
        forward_low = _forward_extreme(low, n, "min")
        if direction == "LONG":
            out[f"mfe_{n}d"] = forward_high / close - 1
            out[f"mae_{n}d"] = forward_low / close - 1
        else:
            out[f"mfe_{n}d"] = 1 - forward_low / close
            out[f"mae_{n}d"] = 1 - forward_high / close

    return out
