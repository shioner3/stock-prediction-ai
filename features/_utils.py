"""Low-level numeric helpers shared across feature modules.

Every function here is a pure function of its inputs (no I/O, no Provider
access) and only ever looks backward in the input series via rolling(),
ewm(), or shift(+n) - never shift(-n) or any operation that reads ahead
of the current row. This is the structural basis for the project's
no-lookahead guarantee (see tests/test_no_lookahead.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. NaN for the first window-1 rows."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (span-based, adjust=False), masked to
    NaN for the first span-1 rows so its warmup matches sma()'s
    convention. Without this, pandas' ewm() returns a (low-quality, since
    barely any history informed it) value from row 0 onward.
    """
    result = series.ewm(span=span, adjust=False).mean()
    return mask_warmup(result, span)


def mask_warmup(series: pd.Series, warmup_period: int) -> pd.Series:
    """Force the first warmup_period - 1 rows to NaN."""
    result = series.astype(float).copy()
    result.iloc[: warmup_period - 1] = np.nan
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise division that yields NaN (never +/-inf) where the
    denominator is zero or NaN, instead of pandas' default inf/-inf.
    """
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def slope(series: pd.Series, window: int) -> pd.Series:
    """Fractional per-day rate of change of `series` over `window` days:

        slope[t] = (series[t] - series[t-window]) / series[t-window] / window

    Backward-looking only (uses shift(+window), never shift(-window)).
    """
    prior = series.shift(window)
    return safe_divide(series - prior, prior) / window


def consecutive_count(condition: pd.Series) -> pd.Series:
    """Length of the current consecutive run of True values in `condition`,
    ending at each row (0 for a row where condition is False). Returned
    as float so callers can NaN out rows where the condition itself was
    undefined (e.g. no prior value to compare against).
    """
    cond = condition.fillna(False).astype(int)
    reset_groups = (cond == 0).cumsum()
    counts = cond.groupby(reset_groups).cumsum()
    return counts.astype(float)
