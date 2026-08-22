"""Relative Strength: excess return of the stock over a market benchmark.

    rs_Nd[t] = stock_return_Nd[t] - market_return_Nd[t]

Both legs use features/momentum.py's compute_return() - the exact same
formula Phase 2 already uses for return_Nd - so RS is never computed from
an independently re-derived return formula (a ratio-based definition,
e.g. stock_return / market_return, is deliberately NOT used: it blows up
to extreme values or inf whenever market_return is near zero).

The market benchmark is currently "TOPIX Proxy" (1306.T, a TOPIX-tracking
ETF), not the raw TOPIX index - see providers/market_index.py and
config/settings.yaml's data.market_index block for why. This module never
fetches 1306.T itself; it only ever receives market_df as a plain OHLCV
DataFrame, already retrieved via MarketIndexProvider upstream (see
pipeline/build_features.py) - so swapping the proxy for a real index feed
later (e.g. J-Quants) requires no change here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.metadata import FeatureMeta
from features.momentum import compute_return

RS_WINDOWS = (5, 20, 60)

DEFAULT_MARKET_LABEL = "TOPIX Proxy (1306.T)"


def compute_relative_strength_features(
    stock_df: pd.DataFrame, market_df: pd.DataFrame | None
) -> pd.DataFrame:
    """stock_df: a single ticker's OHLCV (needs "close" and "date").
    market_df: the benchmark's OHLCV in the same shape, or None when the
    market benchmark could not be fetched (providers/market_index.py's
    topix_available=False case) - every rs_Nd column is then NaN, never
    0 or any other silently-substituted value.
    """
    out = pd.DataFrame(index=stock_df.index)

    if market_df is None or market_df.empty:
        for window in RS_WINDOWS:
            out[f"rs_{window}d"] = np.nan
        return out

    stock_close = stock_df["close"]
    stock_dates = pd.to_datetime(stock_df["date"])

    # Market returns are computed on the market's OWN chronological
    # sequence (never on the stock's row order/positions) and only THEN
    # aligned onto the stock's dates by date value - this is what stops a
    # stock/market trading-calendar mismatch (holiday differences, a
    # halted stock, a gap in the ETF's own history, ...) from silently
    # shifting one side's data onto the wrong day. Dates are normalised
    # via pd.to_datetime on both sides so a dtype difference between the
    # two input DataFrames (e.g. datetime.date vs Timestamp) can't turn
    # into a silent all-NaN alignment failure.
    market_dates = pd.to_datetime(market_df["date"])
    market_by_date = pd.Series(market_df["close"].to_numpy(), index=market_dates).sort_index()
    market_by_date = market_by_date[~market_by_date.index.duplicated(keep="last")]

    for window in RS_WINDOWS:
        stock_return = compute_return(stock_close, window)
        market_return_by_date = compute_return(market_by_date, window)

        # reindex(): a stock date absent from the market's date index
        # becomes NaN here - NEVER forward-filled from a neighbouring (and,
        # across a gap, potentially future) market value. No `method=`
        # argument is passed to reindex() precisely to guarantee this.
        aligned_market_return = market_return_by_date.reindex(stock_dates).to_numpy()

        out[f"rs_{window}d"] = stock_return.to_numpy() - aligned_market_return

    return out


def relative_strength_metadata(
    market_label: str = DEFAULT_MARKET_LABEL,
) -> list[FeatureMeta]:
    # warmup_period is the structural minimum, assuming the market
    # benchmark fully covers the stock's date range with no gaps of its
    # own. A real gap in the benchmark (delisting, halt, missing ETF
    # data) shows up as additional NaN beyond this minimum - that is
    # correct behaviour (see module docstring), not a bug, and is why
    # this project doesn't try to declare a single universally-exact
    # warmup for a two-series feature like this one.
    return [
        FeatureMeta(
            name=f"rs_{w}d",
            description=(
                f"{w}-day excess return of the stock over the market benchmark "
                f"({market_label} - not the raw TOPIX index)"
            ),
            formula=f"return_{w}d(stock) - return_{w}d(market)",
            lookback=w,
            required_columns=("close", "date"),
            warmup_period=w + 1,
            directionality="bullish_high",
        )
        for w in RS_WINDOWS
    ]


RELATIVE_STRENGTH_METADATA: list[FeatureMeta] = relative_strength_metadata()
