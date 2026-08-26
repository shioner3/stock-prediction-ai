"""Future-shock test helpers (spec section 23 A-D): reusable functions
that mutate rows AFTER a cutoff date in a raw OHLCV DataFrame, so a test
can rebuild the V3 Feature panel from the shocked data and assert every
row with date <= cutoff is byte-identical to the unshocked panel. If it
is not, the shocked (future) values leaked backward into an
earlier-dated Feature - LEAKAGE_FOUND.

These operate on raw OHLCV (before Feature computation), not on an
already-computed Feature panel - a shock to `close`/`volume` here
propagates through V1's real rolling()/ewm() computations exactly as a
genuine data anomaly would, which is a stronger test than directly
mutating an already-computed Feature column.
"""

from __future__ import annotations

from datetime import date as date_type

import numpy as np
import pandas as pd


def shock_price_after(
    ohlcv: pd.DataFrame, cutoff: date_type, multiplier: float = 5.0, date_col: str = "date"
) -> pd.DataFrame:
    """spec 23.A: Future price shock - close/high/low/open multiplied
    after `cutoff` (a large, unmissable perturbation)."""
    shocked = ohlcv.copy()
    mask = shocked[date_col] > cutoff
    for col in ("open", "high", "low", "close"):
        if col in shocked.columns:
            shocked.loc[mask, col] = shocked.loc[mask, col] * multiplier
    return shocked


def shock_volume_after(
    ohlcv: pd.DataFrame, cutoff: date_type, multiplier: float = 10.0, date_col: str = "date"
) -> pd.DataFrame:
    """spec 23.C: Future volume shock."""
    shocked = ohlcv.copy()
    mask = shocked[date_col] > cutoff
    shocked.loc[mask, "volume"] = shocked.loc[mask, "volume"] * multiplier
    return shocked


def shock_index_after(
    market_ohlcv: pd.DataFrame, cutoff: date_type, multiplier: float = 5.0, date_col: str = "date"
) -> pd.DataFrame:
    """spec 23.B: Future index (TOPIX Proxy) shock - same shape as
    shock_price_after, applied to the market benchmark's own OHLCV."""
    return shock_price_after(market_ohlcv, cutoff, multiplier=multiplier, date_col=date_col)


def random_perturb_after(
    ohlcv: pd.DataFrame, cutoff: date_type, seed: int, date_col: str = "date"
) -> pd.DataFrame:
    """spec 23.D: Random future perturbation - close/high/low/open/volume
    after `cutoff` replaced with an independent random walk (not merely
    scaled), so a leakage bug that only depends on RELATIVE (not
    absolute) future values would still be caught.
    """
    rng = np.random.default_rng(seed)
    shocked = ohlcv.copy()
    mask = shocked[date_col] > cutoff
    n = int(mask.sum())
    if n == 0:
        return shocked
    base_price = float(shocked.loc[~mask, "close"].iloc[-1]) if (~mask).any() else 1000.0
    random_walk = base_price * np.exp(np.cumsum(rng.normal(0, 0.05, size=n)))
    shocked.loc[mask, "close"] = random_walk
    shocked.loc[mask, "open"] = random_walk * (1 + rng.normal(0, 0.01, size=n))
    shocked.loc[mask, "high"] = np.maximum(shocked.loc[mask, "open"], random_walk) * (
        1 + np.abs(rng.normal(0, 0.01, size=n))
    )
    shocked.loc[mask, "low"] = np.minimum(shocked.loc[mask, "open"], random_walk) * (
        1 - np.abs(rng.normal(0, 0.01, size=n))
    )
    shocked.loc[mask, "volume"] = rng.integers(1000, 1_000_000, size=n)
    return shocked
