"""Spec section 3.C: Market-beta-adjusted Return - the one return
definition variant with no existing V1/V2/V3 primitive to reuse (unlike
TOPIX-relative, which is already `target_topix_relative_5d` - see
`market_decomposition.py`).

Beta is estimated PURELY from two columns already present in every V3
Full Universe dataset row (`return_1d`, `topix_return_1d` - both CORE
Feature Registry entries, `v3/features/registry.py`, no new Feature
added): a trailing `BETA_WINDOW`-day rolling covariance(return_1d,
topix_return_1d) / variance(topix_return_1d), computed causally (pandas
`.rolling()`, ending at and including day t - the same "available at t"
convention every other point-in-time Feature in the Registry already
uses, so this carries no lookahead beyond what `v3/leakage/
availability_check.py` already verifies for `return_1d`/`topix_return_1d`
themselves). NaN until `BETA_WINDOW` trading days of history exist for a
ticker, exactly like any other rolling-window Feature's warmup period.

This is a Phase V3-4 ANALYSIS-time-only quantity (used to decompose an
already-realized forward return into market-driven vs residual
components) - it is never fed back into Model A/B/C, never added to the
Feature Registry, and does not change any V3-1/V3-2/V3-3 spec.

**Bug found and fixed during this Phase's real Full Universe run**
(`research/phase_v3_4_report.md`'s own "Bugs Discovered" section):
when the trailing `BETA_WINDOW`-day window happens to contain almost no
TOPIX movement (a near-zero rolling variance denominator), the resulting
beta explodes to physically implausible magnitudes (observed up to
~7.8 million on real Full Universe data, versus any real stock's beta
realistically sitting within roughly +/-5 of the market). This is the
same "near-zero-denominator blow-up" failure class already documented
for `target_risk_adjusted_5d` in `v3/validation/data_integrity.py` -
`MAX_PLAUSIBLE_BETA` bounds it the same way: values outside the bound are
set to NaN (excluded from downstream beta-adjusted-return rows via the
existing `dropna()` calls in `market_decomposition.py`), never clipped
to the boundary (clipping would silently fabricate a beta value; NaN
honestly represents "no plausible beta estimate available").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA_WINDOW = 60
MIN_BETA_PERIODS = 40  # >= 2/3 of BETA_WINDOW - avoids a beta estimated from too few points
MAX_PLAUSIBLE_BETA = 5.0  # physically-motivated bound - see module docstring's "Bug found" note


def compute_rolling_beta(
    dataset: pd.DataFrame,
    stock_return_col: str = "return_1d",
    market_return_col: str = "topix_return_1d",
    window: int = BETA_WINDOW,
    min_periods: int = MIN_BETA_PERIODS,
) -> pd.DataFrame:
    """Returns date/ticker/beta - one row per (date, ticker) in `dataset`."""
    sorted_panel = dataset[["date", "ticker", stock_return_col, market_return_col]].sort_values(
        ["ticker", "date"]
    )
    betas = []
    for _ticker, group in sorted_panel.groupby("ticker", sort=False):
        stock_r = group[stock_return_col]
        market_r = group[market_return_col]
        cov = stock_r.rolling(window, min_periods=min_periods).cov(market_r)
        var = market_r.rolling(window, min_periods=min_periods).var()
        # np.nan (not pd.NA) - see v3/models/data_prep.py's own docstring
        # for why replacing into a float64 Series with pd.NA silently
        # upcasts it to `object` dtype.
        beta = cov / var.replace(0, np.nan)
        beta = beta.where(beta.abs() <= MAX_PLAUSIBLE_BETA)
        betas.append(pd.DataFrame({"date": group["date"], "ticker": group["ticker"], "beta": beta}))
    return pd.concat(betas, ignore_index=True)


def attach_beta(predictions: pd.DataFrame, beta_panel: pd.DataFrame) -> pd.DataFrame:
    return predictions.merge(beta_panel, on=["date", "ticker"], how="left")
