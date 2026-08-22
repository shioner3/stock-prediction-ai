"""Market Regime classification (Phase 6 section 20), based purely on
the market benchmark's (TOPIX Proxy) own trailing return - no
information beyond what Phase 1-3 already fetch, and no new
forward-looking data of any kind.

    regime[t] = BULL    if return_{lookback_days}d(market)[t] > bull_threshold
                BEAR    if return_{lookback_days}d(market)[t] < bear_threshold
                NEUTRAL otherwise
                NaN     during the return's own warmup (insufficient history)

Reuses features/momentum.py's compute_return() directly - no new
rolling/return calculation is introduced here.
"""

from __future__ import annotations

import pandas as pd

from config.loader import MarketRegimeConfig
from features.momentum import RETURN_WINDOWS, compute_return


def compute_market_regime(market_df: pd.DataFrame, config: MarketRegimeConfig) -> pd.DataFrame:
    if config.lookback_days not in RETURN_WINDOWS:
        raise ValueError(
            f"market_regime.lookback_days={config.lookback_days} has no matching return "
            f"feature - must be one of {RETURN_WINDOWS} (features/momentum.py)"
        )

    trailing_return = compute_return(market_df["close"], config.lookback_days)

    regime = pd.Series("NEUTRAL", index=market_df.index, dtype=object)
    regime = regime.mask(trailing_return > config.bull_threshold, "BULL")
    regime = regime.mask(trailing_return < config.bear_threshold, "BEAR")
    regime = regime.mask(trailing_return.isna(), None)

    return pd.DataFrame({"date": market_df["date"], "regime": regime})
