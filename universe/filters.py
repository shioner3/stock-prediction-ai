"""Dynamic universe filters that require fetched OHLCV data (price,
liquidity, trading activity).

Applied by the ingest pipeline after data has been fetched for a ticker -
kept separate from universe/build.py's static filters because these
cannot be evaluated from the master list alone.

Phase 6.5 Data/Universe Leakage Fix (2026-08-20): this filter evaluates
the EARLIEST `lookback_days` rows of the ticker's fetched history, not
the most recent ones. pipeline/ingest.py fetches a ticker's ENTIRE
configured date range (e.g. 2022-01-04..2024-06-28) and calls this
function ONCE to decide whether the ticker enters the Universe AT ALL -
for its FULL history, including Walk Forward windows years before the
fetch's end date. Using the most recent rows (as this function did
before this fix) meant a stock's inclusion in its own 2022 Signals was
being decided by its liquidity in mid-2024 - textbook look-ahead bias at
the Universe-construction level. Using the earliest rows instead ties
the eligibility decision to "was this stock minimally investable at the
start of the evaluated period", which cannot see anything beyond it.
This is a Universe-construction-time approximation, not a per-day
liquidity gate - a stock that started liquid and later became illiquid
(or vice versa) still has its entire history included or excluded as a
single unit, exactly as this module did before the fix (only the
direction of the lookback window changed, not the two-stage-filter
architecture itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.loader import UniverseFilterConfig


@dataclass
class LiquidityCheckResult:
    ticker: str
    passed: bool
    reason: str | None


def check_price_and_liquidity(
    df: pd.DataFrame,
    ticker: str,
    config: UniverseFilterConfig,
    lookback_days: int = 20,
) -> LiquidityCheckResult:
    if df.empty:
        return LiquidityCheckResult(ticker, False, "no data available")

    # .head(), not .tail(): see this module's docstring (Phase 6.5 Leakage
    # Fix) - the earliest rows represent "at the start of the evaluated
    # period", the only choice that cannot leak information from later
    # Walk Forward windows into an earlier one's Universe membership.
    earliest = df.sort_values("date").head(lookback_days)
    avg_close = earliest["close"].mean()
    avg_volume = earliest["volume"].mean()
    zero_volume_days = int((earliest["volume"] == 0).sum())

    price_cfg = config.price
    if price_cfg.min_close_price is not None and avg_close < price_cfg.min_close_price:
        return LiquidityCheckResult(
            ticker, False, f"avg_close {avg_close:.1f} < min {price_cfg.min_close_price}"
        )
    if price_cfg.max_close_price is not None and avg_close > price_cfg.max_close_price:
        return LiquidityCheckResult(
            ticker, False, f"avg_close {avg_close:.1f} > max {price_cfg.max_close_price}"
        )

    liq_cfg = config.liquidity
    if liq_cfg.min_avg_volume_20d is not None and avg_volume < liq_cfg.min_avg_volume_20d:
        return LiquidityCheckResult(
            ticker, False, f"avg_volume {avg_volume:.0f} < min {liq_cfg.min_avg_volume_20d}"
        )

    act_cfg = config.activity
    if (
        act_cfg.max_zero_volume_days_in_20 is not None
        and zero_volume_days > act_cfg.max_zero_volume_days_in_20
    ):
        return LiquidityCheckResult(
            ticker, False,
            f"{zero_volume_days} zero-volume days > max {act_cfg.max_zero_volume_days_in_20}",
        )

    return LiquidityCheckResult(ticker, True, None)
