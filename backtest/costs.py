"""Transaction Cost sensitivity (Phase 6): applies a fixed round-trip
cost, in basis points, to trade returns. Deliberately simple - a flat
deduction, not a model of spread/impact/partial fills - applied
identically regardless of direction (a round trip costs the same going
in and out whether LONG or SHORT).

The 4 tiers (zero/low/base/high) are fixed in
config/settings.yaml:validation.transaction_cost.tiers and are never
adjusted based on how a Signal performs under them (Phase 6's "最重要
ルール").
"""

from __future__ import annotations

import pandas as pd

from config.loader import TransactionCostTierConfig


def apply_cost(returns: pd.Series, tier: TransactionCostTierConfig) -> pd.Series:
    return returns - (tier.round_trip_bps / 10_000.0)
