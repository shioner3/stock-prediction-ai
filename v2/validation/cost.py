"""Transaction Cost sensitivity (spec section 15): reuses V1's
backtest.costs.apply_cost() (a flat round-trip bps deduction, unmodified)
and config.loader.TransactionCostTierConfig (the same generic 4-field
shape every V1 cost-sensitivity check already uses), instantiated here
with the SAME 4 tiers spec section 15 names verbatim (0/10/30/80bps) -
these are not read from config/settings.yaml (V2 never depends on V1's
config VALUES, only its pure TYPES/FUNCTIONS - see v2/validation/
__init__.py's interpretation note), just literal numbers matching the
spec text exactly.

Applied directly to V2's raw Forward Return (price drift, NOT a
simulated trade - see v2/stats.py's own docstring on this distinction),
per spec section 15's explicit instruction not to conflate Forward
Return with real trading cost mechanics; this is a simple net-of-cost
approximation for research purposes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.costs import apply_cost
from config.loader import TransactionCostTierConfig
from v2.stats import ReturnStats, compute_return_stats

COST_TIERS: list[TransactionCostTierConfig] = [
    TransactionCostTierConfig(name="zero", round_trip_bps=0.0),
    TransactionCostTierConfig(name="low", round_trip_bps=10.0),
    TransactionCostTierConfig(name="base", round_trip_bps=30.0),
    TransactionCostTierConfig(name="high", round_trip_bps=80.0),
]


@dataclass(frozen=True)
class CostSensitivityResult:
    tier_name: str
    round_trip_bps: float
    gross_stats: ReturnStats
    net_stats: ReturnStats


def compute_cost_sensitivity(
    returns: pd.Series, tiers: list[TransactionCostTierConfig] = COST_TIERS
) -> list[CostSensitivityResult]:
    gross_stats = compute_return_stats(returns)
    results = []
    for tier in tiers:
        net_returns = apply_cost(returns, tier)
        results.append(
            CostSensitivityResult(
                tier_name=tier.name, round_trip_bps=tier.round_trip_bps,
                gross_stats=gross_stats, net_stats=compute_return_stats(net_returns),
            )
        )
    return results
