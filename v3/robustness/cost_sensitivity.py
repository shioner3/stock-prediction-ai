"""Spec section 16: Cost Sensitivity across the SAME 4 fixed tiers
(Zero/Low/Base/High, `config/settings.yaml:validation.transaction_cost.
tiers`) every prior Phase already uses, applied via `backtest.costs.
apply_cost()` (unmodified - a flat round-trip bps deduction from each raw
return, applied identically to every row regardless of bucket).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.costs import apply_cost
from config.loader import TransactionCostTierConfig
from v3.robustness.common_eval import SliceEvaluation, evaluate_slice


def apply_cost_tier(
    predictions: pd.DataFrame, tier: TransactionCostTierConfig, actual_col: str = "actual",
) -> pd.DataFrame:
    out = predictions.copy()
    out[f"{actual_col}_cost_{tier.name}"] = apply_cost(out[actual_col], tier)
    return out


@dataclass(frozen=True)
class CostTierResult:
    tier_name: str
    round_trip_bps: float
    evaluation: SliceEvaluation


def run_cost_sensitivity(
    predictions: pd.DataFrame, tiers: list[TransactionCostTierConfig], window_days: int,
    prediction_col: str = "prediction", actual_col: str = "actual",
) -> dict[str, CostTierResult]:
    results: dict[str, CostTierResult] = {}
    for tier in tiers:
        with_cost = apply_cost_tier(predictions, tier, actual_col)
        cost_col = f"{actual_col}_cost_{tier.name}"
        evaluation = evaluate_slice(with_cost, tier.name, window_days, prediction_col, cost_col)
        results[tier.name] = CostTierResult(
            tier_name=tier.name, round_trip_bps=tier.round_trip_bps, evaluation=evaluation
        )
    return results
