"""v3/robustness/cost_sensitivity.py - verifies the flat round-trip bps
deduction is applied identically to every row, and that a higher-cost
tier strictly worsens every summary statistic.
"""

from __future__ import annotations

import pandas as pd

from config.loader import TransactionCostTierConfig
from v3.robustness.cost_sensitivity import apply_cost_tier, run_cost_sensitivity

TIERS = [
    TransactionCostTierConfig(name="zero", round_trip_bps=0.0),
    TransactionCostTierConfig(name="base", round_trip_bps=30.0),
    TransactionCostTierConfig(name="high", round_trip_bps=80.0),
]


def _predictions() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=10)
    rows = []
    for i, d in enumerate(dates):
        for t in range(6):
            rows.append({"date": d.date(), "ticker": f"T{t}", "prediction": t, "actual": 0.01 * t})
    return pd.DataFrame(rows)


def test_apply_cost_tier_deducts_exact_bps() -> None:
    predictions = _predictions()
    tier = TransactionCostTierConfig(name="base", round_trip_bps=30.0)
    out = apply_cost_tier(predictions, tier)
    expected = predictions["actual"] - 30.0 / 10_000
    assert (abs(out["actual_cost_base"] - expected) < 1e-12).all()


def test_higher_cost_tier_reduces_mean_return() -> None:
    predictions = _predictions()
    results = run_cost_sensitivity(predictions, TIERS, window_days=5)
    zero_top5 = results["zero"].evaluation.topn[5].base.stats.mean_return
    base_top5 = results["base"].evaluation.topn[5].base.stats.mean_return
    high_top5 = results["high"].evaluation.topn[5].base.stats.mean_return
    assert zero_top5 > base_top5 > high_top5


def test_zero_cost_tier_matches_uncosted_evaluation() -> None:
    predictions = _predictions()
    from v3.validation.ranking_metrics import evaluate_ranking

    results = run_cost_sensitivity(predictions, TIERS, window_days=5)
    zero_spread = results["zero"].evaluation.ranking.q5_q1_spread
    uncosted_spread = evaluate_ranking(predictions, 5).q5_q1_spread
    assert zero_spread == uncosted_spread
