from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v3.models.cross_sectional import add_random_baseline_column, evaluate_cross_sectional_ranking


def _synthetic_scored(n_days: int = 20, n_tickers: int = 30, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        pred = rng.uniform(0, 1, size=n_tickers)
        target = pred * 0.02 + rng.normal(0, 0.001, size=n_tickers)  # monotonic relationship
        for i in range(n_tickers):
            rows.append((d, f"T{i}", pred[i], target[i]))
    return pd.DataFrame(rows, columns=["date", "ticker", "_prediction", "target_raw_5d"])


def test_cross_sectional_detects_a_real_monotonic_relationship() -> None:
    scored = _synthetic_scored()
    result = evaluate_cross_sectional_ranking(scored, "_prediction", "target_raw_5d", window_days=5)
    assert result.q5_q1_spread is not None
    assert result.q5_q1_spread > 0
    assert len(result.bucket_stats) == 5


def test_random_baseline_column_has_no_systematic_relationship() -> None:
    scored = _synthetic_scored()
    with_random = add_random_baseline_column(scored, seed=101)
    result = evaluate_cross_sectional_ranking(
        with_random, "_random_baseline", "target_raw_5d", window_days=5
    )
    # A genuinely random ranking's spread should be much smaller in
    # magnitude than the real (monotonic-by-construction) prediction's.
    real_result = evaluate_cross_sectional_ranking(
        scored, "_prediction", "target_raw_5d", window_days=5
    )
    assert abs(result.q5_q1_spread) < abs(real_result.q5_q1_spread)


def test_random_baseline_is_deterministic_given_seed() -> None:
    scored = _synthetic_scored()
    r1 = add_random_baseline_column(scored, seed=101)
    r2 = add_random_baseline_column(scored, seed=101)
    assert (r1["_random_baseline"] == r2["_random_baseline"]).all()
