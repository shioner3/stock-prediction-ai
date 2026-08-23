"""v3/robustness/leave_one_out.py - Regime/Year/Day/Stock leave-one-out
robustness, verified against small constructed panels where the exclusion
effect is known by construction.
"""

from __future__ import annotations

import pandas as pd

from v3.robustness.leave_one_out import (
    run_day_concentration_robustness,
    run_regime_robustness,
    run_stock_concentration_robustness,
    run_year_robustness,
)


def _make_predictions(n_days: int = 40, n_tickers: int = 10, seed: int = 3) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for d in dates:
        preds = rng.normal(0, 1, n_tickers)
        actual = preds * 0.01 + rng.normal(0, 0.001, n_tickers)  # genuine, day-local signal
        for i in range(n_tickers):
            rows.append({
                "date": d.date(), "ticker": f"T{i}", "prediction": preds[i], "actual": actual[i],
            })
    return pd.DataFrame(rows)


def test_regime_leave_one_out_runs_and_all_regimes_present() -> None:
    predictions = _make_predictions()
    regime_dates = predictions["date"].drop_duplicates().tolist()
    cycle = ["BULL", "NEUTRAL", "BEAR"] * (len(regime_dates) // 3 + 1)
    regime_labels = cycle[: len(regime_dates)]
    regime_df = pd.DataFrame({"date": regime_dates, "regime": regime_labels})
    result = run_regime_robustness(predictions, regime_df, window_days=5)
    assert set(result.breakdown.keys()) == {"BULL", "NEUTRAL", "BEAR"}
    assert set(result.leave_one_out.keys()) == {"excl_BULL", "excl_NEUTRAL", "excl_BEAR"}
    # excluding one regime should strictly reduce N versus using all regimes
    assert result.leave_one_out["excl_BEAR"].n < result.all_regimes.n


def test_regime_dependent_flag_true_when_excluding_bear_kills_spread() -> None:
    # Construct so ALL of the positive signal lives only on BEAR days.
    dates = pd.bdate_range("2023-01-02", periods=30)
    rows = []
    for i, d in enumerate(dates):
        regime = "BEAR" if i % 3 == 0 else "BULL"
        for t in range(6):
            pred = t
            actual = 0.01 * t if regime == "BEAR" else 0.0  # zero signal outside BEAR
            rows.append({"date": d.date(), "ticker": f"T{t}", "prediction": pred, "actual": actual})
    predictions = pd.DataFrame(rows)
    regime_labels = ["BEAR" if i % 3 == 0 else "BULL" for i in range(30)]
    regime_df = pd.DataFrame({"date": [d.date() for d in dates], "regime": regime_labels})
    result = run_regime_robustness(predictions, regime_df, window_days=5)
    assert result.regime_dependent is True
    excl_bear_spread = result.leave_one_out["excl_BEAR"].ranking.q5_q1_spread
    assert excl_bear_spread is not None and excl_bear_spread <= 0


def test_year_leave_one_out_covers_years_present() -> None:
    dates_2023 = pd.bdate_range("2023-06-01", periods=10)
    dates_2024 = pd.bdate_range("2024-06-01", periods=10)
    rows = []
    for d in list(dates_2023) + list(dates_2024):
        for t in range(6):
            rows.append({"date": d.date(), "ticker": f"T{t}", "prediction": t, "actual": 0.01 * t})
    predictions = pd.DataFrame(rows)
    result = run_year_robustness(predictions, window_days=5)
    assert set(result.breakdown.keys()) == {2023, 2024}
    assert result.leave_one_out[2023].n < len(predictions)
    assert result.leave_one_out[2023].n == predictions[
        predictions["date"].apply(lambda d: d.year) != 2023
    ].shape[0]


def test_day_concentration_top_k_exclusion_removes_exactly_k_days() -> None:
    predictions = _make_predictions(n_days=25, n_tickers=8)
    result = run_day_concentration_robustness(predictions, window_days=5)
    n_days_total = predictions["date"].nunique()
    remaining_days = result.top_k_exclusion["top5"].n
    # exactly 5 fewer days' worth of rows should remain (8 tickers/day)
    assert remaining_days == (n_days_total - 5) * 8
    assert result.gini_day_contribution is not None
    assert 0.0 <= result.gini_day_contribution <= 1.0


def test_stock_concentration_top_k_exclusion_removes_ticker_everywhere() -> None:
    predictions = _make_predictions(n_days=20, n_tickers=8)
    result = run_stock_concentration_robustness(predictions, window_days=5)
    n_days_total = predictions["date"].nunique()
    remaining = result.top_k_exclusion["top1"].n
    # the single top-contributing ticker is removed from EVERY day, not
    # just from Q5 - exactly n_days_total fewer rows should remain.
    assert remaining == n_days_total * 7
