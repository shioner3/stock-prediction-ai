from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.causal.placebo import PLACEBO_SHIFTS, run_timing_placebo


def test_real_lag_zero_effect_does_not_appear_at_other_lags() -> None:
    """A single, non-periodic 3-day block (every ticker Q1, return=0.05)
    is the deliberate "real" signal. Elsewhere, a small random fraction of
    rows are independently labelled Q1 with near-zero noise return - a
    genuinely uninformative background. Since the block is only 3 days
    wide and every tested shift has |lag| >= 5, no shifted (placebo)
    lookup can ever land back inside the block, so placebo lags should
    only see the near-zero background rate, never the 0.05 block.
    """
    rng = np.random.default_rng(21)
    n_days = 80
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    tickers = [f"T{i}" for i in range(10)]
    block_days = {30, 31, 32}
    rows = []
    for d_idx, d in enumerate(dates):
        for ticker in tickers:
            if d_idx in block_days:
                bucket, ret = "Q1", 0.05
            elif rng.uniform() < 0.05:
                bucket, ret = "Q1", rng.normal(0, 0.0005)
            else:
                bucket, ret = "Q2", rng.normal(0, 0.0005)
            rows.append((ticker, d, bucket, ret))
    panel = pd.DataFrame(rows, columns=["ticker", "date", "score_bucket", "ret"])

    results = run_timing_placebo(panel, "ret", shifts=PLACEBO_SHIFTS)
    by_lag = {r.lag_days: r for r in results}
    real = by_lag[0]
    assert real.stats.mean_return is not None
    assert real.stats.mean_return > 0.01  # the deliberate 3-day block dominates at lag 0

    for lag in PLACEBO_SHIFTS:
        placebo = by_lag[lag]
        if placebo.stats.mean_return is not None:
            assert placebo.stats.mean_return < real.stats.mean_return / 2


def test_returns_one_result_per_shift_plus_lag_zero() -> None:
    rng = np.random.default_rng(1)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    rows = []
    for d in dates:
        for i in range(5):
            rows.append((f"T{i}", d, "Q1" if i == 0 else "Q2", rng.normal(0, 0.01)))
    panel = pd.DataFrame(rows, columns=["ticker", "date", "score_bucket", "ret"])
    results = run_timing_placebo(panel, "ret")
    assert {r.lag_days for r in results} == {0, *PLACEBO_SHIFTS}
