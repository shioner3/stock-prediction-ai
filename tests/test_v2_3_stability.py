from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.causal.stability import compute_daily_stability


def test_stability_perfectly_even_days_gives_zero_gini() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    rows = []
    for d in dates:
        for i in range(5):
            rows.append((d, f"T{i}", "Q1", -0.01))  # identical every day
    panel = pd.DataFrame(rows, columns=["date", "ticker", "score_bucket", "ret"])
    result = compute_daily_stability(panel, "ret", window_days=5)
    assert result.n_days == 10
    assert result.gini_of_below_average_days == 0.0
    assert result.positive_day_ratio == 0.0


def test_stability_flags_worst_and_best_day() -> None:
    rng = np.random.default_rng(4)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(20)]
    rows = []
    for d_idx, d in enumerate(dates):
        for i in range(5):
            ret = -0.5 if d_idx == 10 else rng.normal(0, 0.001)
            rows.append((d, f"T{i}", "Q1", ret))
    panel = pd.DataFrame(rows, columns=["date", "ticker", "score_bucket", "ret"])
    result = compute_daily_stability(panel, "ret", window_days=5)
    assert result.worst_day == dates[10]
    assert result.worst_day_return is not None and result.worst_day_return < -0.4


def test_stability_empty_bucket_returns_zero_days() -> None:
    panel = pd.DataFrame({"date": [], "ticker": [], "score_bucket": [], "ret": []})
    result = compute_daily_stability(panel, "ret", window_days=5)
    assert result.n_days == 0
    assert result.gini_of_below_average_days is None
