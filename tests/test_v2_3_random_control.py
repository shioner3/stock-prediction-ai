from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.causal.random_control import (
    RANDOM_CONTROL_SEEDS,
    build_random_control_sample,
    run_random_control,
)


def _synthetic_clean(n_days: int = 30, n_tickers: int = 20, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        for i in range(n_tickers):
            bucket = "Q1" if i < 4 else ("Q5" if i >= n_tickers - 4 else "Q2")
            rows.append((d, f"T{i}", bucket, rng.normal(0, 0.01)))
    return pd.DataFrame(rows, columns=["date", "ticker", "score_bucket", "ret"])


def test_random_control_sample_matches_q1_size_per_day() -> None:
    clean = _synthetic_clean()
    sample = build_random_control_sample(clean, seed=RANDOM_CONTROL_SEEDS[0])
    q1_counts = clean[clean["score_bucket"] == "Q1"].groupby("date").size()
    sample_counts = sample.groupby("date").size()
    assert (sample_counts == q1_counts).all()


def test_random_control_is_deterministic_given_seed() -> None:
    clean = _synthetic_clean()
    s1 = build_random_control_sample(clean, seed=201)
    s2 = build_random_control_sample(clean, seed=201)
    assert list(s1["ticker"]) == list(s2["ticker"])
    assert list(s1["date"]) == list(s2["date"])


def test_run_random_control_returns_one_result_per_seed() -> None:
    clean = _synthetic_clean()
    result = run_random_control(clean, "ret", window_days=5)
    assert {r.seed for r in result.per_seed} == set(RANDOM_CONTROL_SEEDS)
    assert result.pooled_stats.n == sum(r.n for r in result.per_seed)
