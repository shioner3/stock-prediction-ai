from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.causal.heterogeneity import analyze_q1_heterogeneity


def test_heterogeneity_splits_q1_into_five_sub_buckets() -> None:
    rng = np.random.default_rng(11)
    n_days, n_tickers = 40, 50
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        scores = rng.uniform(0, 1, size=n_tickers)
        ret = rng.normal(0, 0.01, size=n_tickers)
        for i in range(n_tickers):
            rows.append((d, f"T{i}", scores[i], ret[i]))
    panel = pd.DataFrame(rows, columns=["date", "ticker", "total_score", "ret"])
    panel["score_bucket"] = pd.qcut(
        panel["total_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    result = analyze_q1_heterogeneity(panel, "ret", window_days=5)
    labels = [b.bucket for b in result.sub_bucket_stats]
    assert labels == ["Q1-a", "Q1-b", "Q1-c", "Q1-d", "Q1-e"]
    total_n = sum(b.stats.n for b in result.sub_bucket_stats)
    assert total_n == (panel["score_bucket"] == "Q1").sum()


def test_heterogeneity_detects_worsening_toward_bottom() -> None:
    rng = np.random.default_rng(13)
    n_days, n_tickers = 40, 50
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        scores = rng.uniform(0, 1, size=n_tickers)
        # Within the whole population, lower score -> lower return (so
        # the LOWEST sub-bucket of Q1 should have the worst mean).
        ret = scores * 0.02 + rng.normal(0, 0.0005, size=n_tickers)
        for i in range(n_tickers):
            rows.append((d, f"T{i}", scores[i], ret[i]))
    panel = pd.DataFrame(rows, columns=["date", "ticker", "total_score", "ret"])
    panel["score_bucket"] = pd.qcut(
        panel["total_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    result = analyze_q1_heterogeneity(panel, "ret", window_days=5)
    by_label = {b.bucket: b.stats.mean_return for b in result.sub_bucket_stats}
    assert by_label["Q1-a"] < by_label["Q1-e"]
