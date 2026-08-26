from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from v2.causal.segment import (
    JPX_MASTER_CACHE_PATH,
    attach_liquidity_columns,
    attach_segment_columns,
    compute_group_stats,
    compute_liquidity_profile,
    load_ticker_segment_map,
)


def _synthetic_scored(n_days: int = 20, n_tickers: int = 10, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append(
                (
                    d, f"T{i}",
                    "Q1" if i < n_tickers // 2 else "Q5",
                    rng.normal(0, 0.01),
                    100.0 + i, 1000.0 * (i + 1),
                )
            )
    return pd.DataFrame(
        rows, columns=["date", "ticker", "score_bucket", "ret", "close", "volume"]
    )


def test_load_ticker_segment_map_reads_local_cache() -> None:
    assert Path(JPX_MASTER_CACHE_PATH).exists()
    segment_map = load_ticker_segment_map()
    assert {"code", "market_segment", "sector33", "scale"} <= set(segment_map.columns)
    assert len(segment_map) > 1000


def test_attach_segment_columns_merges_by_ticker() -> None:
    scored = _synthetic_scored()
    segment_map = pd.DataFrame(
        {
            "code": [f"T{i}" for i in range(10)],
            "market_segment": ["Prime"] * 5 + ["Standard"] * 5,
            "sector33": ["銀行"] * 10,
            "scale": ["TOPIX Core30"] * 10,
        }
    )
    merged = attach_segment_columns(scored, segment_map)
    assert "market_segment" in merged.columns
    assert set(merged["market_segment"].dropna().unique()) == {"Prime", "Standard"}


def test_compute_group_stats_restricted_to_one_bucket() -> None:
    scored = _synthetic_scored()
    segment_map = pd.DataFrame(
        {
            "code": [f"T{i}" for i in range(10)],
            "market_segment": ["Prime"] * 5 + ["Standard"] * 5,
            "sector33": ["銀行"] * 10,
            "scale": ["TOPIX Core30"] * 10,
        }
    )
    merged = attach_segment_columns(scored, segment_map)
    results = compute_group_stats(merged, "market_segment", "ret", bucket_label="Q1")
    assert {r.group_value for r in results} <= {"Prime", "Standard"}
    total_n = sum(r.n for r in results)
    assert total_n == (merged["score_bucket"] == "Q1").sum()


def test_liquidity_profile_uses_close_times_volume() -> None:
    scored = _synthetic_scored()
    with_liquidity = attach_liquidity_columns(scored)
    profile_q1 = compute_liquidity_profile(with_liquidity, bucket_label="Q1")
    profile_q5 = compute_liquidity_profile(with_liquidity, bucket_label="Q5")
    assert profile_q1.turnover_mean is not None
    assert profile_q5.turnover_mean is not None
    # Q5 tickers (higher index) have deliberately higher price/volume in
    # the synthetic fixture, so should show higher turnover.
    assert profile_q5.turnover_mean > profile_q1.turnover_mean
