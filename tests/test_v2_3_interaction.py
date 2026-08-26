from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.causal.feature_stats import percentile_column
from v2.causal.interaction import (
    CATEGORY_REPRESENTATIVE_FEATURE,
    pairwise_feature_interaction,
    score_feature_crosstab,
)
from v2.ranking.score import CATEGORY_FEATURES


def _synthetic_panel(n_days: int = 30, n_tickers: int = 40, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        score = rng.uniform(0, 1, size=n_tickers)
        feat_a = rng.uniform(0, 1, size=n_tickers)
        feat_b = rng.uniform(0, 1, size=n_tickers)
        ret = rng.normal(0, 0.01, size=n_tickers)
        for i in range(n_tickers):
            rows.append((d, f"T{i}", score[i], feat_a[i], feat_b[i], ret[i]))
    df = pd.DataFrame(
        rows,
        columns=[
            "date", "ticker", "total_score",
            percentile_column("return_20d"), percentile_column("sma_20_slope"), "ret",
        ],
    )
    df["score_bucket"] = pd.qcut(df["total_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return df


def test_category_representative_features_cover_every_category() -> None:
    assert set(CATEGORY_REPRESENTATIVE_FEATURE.keys()) == set(CATEGORY_FEATURES.keys())
    for category, feature in CATEGORY_REPRESENTATIVE_FEATURE.items():
        member_columns = {col for col, _hib in CATEGORY_FEATURES[category]}
        assert feature in member_columns


def test_score_feature_crosstab_has_expected_cell_count() -> None:
    panel = _synthetic_panel()
    result = score_feature_crosstab(panel, "return_20d", "ret", window_days=5)
    assert len(result.cells) == 25  # 5 score buckets x 5 feature buckets
    assert sum(c.stats.n for c in result.cells) == len(panel)


def test_score_feature_crosstab_can_restrict_to_one_score_bucket() -> None:
    panel = _synthetic_panel()
    result = score_feature_crosstab(
        panel, "return_20d", "ret", window_days=5, score_buckets=("Q1",)
    )
    assert len(result.cells) == 5
    assert all(c.score_bucket == "Q1" for c in result.cells)


def test_pairwise_feature_interaction_has_four_cells() -> None:
    panel = _synthetic_panel()
    result = pairwise_feature_interaction(panel, "return_20d", "sma_20_slope", "ret", window_days=5)
    assert {c.label for c in result.cells} == {"low/low", "low/high", "high/low", "high/high"}
