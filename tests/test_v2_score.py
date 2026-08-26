from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v2.config.loader import V2ScoreWeightsConfig
from v2.ranking.score import CATEGORY_FEATURES, compute_category_ranks, compute_v2_score


def _synthetic_panel() -> pd.DataFrame:
    """3 tickers, 1 date, every CATEGORY_FEATURES column populated with
    distinct values so ranking is unambiguous.
    """
    tickers = ["A", "B", "C"]
    data = {"date": ["D1"] * 3, "ticker": tickers}
    value = 1.0
    for members in CATEGORY_FEATURES.values():
        for column, _ in members:
            data[column] = [value, value + 1, value + 2]
            value += 10
    return pd.DataFrame(data)


def test_category_rank_columns_present() -> None:
    panel = _synthetic_panel()
    ranks = compute_category_ranks(panel)
    for category in CATEGORY_FEATURES:
        assert f"{category}_rank" in ranks.columns


def test_category_ranks_bounded() -> None:
    panel = _synthetic_panel()
    ranks = compute_category_ranks(panel)
    for category in CATEGORY_FEATURES:
        col = ranks[f"{category}_rank"]
        assert (col.dropna() > 0).all()
        assert (col.dropna() <= 1.0).all()


def test_score_weights_sum_to_one_in_default_config() -> None:
    weights = V2ScoreWeightsConfig()
    total = (
        weights.momentum + weights.trend + weights.volume
        + weights.relative_strength + weights.pullback + weights.volatility
    )
    assert total == pytest.approx(1.0)


def test_score_is_weighted_sum_of_category_ranks() -> None:
    panel = _synthetic_panel()
    ranks = compute_category_ranks(panel)
    weights = V2ScoreWeightsConfig()
    score = compute_v2_score(ranks, weights)

    expected = (
        ranks["momentum_rank"] * weights.momentum
        + ranks["trend_rank"] * weights.trend
        + ranks["volume_rank"] * weights.volume
        + ranks["volatility_rank"] * weights.volatility
        + ranks["relative_strength_rank"] * weights.relative_strength
        + ranks["pullback_rank"] * weights.pullback
    )
    assert np.allclose(score.to_numpy(), expected.to_numpy())


def test_score_is_nan_when_any_category_is_entirely_nan() -> None:
    panel = _synthetic_panel()
    # Blank out every momentum feature for every row -> momentum_rank
    # must be all-NaN, and total_score must be NaN for every row (never
    # imputed/renormalized - see v2/ranking/score.py's module docstring).
    for column, _ in CATEGORY_FEATURES["momentum"]:
        panel[column] = np.nan

    ranks = compute_category_ranks(panel)
    assert ranks["momentum_rank"].isna().all()

    score = compute_v2_score(ranks, V2ScoreWeightsConfig())
    assert score.isna().all()


def test_higher_raw_momentum_gives_higher_score_all_else_equal() -> None:
    tickers = ["A", "B"]
    data = {"date": ["D1"] * 2, "ticker": tickers}
    for category, members in CATEGORY_FEATURES.items():
        for column, higher_is_better in members:
            if category == "momentum":
                continue
            data[column] = [1.0, 1.0]  # identical elsewhere
    for column, higher_is_better in CATEGORY_FEATURES["momentum"]:
        data[column] = [1.0, 2.0]  # B strictly higher momentum
    panel = pd.DataFrame(data)

    ranks = compute_category_ranks(panel)
    score = compute_v2_score(ranks, V2ScoreWeightsConfig())
    assert score.iloc[1] > score.iloc[0]
