from __future__ import annotations

from pathlib import Path

from v2_3_test_helpers import build_scored_panel_for_tests

from v2.causal.feature_stats import (
    FEATURE_LIST,
    compute_category_contribution,
    compute_category_correlation_matrix,
    compute_feature_bucket_profile,
    percentile_column,
    rank_categories_by_q1_deviation,
)
from v2.ranking.score import CATEGORY_FEATURES


def test_percentile_columns_added_for_every_feature(tmp_path: Path) -> None:
    _config, _tickers, _ranked, scored = build_scored_panel_for_tests(
        tmp_path, n_tickers=12, n_days=280
    )
    for _category, feature, _higher_is_better in FEATURE_LIST:
        assert percentile_column(feature) in scored.columns
        values = scored[percentile_column(feature)].dropna()
        assert (values > 0).all() and (values <= 1.0).all()


def test_feature_bucket_profile_covers_every_feature(tmp_path: Path) -> None:
    _config, _tickers, _ranked, scored = build_scored_panel_for_tests(
        tmp_path, n_tickers=12, n_days=280
    )
    profile = compute_feature_bucket_profile(scored, bucket_label="Q1")
    assert len(profile) == len(FEATURE_LIST)
    for p in profile:
        assert p.bucket == "Q1"
        assert p.n >= 0


def test_category_contribution_covers_every_category(tmp_path: Path) -> None:
    _config, _tickers, _ranked, scored = build_scored_panel_for_tests(
        tmp_path, n_tickers=12, n_days=280
    )
    contributions = compute_category_contribution(scored, bucket_label="Q1")
    assert {c.category for c in contributions} == set(CATEGORY_FEATURES)
    # Population mean of a category rank is a uniform-percentile average -
    # always close to 0.5, regardless of the synthetic data's specifics.
    for c in contributions:
        if c.population_mean_category_rank is not None:
            assert 0.3 < c.population_mean_category_rank < 0.7


def test_rank_categories_by_q1_deviation_sorts_ascending(tmp_path: Path) -> None:
    _config, _tickers, _ranked, scored = build_scored_panel_for_tests(
        tmp_path, n_tickers=12, n_days=280
    )
    contributions = compute_category_contribution(scored, bucket_label="Q1")
    ranked = rank_categories_by_q1_deviation(contributions)
    deviations = [
        c.deviation_from_population for c in ranked if c.deviation_from_population is not None
    ]
    assert deviations == sorted(deviations)


def test_category_correlation_matrix_is_square_and_symmetric(tmp_path: Path) -> None:
    _config, _tickers, _ranked, scored = build_scored_panel_for_tests(
        tmp_path, n_tickers=12, n_days=280
    )
    corr = compute_category_correlation_matrix(scored)
    assert corr.shape == (len(CATEGORY_FEATURES), len(CATEGORY_FEATURES))
    assert (abs(corr - corr.T) < 1e-9).all().all()
    # Diagonal is a perfect self-correlation.
    for category in CATEGORY_FEATURES:
        assert abs(corr.loc[f"{category}_rank", f"{category}_rank"] - 1.0) < 1e-9
