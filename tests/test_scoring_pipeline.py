from __future__ import annotations

import pandas as pd
from conftest import make_synthetic_ohlcv

from config.loader import ScoringConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from scoring.pipeline import SCORE_RECORD_COLUMNS, compute_score_records
from scoring.scorer import SUBSCORE_COLUMNS
from signals.pipeline import compute_signal_records


def _panel_and_signals(n: int = 400, seed: int = 1):
    panel = compute_feature_panel(make_synthetic_ohlcv(n, seed=seed))
    signal_records = compute_signal_records(panel, SignalsConfig())
    return panel, signal_records


def test_score_records_only_cover_triggered_rows() -> None:
    panel, signal_records = _panel_and_signals()
    scores = compute_score_records(panel, signal_records, ScoringConfig())

    assert len(scores) == len(signal_records)
    assert set(zip(scores["date"], scores["signal_name"])) == set(
        zip(signal_records["date"], signal_records["signal_name"])
    )


def test_score_records_have_expected_columns() -> None:
    panel, signal_records = _panel_and_signals()
    scores = compute_score_records(panel, signal_records, ScoringConfig())
    assert list(scores.columns) == SCORE_RECORD_COLUMNS


def test_score_records_total_equals_subscore_sum() -> None:
    panel, signal_records = _panel_and_signals()
    scores = compute_score_records(panel, signal_records, ScoringConfig())
    if scores.empty:
        return
    manual_sum = scores[SUBSCORE_COLUMNS].sum(axis=1)
    assert (scores["total_score"] == manual_sum).all()


def test_score_records_all_within_0_100() -> None:
    panel, signal_records = _panel_and_signals()
    scores = compute_score_records(panel, signal_records, ScoringConfig())
    if scores.empty:
        return
    assert (scores["total_score"] >= 0).all()
    assert (scores["total_score"] <= 100).all()


def test_empty_signal_records_gives_empty_scores() -> None:
    panel = compute_feature_panel(make_synthetic_ohlcv(10, seed=2))  # too short to trigger
    signal_records = compute_signal_records(panel, SignalsConfig())
    assert signal_records.empty
    scores = compute_score_records(panel, signal_records, ScoringConfig())
    assert scores.empty
    assert list(scores.columns) == SCORE_RECORD_COLUMNS


def test_score_records_are_deterministic() -> None:
    panel, signal_records = _panel_and_signals()
    scores_a = compute_score_records(panel, signal_records, ScoringConfig())
    scores_b = compute_score_records(panel, signal_records, ScoringConfig())
    pd.testing.assert_frame_equal(scores_a, scores_b)


def test_score_records_direction_matches_signal_direction() -> None:
    panel, signal_records = _panel_and_signals()
    scores = compute_score_records(panel, signal_records, ScoringConfig())
    if scores.empty:
        return
    lookup = signal_records.set_index(["date", "signal_name"])["direction"]
    for _, row in scores.iterrows():
        assert row["direction"] == lookup.loc[(row["date"], row["signal_name"])]
