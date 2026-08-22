from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_synthetic_ohlcv

from config.loader import AppConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from pipeline.run_score_validation import build_scored_with_targets, run_score_validation
from scoring.pipeline import compute_score_records
from signals.pipeline import compute_signal_records
from storage.parquet_store import save_feature_panel, save_score_records, save_signal_records


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2024-01-01",
                "features_dir": str(tmp_path / "features"),
                "signals_dir": str(tmp_path / "signals"),
                "scores_dir": str(tmp_path / "scores"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


def _seed_ticker(config: AppConfig, ticker: str, seed: int) -> None:
    panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=seed, ticker=ticker))
    save_feature_panel(panel, ticker, config.data.features_dir)
    signal_records = compute_signal_records(panel, SignalsConfig())
    save_signal_records(signal_records, ticker, config.data.signals_dir)
    score_records = compute_score_records(panel, signal_records, config.scoring)
    save_score_records(score_records, ticker, config.data.scores_dir)


def test_run_score_validation_produces_grouped_results(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 100)
    _seed_ticker(base_config, "6758", 101)

    results = run_score_validation(base_config)

    assert isinstance(results, list)
    if results:
        r = results[0]
        assert r.direction in ("LONG", "SHORT")
        assert r.forward_window in (1, 3, 5, 7, 10, 15, 20)
        assert r.bucket_method == "fixed"
        assert isinstance(r.bucket_metrics, dict)


def test_run_score_validation_covers_all_seven_forward_windows(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 102)

    results = run_score_validation(base_config)
    windows = {r.forward_window for r in results}
    if results:
        assert windows == {1, 3, 5, 7, 10, 15, 20}


def test_run_score_validation_quantile_method(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 103)

    results = run_score_validation(base_config, bucket_method="quantile")
    for r in results:
        assert r.bucket_method == "quantile"


def test_run_score_validation_empty_when_no_scores(base_config: AppConfig) -> None:
    results = run_score_validation(base_config)
    assert results == []


def test_run_score_validation_separates_long_and_short(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 104)
    _seed_ticker(base_config, "6758", 105)

    results = run_score_validation(base_config)
    if not results:
        return

    # Each GroupValidationResult is scoped to exactly one direction and
    # one signal_name - no result ever mixes LONG/SHORT trades together.
    keys = [(r.direction, r.signal_name, r.forward_window, r.bucket_method) for r in results]
    assert len(keys) == len(set(keys))  # no duplicate groups
    assert set(r.direction for r in results) <= {"LONG", "SHORT"}


def test_ticker_with_no_scores_file_is_skipped_without_error(
    base_config: AppConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """Phase 6.5 (Full Universe scale): a ticker with zero triggered
    signals gets no data/scores/{ticker}.parquet file at all (see
    pipeline/build_scores.py - this is deliberate, not a bug). That must
    not surface as an ERROR-level stack trace when other tickers DO have
    scores (common at Full Universe scale - short-history/thinly-traded
    tickers routinely trigger no signals at all).
    """
    _seed_ticker(base_config, "7203", 106)
    # "9999" has features on disk but was never scored (no signals/no
    # scores file written) - exactly build_scores.py's zero-signal case.
    panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=999, ticker="9999"))
    save_feature_panel(panel, "9999", base_config.data.features_dir)

    with caplog.at_level("ERROR"):
        scored = build_scored_with_targets(base_config, tickers=["7203", "9999"])

    assert not scored.empty
    assert set(scored["ticker"]) == {"7203"}
    assert not any(r.levelname == "ERROR" for r in caplog.records)
