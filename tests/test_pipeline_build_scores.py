from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import make_synthetic_ohlcv

from config.loader import AppConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from pipeline.build_scores import run_build_scores
from scoring.pipeline import SCORE_RECORD_COLUMNS
from signals.pipeline import compute_signal_records
from storage.parquet_store import load_score_records, save_feature_panel, save_signal_records


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


def test_run_build_scores_writes_records(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 90)
    _seed_ticker(base_config, "6758", 91)

    summary = run_build_scores(base_config)

    assert summary.ticker_count == 2
    assert summary.success_count == 2
    assert summary.failed_tickers == []

    scores = load_score_records("7203", base_config.data.scores_dir)
    assert list(scores.columns) == SCORE_RECORD_COLUMNS
    if not scores.empty:
        assert (scores["total_score"] >= 0).all()
        assert (scores["total_score"] <= 100).all()


def test_run_build_scores_handles_no_signals(base_config: AppConfig) -> None:
    panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=92, ticker="7203"))
    save_feature_panel(panel, "7203", base_config.data.features_dir)
    empty = pd.DataFrame(
        columns=["ticker", "date", "signal_name", "direction", "triggered", "signal_version"]
    )
    save_signal_records(empty, "7203", base_config.data.signals_dir)

    summary = run_build_scores(base_config)

    assert summary.success_count == 1
    assert summary.total_scores_computed == 0


def test_run_build_scores_skips_failing_ticker(base_config: AppConfig) -> None:
    good_panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=93, ticker="7203"))
    save_feature_panel(good_panel, "7203", base_config.data.features_dir)
    signal_records = compute_signal_records(good_panel, SignalsConfig())
    save_signal_records(signal_records, "7203", base_config.data.signals_dir)

    # A signals file that references a ticker with no matching Feature panel.
    save_signal_records(signal_records, "9999", base_config.data.signals_dir)

    summary = run_build_scores(base_config)

    assert summary.success_count == 1
    assert summary.failed_tickers == ["9999"]
