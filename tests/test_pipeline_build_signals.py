from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_synthetic_ohlcv

from config.loader import AppConfig
from features.pipeline import compute_feature_panel
from pipeline.build_signals import run_build_signals
from storage.parquet_store import load_signal_records, save_feature_panel


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2024-01-01",
                "features_dir": str(tmp_path / "features"),
                "signals_dir": str(tmp_path / "signals"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


def test_run_build_signals_writes_records(base_config: AppConfig) -> None:
    features_dir = Path(base_config.data.features_dir)
    for ticker, seed in [("7203", 70), ("6758", 71)]:
        panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=seed, ticker=ticker))
        save_feature_panel(panel, ticker, features_dir)

    summary = run_build_signals(base_config)

    assert summary.ticker_count == 2
    assert summary.success_count == 2
    assert summary.failed_tickers == []

    records = load_signal_records("7203", base_config.data.signals_dir)
    assert list(records.columns) == [
        "ticker", "date", "signal_name", "direction", "triggered", "signal_version",
    ]
    assert records["triggered"].all()
    assert (records["ticker"] == "7203").all()


def test_run_build_signals_skips_failing_ticker(base_config: AppConfig) -> None:
    features_dir = Path(base_config.data.features_dir)
    good = compute_feature_panel(make_synthetic_ohlcv(400, seed=72, ticker="7203"))
    save_feature_panel(good, "7203", features_dir)
    bad = good.drop(columns=["close"])  # will fail - signals need close
    save_feature_panel(bad, "9999", features_dir)

    summary = run_build_signals(base_config)

    assert summary.success_count == 1
    assert summary.failed_tickers == ["9999"]


def test_run_build_signals_respects_explicit_tickers(base_config: AppConfig) -> None:
    features_dir = Path(base_config.data.features_dir)
    for ticker, seed in [("7203", 73), ("6758", 74)]:
        panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=seed, ticker=ticker))
        save_feature_panel(panel, ticker, features_dir)

    summary = run_build_signals(base_config, tickers=["7203"])

    assert summary.ticker_count == 1
    assert summary.success_count == 1
