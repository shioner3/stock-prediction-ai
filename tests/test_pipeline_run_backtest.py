from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import make_synthetic_ohlcv

from config.loader import AppConfig
from features.pipeline import compute_feature_panel
from pipeline.run_backtest import run_backtest
from signals.pipeline import compute_signal_records
from storage.parquet_store import save_feature_panel, save_signal_records


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


def _seed_ticker(config: AppConfig, ticker: str, seed: int) -> None:
    panel = compute_feature_panel(make_synthetic_ohlcv(400, seed=seed, ticker=ticker))
    save_feature_panel(panel, ticker, config.data.features_dir)
    records = compute_signal_records(panel, config.signals)
    save_signal_records(records, ticker, config.data.signals_dir)


def test_run_backtest_produces_trades_across_tickers(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 80)
    _seed_ticker(base_config, "6758", 81)

    summary = run_backtest(base_config)

    assert summary.ticker_count == 2
    assert summary.success_count == 2
    assert summary.failed_tickers == []
    assert list(summary.trades.columns) == [
        "ticker", "signal_name", "direction", "signal_date",
        "entry_date", "exit_date", "entry_price", "exit_price", "return",
    ]
    if not summary.trades.empty:
        assert set(summary.trades["ticker"]) <= {"7203", "6758"}


def test_run_backtest_handles_ticker_with_no_signals(base_config: AppConfig) -> None:
    # A ticker with a signals file that has zero rows (nothing triggered).
    empty = pd.DataFrame(
        columns=["ticker", "date", "signal_name", "direction", "triggered", "signal_version"]
    )
    save_signal_records(empty, "0000", base_config.data.signals_dir)

    summary = run_backtest(base_config, tickers=["0000"])

    assert summary.success_count == 1
    assert summary.trades.empty


def test_run_backtest_respects_explicit_tickers(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 82)
    _seed_ticker(base_config, "6758", 83)

    summary = run_backtest(base_config, tickers=["7203"])

    assert summary.ticker_count == 1
    if not summary.trades.empty:
        assert set(summary.trades["ticker"]) == {"7203"}
