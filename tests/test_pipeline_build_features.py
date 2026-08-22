from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_synthetic_ohlcv

from config.loader import AppConfig
from pipeline.build_features import run_build_features
from storage.parquet_store import load_feature_panel, save_ohlcv


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2024-01-01",
                "processed_dir": str(tmp_path / "processed"),
                "features_dir": str(tmp_path / "features"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


def test_run_build_features_writes_feature_panels(base_config: AppConfig) -> None:
    processed_dir = Path(base_config.data.processed_dir)
    save_ohlcv(make_synthetic_ohlcv(300, seed=20, ticker="7203"), "7203", processed_dir)
    save_ohlcv(make_synthetic_ohlcv(300, seed=21, ticker="6758"), "6758", processed_dir)
    save_ohlcv(make_synthetic_ohlcv(300, seed=22, ticker="TOPIX"), "TOPIX", processed_dir)

    summary = run_build_features(base_config)

    # TOPIX is a benchmark, not a universe ticker - excluded from auto-discovery.
    assert summary.ticker_count == 2
    assert summary.success_count == 2
    assert summary.failed_tickers == []
    assert summary.market_data_available is True

    panel = load_feature_panel("7203", base_config.data.features_dir)
    assert "sma_20" in panel.columns
    assert "rsi_14" in panel.columns
    assert len(panel) == 300

    # TOPIX (seed=22) fully overlaps 7203's (seed=20) synthetic date
    # range, so rs_20d should be populated past its warmup, never
    # silently left all-NaN.
    assert panel["rs_20d"].iloc[20:].notna().all()


def test_run_build_features_without_topix_gives_nan_rs(base_config: AppConfig) -> None:
    processed_dir = Path(base_config.data.processed_dir)
    save_ohlcv(make_synthetic_ohlcv(300, seed=20, ticker="7203"), "7203", processed_dir)
    # No TOPIX.parquet saved - simulates providers/market_index.py's
    # topix_available=False case from pipeline/ingest.py.

    summary = run_build_features(base_config)

    assert summary.market_data_available is False
    assert summary.success_count == 1

    panel = load_feature_panel("7203", base_config.data.features_dir)
    assert panel["rs_5d"].isna().all()
    assert panel["rs_20d"].isna().all()
    assert panel["rs_60d"].isna().all()
    # Every other feature category is still computed normally.
    assert panel["sma_20"].iloc[20:].notna().all()


def test_run_build_features_skips_failing_ticker_without_aborting(
    base_config: AppConfig,
) -> None:
    processed_dir = Path(base_config.data.processed_dir)
    good = make_synthetic_ohlcv(300, seed=23, ticker="7203")
    bad = good.drop(columns=["volume"])  # will fail compute_feature_panel's column check
    save_ohlcv(good, "7203", processed_dir)
    save_ohlcv(bad, "9999", processed_dir)

    summary = run_build_features(base_config)

    assert summary.ticker_count == 2
    assert summary.success_count == 1
    assert summary.failed_tickers == ["9999"]


def test_run_build_features_respects_explicit_ticker_list(base_config: AppConfig) -> None:
    processed_dir = Path(base_config.data.processed_dir)
    save_ohlcv(make_synthetic_ohlcv(300, seed=24, ticker="7203"), "7203", processed_dir)
    save_ohlcv(make_synthetic_ohlcv(300, seed=25, ticker="6758"), "6758", processed_dir)

    summary = run_build_features(base_config, tickers=["7203"])

    assert summary.ticker_count == 1
    assert summary.success_count == 1
