from __future__ import annotations

from pathlib import Path

from v3_3_test_helpers import build_small_dataset_and_windows

from storage.parquet_store import load_ohlcv
from v2.config.loader import load_v2_config
from v3.validation.orchestrator import run_primary_analysis, run_v3_3_validation
from v3.validation.wfo_config import MARKET_REGIME_CONFIG


def test_run_primary_analysis_end_to_end_structure(tmp_path: Path) -> None:
    config, _tickers, dataset, windows = build_small_dataset_and_windows(tmp_path)
    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    result = run_primary_analysis(dataset, windows, topix, MARKET_REGIME_CONFIG)

    assert len(result.per_window) == len(windows)
    assert len(result.regime) == 3
    assert "full_period" in result.event
    assert result.spread_bootstrap.trade_level is not None
    assert len(result.permutation) == 2  # Q1, Q5
    assert result.window_stability.n_slices <= len(windows)


def test_run_v3_3_validation_end_to_end_structure(tmp_path: Path) -> None:
    config, tickers, dataset, windows = build_small_dataset_and_windows(tmp_path, n_tickers=6)
    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    v2_config = load_v2_config().model_copy(
        update={
            "source_universe_manifest": config.source_universe_manifest,
            "source_processed_dir": config.source_processed_dir,
            "source_features_dir": config.source_features_dir,
        }
    )
    report = run_v3_3_validation(
        dataset, tickers, v2_config, topix, MARKET_REGIME_CONFIG, windows=windows
    )

    assert report.n_tickers == len(tickers)
    assert len(report.windows) == len(windows)
    assert set(report.secondary.keys()) == {
        "target_raw_10d", "target_raw_15d", "target_raw_20d", "target_topix_relative_5d",
        "target_vol_adjusted_5d", "target_risk_adjusted_5d", "model_b", "model_c",
    }
    assert set(report.benchmarks.keys()) == {"random", "momentum", "v2_score"}
    for fdr in report.fdr_results.values():
        assert 0.0 <= fdr.adjusted_p_value <= 1.0
