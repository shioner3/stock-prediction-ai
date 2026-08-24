"""Phase V3-5 orchestrator integration smoke test - end-to-end on a
small synthetic Universe (mirrors `tests/test_v3_4_orchestrator.py`'s
pattern), verifying `run_v3_5_analysis()` executes without error and
produces a structurally sane `V3_5Report` across all 16 Target x Horizon
combinations. Not a check on the NUMBERS - synthetic data has no real
market-neutral signal to detect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from v3_3_test_helpers import build_small_dataset_and_windows

from config.loader import TransactionCostConfig
from storage.parquet_store import load_ohlcv
from v3.residual.orchestrator import HORIZONS, run_v3_5_analysis
from v3.residual.reproduce import (
    TARGET_A_RAW,
    TARGET_DEFINITIONS,
    build_augmented_dataset,
    reproduce_residual_predictions,
)
from v3.robustness.reproduce import reproduce_predictions
from v3.validation.wfo_config import MARKET_REGIME_CONFIG


def test_v3_5_analysis_runs_end_to_end(tmp_path: Path) -> None:
    config, tickers, dataset, windows = build_small_dataset_and_windows(
        tmp_path, n_tickers=10, n_days=700
    )
    assert len(windows) >= 2

    # The real JPX master cache has no entries for synthetic "T{i}"
    # tickers (attach_sector_and_scale would return all-NaN sector33,
    # which - unlike V3-4's own evaluation-only use of sector_map -
    # would make target_sector_relative_*d an all-NaN TRAINING target
    # here and crash LightGBM on an empty training set). A hand-built
    # synthetic sector_map with real (non-NaN) group membership properly
    # exercises compute_residual_targets()'s sector-mean logic instead.
    sector_map = pd.DataFrame({
        "ticker": tickers, "sector33": ["Tech", "Bank"] * (len(tickers) // 2),
    })
    augmented_dataset = build_augmented_dataset(dataset, sector_map)

    raw_predictions = reproduce_predictions(dataset, windows)
    predictions_by_combo = {
        (TARGET_A_RAW, horizon): raw_predictions[f"target_raw_{horizon}d"] for horizon in HORIZONS
    }
    new_predictions = reproduce_residual_predictions(augmented_dataset, windows, HORIZONS)
    predictions_by_combo.update(new_predictions)

    assert set(predictions_by_combo.keys()) == {
        (d, h) for d in TARGET_DEFINITIONS for h in HORIZONS
    }

    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    cost_tiers = TransactionCostConfig().tiers

    report = run_v3_5_analysis(
        augmented_dataset=augmented_dataset, predictions_by_combo=predictions_by_combo,
        tickers=tickers, topix_ohlcv=topix, market_regime_config=MARKET_REGIME_CONFIG,
        cost_tiers=cost_tiers, v3_config=config,
    )

    assert len(report.light_results) == 16
    assert set(report.primary_results.keys()) == set(TARGET_DEFINITIONS)
    assert len(report.market_neutralization_table) == 4
    assert len(report.residual_strength_by_horizon) == 12  # 3 definitions x 4 horizons
    assert report.edge_classification is not None
    assert report.edge_classification.classification is not None
