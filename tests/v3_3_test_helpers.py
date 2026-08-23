"""Shared test helpers for Phase V3-3 (tests/test_v3_3_*.py) - mirrors
tests/v3_2_test_helpers.py's pattern, but with enough synthetic history
(n_days=700 business days, ~2.7 years) for `v3.validation.windows.
get_v3_3_windows()`'s real 18/1/6-month WFO arithmetic to produce at
least 2 windows when given the synthetic data's own (start, end) dates
(never the real Phase V3-3 DATA_START/DATA_END - those are for the real
Full Universe run only).
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from storage.parquet_store import save_feature_panel, save_ohlcv
from v3.config.loader import V3Config, load_v3_config
from v3.dataset import build_v3_dataset
from v3.validation.windows import get_v3_3_windows

N_SYNTHETIC_DAYS = 700


def build_v3_3_config_and_tickers(
    tmp_path: Path, n_tickers: int = 8, n_days: int = N_SYNTHETIC_DAYS
) -> tuple[V3Config, list[str]]:
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    manifest_path = tmp_path / "manifest.json"

    market = make_synthetic_ohlcv(n_days, seed=999, ticker="TOPIX")
    save_ohlcv(market, "TOPIX", processed_dir)

    tickers = [f"T{i}" for i in range(n_tickers)]
    manifest_tickers = {}
    for i, ticker in enumerate(tickers):
        ohlcv = make_synthetic_ohlcv(n_days, seed=i + 1, ticker=ticker)
        save_ohlcv(ohlcv, ticker, processed_dir)
        # V2's run_v2_ranking() (used as V3-3's V2 Score benchmark) reads
        # a pre-computed V1 Feature panel, not raw OHLCV - saved here too
        # so the benchmark's code path is genuinely exercised, not skipped.
        panel = compute_feature_panel(ohlcv, market_df=market)
        save_feature_panel(panel, ticker, features_dir)
        manifest_tickers[ticker] = {"included_in_universe": True}
    manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

    config = load_v3_config().model_copy(
        update={
            "source_universe_manifest": manifest_path, "source_processed_dir": processed_dir,
            "source_features_dir": features_dir,
        }
    )
    return config, tickers


def build_small_dataset_and_windows(
    tmp_path: Path, n_tickers: int = 8, n_days: int = N_SYNTHETIC_DAYS
):
    config, tickers = build_v3_3_config_and_tickers(tmp_path, n_tickers, n_days)
    dataset = build_v3_dataset(tickers, config)
    windows = get_v3_3_windows(dataset["date"].min(), dataset["date"].max())
    return config, tickers, dataset, windows
