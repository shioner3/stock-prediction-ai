"""Shared test helpers for Phase V3-2 (tests/test_v3_2_*.py) - builds a
small synthetic Universe through the REAL V1->V3 pipeline (mirrors
tests/test_v3_dataset.py's own `_build_config()` pattern exactly; kept
separate so V3-2's test files don't need to import from another test
module).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from conftest import make_synthetic_ohlcv

from storage.parquet_store import save_ohlcv
from v3.config.loader import V3Config, load_v3_config
from v3.dataset import build_v3_dataset, time_split


def build_v3_2_config_and_tickers(
    tmp_path: Path, n_tickers: int = 10, n_days: int = 500
) -> tuple[V3Config, list[str]]:
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifest.json"

    market = make_synthetic_ohlcv(n_days, seed=999, ticker="TOPIX")
    save_ohlcv(market, "TOPIX", processed_dir)

    tickers = [f"T{i}" for i in range(n_tickers)]
    manifest_tickers = {}
    for i, ticker in enumerate(tickers):
        ohlcv = make_synthetic_ohlcv(n_days, seed=i + 1, ticker=ticker)
        save_ohlcv(ohlcv, ticker, processed_dir)
        manifest_tickers[ticker] = {"included_in_universe": True}
    manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

    config = load_v3_config().model_copy(
        update={"source_universe_manifest": manifest_path, "source_processed_dir": processed_dir}
    )
    return config, tickers


def build_small_train_test(
    tmp_path: Path, n_tickers: int = 10, n_days: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic dataset large enough (n_days=500 trading days ~ 2 years)
    to have a meaningful train/test split with the embargo, split roughly
    2/3 train, 1/3 test with a 20-trading-day embargo (>= the largest
    Horizon in targets.registry.HORIZONS).
    """
    config, tickers = build_v3_2_config_and_tickers(tmp_path, n_tickers, n_days)
    dataset = build_v3_dataset(tickers, config)
    dates = sorted(dataset["date"].unique())
    split_idx = int(len(dates) * 0.7)
    train_end = dates[split_idx]
    test_start_idx = min(split_idx + 20, len(dates) - 1)
    test_start = dates[test_start_idx]
    train, test = time_split(dataset, train_end=train_end, test_start=test_start)
    return train, test


def build_small_dataset(tmp_path: Path, n_tickers: int = 10, n_days: int = 500) -> pd.DataFrame:
    config, tickers = build_v3_2_config_and_tickers(tmp_path, n_tickers, n_days)
    return build_v3_dataset(tickers, config)
