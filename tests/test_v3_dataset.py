from __future__ import annotations

import json
from pathlib import Path

from conftest import make_synthetic_ohlcv

from storage.parquet_store import save_ohlcv
from v3.config.loader import load_v3_config
from v3.dataset import build_v3_dataset, load_universe_tickers
from v3.features.registry import CORE_FEATURE_NAMES
from v3.targets.registry import TARGET_COLUMN_NAMES


def _build_config(tmp_path: Path, n_tickers: int = 8, n_days: int = 250):
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


def test_load_universe_tickers_reads_manifest(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path, n_tickers=5)
    assert load_universe_tickers(config) == sorted(tickers)


def test_build_v3_dataset_has_exactly_the_registered_columns(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path, n_tickers=6, n_days=250)
    dataset = build_v3_dataset(tickers, config)
    assert not dataset.empty
    expected = {"date", "ticker", *CORE_FEATURE_NAMES, *TARGET_COLUMN_NAMES}
    assert set(dataset.columns) == expected
    assert set(dataset["ticker"].unique()) == set(tickers)


def test_build_v3_dataset_is_deterministic(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path, n_tickers=5, n_days=220)
    d1 = build_v3_dataset(tickers, config)
    d2 = build_v3_dataset(tickers, config)
    assert d1.equals(d2)


def test_build_v3_dataset_excludes_missing_ticker(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path, n_tickers=4, n_days=220)
    dataset = build_v3_dataset([*tickers, "NOT_A_REAL_TICKER"], config)
    assert "NOT_A_REAL_TICKER" not in dataset["ticker"].unique()
