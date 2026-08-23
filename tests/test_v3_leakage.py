"""Phase V3-1 Leakage Framework tests (spec section 23/36):

- Mechanical AVAILABLE_AT<=t check (v3/leakage/availability_check.py) on
  the REAL v3/features/*.py source.
- Four future-shock tests (A price / B index / C volume / D random
  perturbation, spec section 23 A-D): mutate rows AFTER a cutoff date in
  the raw OHLCV, rebuild the full V3 dataset, and assert every CORE
  Feature column at date <= cutoff is unchanged. Only Feature columns are
  checked - Target columns are EXPECTED to change post-shock (they are
  the future-looking side of the boundary by design, spec section 2).
- A static import-direction check mirroring tests/test_v2_leakage.py's
  own pattern: v3/ must never import V1's decision-making layers
  (signals/scoring/backtest/forward_test/ensemble).
"""

from __future__ import annotations

import ast
import json
from datetime import date as date_type
from pathlib import Path

import numpy as np
from conftest import make_synthetic_ohlcv

from storage.parquet_store import load_ohlcv, save_ohlcv
from v3.config.loader import load_v3_config
from v3.dataset import build_v3_dataset
from v3.features.registry import CORE_FEATURE_NAMES
from v3.leakage.availability_check import check_v3_features_no_forward_reads
from v3.leakage.shock_tests import (
    random_perturb_after,
    shock_index_after,
    shock_price_after,
    shock_volume_after,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_DECISION_DIRS = ["signals", "scoring", "backtest", "forward_test", "ensemble"]
CUTOFF = date_type(2024, 6, 1)


def test_availability_check_finds_nothing_in_real_v3_features() -> None:
    findings = check_v3_features_no_forward_reads()
    assert findings == []


def test_v3_never_imports_v1_decision_layers() -> None:
    offending = []
    for path in (REPO_ROOT / "v3").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == d or name.startswith(f"{d}.") for d in V1_DECISION_DIRS):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")
    assert not offending, offending


def _build_config(tmp_path: Path, n_tickers: int = 6, n_days: int = 400):
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
    return config, tickers, processed_dir


def _assert_features_unchanged_at_and_before_cutoff(baseline, shocked, cutoff) -> None:
    baseline_early = (
        baseline[baseline["date"] <= cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    shocked_early = (
        shocked[shocked["date"] <= cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    assert list(baseline_early["ticker"]) == list(shocked_early["ticker"])
    assert list(baseline_early["date"]) == list(shocked_early["date"])
    for col in CORE_FEATURE_NAMES:
        b = baseline_early[col].to_numpy(dtype=float)
        s = shocked_early[col].to_numpy(dtype=float)
        mismatch = ~((b == s) | (np.isnan(b) & np.isnan(s)))
        assert not mismatch.any(), f"LEAKAGE_FOUND in {col}: {mismatch.sum()} rows changed"


def test_future_price_shock_never_changes_earlier_rows(tmp_path: Path) -> None:
    config, tickers, processed_dir = _build_config(tmp_path)
    baseline = build_v3_dataset(tickers, config)

    shocked_dir = tmp_path / "processed_price_shocked"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, processed_dir)
        shocked = shock_price_after(ohlcv, CUTOFF) if ticker != "TOPIX" else ohlcv
        save_ohlcv(shocked, ticker, shocked_dir)
    shocked_config = config.model_copy(update={"source_processed_dir": shocked_dir})
    shocked_dataset = build_v3_dataset(tickers, shocked_config)

    _assert_features_unchanged_at_and_before_cutoff(baseline, shocked_dataset, CUTOFF)


def test_future_index_shock_never_changes_earlier_rows(tmp_path: Path) -> None:
    config, tickers, processed_dir = _build_config(tmp_path)
    baseline = build_v3_dataset(tickers, config)

    shocked_dir = tmp_path / "processed_index_shocked"
    for ticker in tickers:
        save_ohlcv(load_ohlcv(ticker, processed_dir), ticker, shocked_dir)
    topix = load_ohlcv("TOPIX", processed_dir)
    save_ohlcv(shock_index_after(topix, CUTOFF), "TOPIX", shocked_dir)
    shocked_config = config.model_copy(update={"source_processed_dir": shocked_dir})
    shocked_dataset = build_v3_dataset(tickers, shocked_config)

    _assert_features_unchanged_at_and_before_cutoff(baseline, shocked_dataset, CUTOFF)


def test_future_volume_shock_never_changes_earlier_rows(tmp_path: Path) -> None:
    config, tickers, processed_dir = _build_config(tmp_path)
    baseline = build_v3_dataset(tickers, config)

    shocked_dir = tmp_path / "processed_volume_shocked"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, processed_dir)
        shocked = shock_volume_after(ohlcv, CUTOFF) if ticker != "TOPIX" else ohlcv
        save_ohlcv(shocked, ticker, shocked_dir)
    shocked_config = config.model_copy(update={"source_processed_dir": shocked_dir})
    shocked_dataset = build_v3_dataset(tickers, shocked_config)

    _assert_features_unchanged_at_and_before_cutoff(baseline, shocked_dataset, CUTOFF)


def test_future_random_perturbation_never_changes_earlier_rows(tmp_path: Path) -> None:
    config, tickers, processed_dir = _build_config(tmp_path)
    baseline = build_v3_dataset(tickers, config)

    shocked_dir = tmp_path / "processed_random_shocked"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, processed_dir)
        shocked = (
            random_perturb_after(ohlcv, CUTOFF, seed=7) if ticker != "TOPIX" else ohlcv
        )
        save_ohlcv(shocked, ticker, shocked_dir)
    shocked_config = config.model_copy(update={"source_processed_dir": shocked_dir})
    shocked_dataset = build_v3_dataset(tickers, shocked_config)

    _assert_features_unchanged_at_and_before_cutoff(baseline, shocked_dataset, CUTOFF)
