"""Shared test helpers for Phase V2-3 (tests/test_v2_3_*.py) - builds a
small synthetic Universe through the REAL V1/V2-1 pipeline (compute_feature_panel
-> V2-1's run_v2_ranking) rather than hand-authoring all 28 raw Feature
columns per test, mirroring tests/test_v2_2_orchestrator.py's own
`_build_config()` pattern exactly (kept as a separate module rather than
added to tests/conftest.py, so it stays scoped to V2-3's own test files).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from scoring.validation import assign_quantile_buckets
from storage.parquet_store import save_feature_panel, save_ohlcv
from v2.causal.feature_stats import compute_feature_percentiles
from v2.config.loader import V2Config, load_v2_config
from v2.pipeline import run_v2_ranking


def build_v2_3_config_and_tickers(
    tmp_path: Path, n_tickers: int = 25, n_days: int = 350
) -> tuple[V2Config, list[str]]:
    features_dir = tmp_path / "features"
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifest.json"

    market = make_synthetic_ohlcv(n_days, seed=999, ticker="TOPIX")
    save_ohlcv(market, "TOPIX", processed_dir)

    tickers = [f"T{i}" for i in range(n_tickers)]
    manifest_tickers = {}
    for i, ticker in enumerate(tickers):
        ohlcv = make_synthetic_ohlcv(n_days, seed=i + 1, ticker=ticker)
        panel = compute_feature_panel(ohlcv, market_df=market)
        save_feature_panel(panel, ticker, features_dir)
        manifest_tickers[ticker] = {"included_in_universe": True}
    manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

    config = load_v2_config().model_copy(
        update={
            "source_universe_manifest": manifest_path,
            "source_features_dir": features_dir,
            "source_processed_dir": processed_dir,
        }
    )
    return config, tickers


def build_scored_panel_for_tests(
    tmp_path: Path, n_tickers: int = 25, n_days: int = 350
) -> tuple[V2Config, list[str], pd.DataFrame, pd.DataFrame]:
    """Returns (config, tickers, ranked, scored) - `scored` already has
    every pct_<feature> column (v2/causal/feature_stats.py) and
    score_bucket assigned, matching v2/causal/orchestrator.py::
    build_scored_panel()'s own construction exactly.
    """
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers, n_days)
    ranked = run_v2_ranking(config, tickers=tickers)
    ranked = compute_feature_percentiles(ranked)
    scored = ranked.dropna(subset=["total_score"]).copy()
    scored["score_bucket"] = assign_quantile_buckets(scored["total_score"])
    return config, tickers, ranked, scored
