"""Spec section 19 Determinism: same data + same config, run twice,
must produce identical results (no unseeded randomness anywhere in the
V2 pipeline - it is rule-based/statistical, never ML, so this should
hold trivially, but is verified directly rather than assumed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from storage.parquet_store import save_feature_panel, save_ohlcv
from v2.config.loader import load_v2_config
from v2.pipeline import candidate_table_for_date, run_v2_ranking


def _build_config(tmp_path: Path):
    features_dir = tmp_path / "features"
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifest.json"

    market = make_synthetic_ohlcv(200, seed=999, ticker="TOPIX")
    save_ohlcv(market, "TOPIX", processed_dir)

    tickers = [f"T{i}" for i in range(12)]
    manifest_tickers = {}
    for i, ticker in enumerate(tickers):
        ohlcv = make_synthetic_ohlcv(200, seed=i + 1, ticker=ticker)
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


def test_ranking_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path)

    run1 = run_v2_ranking(config, tickers=tickers)
    run2 = run_v2_ranking(config, tickers=tickers)

    pd.testing.assert_frame_equal(
        run1.sort_values(["date", "ticker"]).reset_index(drop=True),
        run2.sort_values(["date", "ticker"]).reset_index(drop=True),
    )


def test_candidate_table_is_deterministic(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path)
    ranked = run_v2_ranking(config, tickers=tickers)
    last_date = ranked["date"].max()

    records1 = candidate_table_for_date(ranked, last_date)
    records2 = candidate_table_for_date(ranked, last_date)

    assert [(r.ticker, r.rank, r.score) for r in records1] == [
        (r.ticker, r.rank, r.score) for r in records2
    ]
