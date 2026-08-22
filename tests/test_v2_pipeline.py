from __future__ import annotations

import json
from pathlib import Path

from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from scoring.validation import assign_quantile_buckets
from storage.parquet_store import save_feature_panel, save_ohlcv
from v2.config.loader import load_v2_config
from v2.pipeline import (
    build_ticker_panel,
    build_universe_panel,
    candidate_table_for_date,
    load_universe_tickers,
    run_v2_ranking,
)
from v2.stats import compute_q5_q1_spread, compute_quantile_bucket_stats


def _seed_universe(tmp_path: Path, n_tickers: int = 12, n_days: int = 250):
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
    # One extra ticker registered in the manifest but with NO cached
    # Feature panel - must be silently skipped, never raise.
    manifest_tickers["MISSING"] = {"included_in_universe": True}
    manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

    config = load_v2_config().model_copy(
        update={
            "source_universe_manifest": manifest_path,
            "source_features_dir": features_dir,
            "source_processed_dir": processed_dir,
        }
    )
    return config, tickers


def test_load_universe_tickers_filters_included_only(tmp_path: Path) -> None:
    config, tickers = _seed_universe(tmp_path)
    loaded = load_universe_tickers(config)
    assert set(loaded) == set(tickers) | {"MISSING"}


def test_build_ticker_panel_returns_none_for_missing_ticker(tmp_path: Path) -> None:
    config, _ = _seed_universe(tmp_path)
    assert build_ticker_panel("MISSING", config) is None


def test_build_universe_panel_skips_missing_ticker(tmp_path: Path) -> None:
    config, tickers = _seed_universe(tmp_path)
    panel = build_universe_panel(tickers + ["MISSING"], config)
    assert "MISSING" not in panel["ticker"].unique()
    assert set(panel["ticker"].unique()) == set(tickers)


def test_run_v2_ranking_with_no_explicit_tickers_uses_manifest(tmp_path: Path) -> None:
    config, tickers = _seed_universe(tmp_path)
    ranked = run_v2_ranking(config)
    assert set(ranked["ticker"].unique()) == set(tickers)


def test_candidate_table_for_nonexistent_date_is_empty(tmp_path: Path) -> None:
    config, tickers = _seed_universe(tmp_path)
    ranked = run_v2_ranking(config, tickers=tickers)
    from datetime import date

    records = candidate_table_for_date(ranked, date(1999, 1, 1))
    assert records == []


def test_q1_q5_research_stats_reuses_v1_quantile_buckets(tmp_path: Path) -> None:
    """Integration check: V2's own Score Q1-Q5 analysis (spec section
    10) is built from scoring.validation.assign_quantile_buckets()
    (unmodified V1 code) + v2/stats.py's own ReturnStats - exercised
    end-to-end against real ranked V2 output, not just unit-level.
    """
    config, tickers = _seed_universe(tmp_path, n_tickers=20, n_days=300)
    ranked = run_v2_ranking(config, tickers=tickers)

    scored = ranked.dropna(subset=["total_score", "forward_return_5d"]).copy()
    assert not scored.empty
    scored["bucket"] = assign_quantile_buckets(scored["total_score"])

    results = compute_quantile_bucket_stats(scored, "bucket", "forward_return_5d", 5)
    assert len(results) > 0
    for r in results:
        assert r.bucket.startswith("Q")
        assert r.stats.n > 0

    spread = compute_q5_q1_spread(results)
    # May legitimately be None if Q1/Q5 didn't both appear (small
    # synthetic sample) - just confirm the call doesn't raise and, when
    # present, is a finite float.
    if spread is not None:
        assert spread == spread  # not NaN
