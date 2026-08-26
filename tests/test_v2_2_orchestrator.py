from __future__ import annotations

import json
from pathlib import Path

from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from storage.parquet_store import save_feature_panel, save_ohlcv
from v2.config.loader import load_v2_config
from v2.validation.orchestrator import PRIMARY_WINDOW_DAYS, run_v2_2_validation


def _build_config(tmp_path: Path, n_tickers: int = 25, n_days: int = 350):
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


def test_orchestrator_end_to_end_structure(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path)
    report = run_v2_2_validation(config, tickers=tickers)

    assert report.n_tickers == len(tickers)
    assert report.n_rows_scored <= report.n_rows_total
    assert report.primary.window_days == PRIMARY_WINDOW_DAYS

    # All 7 forward windows represented in the holding period comparison.
    assert set(report.holding_period_comparison.keys()) == {1, 3, 5, 7, 10, 15, 20}

    # Every FDR result's adjusted p-value is a valid probability.
    for fdr in report.fdr_results.values():
        assert 0.0 <= fdr.adjusted_p_value <= 1.0
        assert 0.0 <= fdr.raw_p_value <= 1.0

    # Bootstrap methods report a confidence level and resample count.
    assert report.primary.day_cluster_bootstrap.confidence_level == 0.95
    assert report.primary.block_bootstrap.block_length_days is not None

    # Concentration/regime/event/year sections all populated.
    assert len(report.primary.concentration.ticker_contribution) == 4
    assert len(report.primary.regime) == 3
    assert "full_period" in report.primary.event_exclusion
    assert len(report.primary.cost_sensitivity) == 4


def test_orchestrator_is_deterministic(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path, n_tickers=15, n_days=300)
    r1 = run_v2_2_validation(config, tickers=tickers)
    r2 = run_v2_2_validation(config, tickers=tickers)

    assert r1.primary.light.q5_q1_spread == r2.primary.light.q5_q1_spread
    assert r1.primary.day_cluster_bootstrap == r2.primary.day_cluster_bootstrap
    assert r1.primary.block_bootstrap == r2.primary.block_bootstrap
    assert [p.result.p_value for p in r1.primary.permutation] == [
        p.result.p_value for p in r2.primary.permutation
    ]
    assert r1.fdr_results.keys() == r2.fdr_results.keys()


def test_orchestrator_never_recomputes_ranking_per_window(tmp_path: Path, monkeypatch) -> None:
    """Spec section 28's efficiency requirement: the expensive ranking
    build (run_v2_ranking) must happen exactly ONCE per orchestrator
    call, never once per Holding Period/analysis axis.
    """
    config, tickers = _build_config(tmp_path, n_tickers=10, n_days=200)

    import v2.validation.orchestrator as orch_module

    call_count = {"n": 0}
    original = orch_module.run_v2_ranking

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(orch_module, "run_v2_ranking", counting_wrapper)
    run_v2_2_validation(config, tickers=tickers)
    assert call_count["n"] == 1
