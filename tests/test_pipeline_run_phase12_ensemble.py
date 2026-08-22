from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

import pytest
from conftest import make_synthetic_ohlcv

from common.hashing import hash_files
from config.loader import AppConfig, ScoringConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from forward_test.manifest import build_manifest
from pipeline.run_phase8_analysis import ConfigMismatchError
from pipeline.run_phase12_ensemble import run_phase12_ensemble
from pipeline.run_walk_forward import CONFIG_FILES
from scoring.pipeline import compute_score_records
from signals.pipeline import compute_signal_records
from storage.parquet_store import (
    save_feature_panel,
    save_ohlcv,
    save_score_records,
    save_signal_records,
)


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2020-01-01",
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "features_dir": str(tmp_path / "features"),
                "signals_dir": str(tmp_path / "signals"),
                "scores_dir": str(tmp_path / "scores"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
            "validation": {
                "walk_forward": {
                    "train_months": 6, "validation_months": 1,
                    "oos_months": 1, "step_months": 1, "min_oos_completeness": 0.3,
                },
                "bootstrap": {"n_resamples": 100, "seed": 1},
                "permutation": {"n_permutations": 100, "seed": 2, "forward_window": 5},
                "min_sample": {"min_oos_trades": 1},
            },
        }
    )


def _seed_ticker(config: AppConfig, ticker: str, seed: int, n: int = 700) -> None:
    ohlcv = make_synthetic_ohlcv(n, seed=seed, ticker=ticker)
    market = make_synthetic_ohlcv(n, seed=9999, ticker="TOPIX")
    save_ohlcv(ohlcv, ticker, config.data.raw_dir)
    save_ohlcv(ohlcv, ticker, config.data.processed_dir)
    if not (Path(config.data.processed_dir) / "TOPIX.parquet").exists():
        save_ohlcv(market, "TOPIX", config.data.processed_dir)

    panel = compute_feature_panel(ohlcv, market_df=market)
    save_feature_panel(panel, ticker, config.data.features_dir)
    signal_records = compute_signal_records(panel, SignalsConfig())
    save_signal_records(signal_records, ticker, config.data.signals_dir)
    score_records = compute_score_records(panel, signal_records, ScoringConfig())
    save_score_records(score_records, ticker, config.data.scores_dir)


def _json_default(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, float) and obj != obj:
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _write_fake_prior_report(path: Path, config_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config_hash": config_hash, "signal_results": []}, default=_json_default),
        encoding="utf-8",
    )


def _seeded_tickers(config: AppConfig, seeds: list[int]) -> list[str]:
    tickers = [f"T{i}" for i in range(len(seeds))]
    for ticker, seed in zip(tickers, seeds):
        _seed_ticker(config, ticker, seed)
    return tickers


def _strategy_manifest(config: AppConfig) -> dict:
    manifest = build_manifest(
        config, "v1", datetime.date(2026, 8, 20), "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    return json.loads(json.dumps(dataclasses.asdict(manifest), default=_json_default))


def test_run_phase12_ensemble_raises_config_mismatch(
    base_config: AppConfig, tmp_path: Path
) -> None:
    _seeded_tickers(base_config, [500])
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, "wrong_hash")
    _write_fake_prior_report(p7, "wrong_hash")

    with pytest.raises(ConfigMismatchError):
        run_phase12_ensemble(
            base_config, ["T0"], p65, p7, _strategy_manifest(base_config)
        )


def test_run_phase12_ensemble_end_to_end_structure(
    base_config: AppConfig, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [601, 602, 603, 604, 605, 606, 607, 608, 609, 610]
    tickers = _seeded_tickers(base_config, seeds)

    manifest = _strategy_manifest(base_config)
    report = run_phase12_ensemble(base_config, tickers, p65, p7, manifest)

    assert report.config_check.matches is True
    # Same code/config just used to build the manifest a moment ago ->
    # the integrity hash check must agree.
    assert report.integrity_hash_matches_strategy_v1 is True
    assert report.integrity_hash_mismatches == []

    assert len(report.long_count_buckets) == 4
    assert [b.label for b in report.long_count_buckets] == [
        "LONG_COUNT=1", "LONG_COUNT=2", "LONG_COUNT=3", "LONG_COUNT=4+",
    ]
    assert len(report.short_count_buckets) == 4
    assert len(report.net_count_buckets) == 9

    for bucket in report.long_count_buckets + report.short_count_buckets:
        assert bucket.direction in ("LONG", "SHORT")
        # Forward Return stats cover all 7 windows (Phase 12's 15d/20d addition).
        assert [s.window_days for s in bucket.forward_return_stats] == [1, 3, 5, 7, 10, 15, 20]
        assert bucket.decision is not None

    net_zero = next(b for b in report.net_count_buckets if b.label == "NET=0")
    assert net_zero.direction is None
    assert net_zero.cost_metrics is None  # no direction -> no trade-based analysis

    # Combinations: every reported combo actually has >= 2 Signals.
    for combo_analysis in report.long_combinations + report.short_combinations:
        assert combo_analysis.combo.combo_size >= 2
        assert combo_analysis.decision is not None

    # FDR results only cover units that actually had a permutation p-value.
    for key in report.fdr_results:
        assert isinstance(key, str)

    assert report.total_trading_days > 0


def test_run_phase12_ensemble_hash_mismatch_detected(
    base_config: AppConfig, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    tickers = _seeded_tickers(base_config, [701, 702, 703])

    manifest = _strategy_manifest(base_config)
    manifest["hashes"]["features_hash"] = "0" * 64  # simulate a code change since T0

    report = run_phase12_ensemble(base_config, tickers, p65, p7, manifest)
    assert report.integrity_hash_matches_strategy_v1 is False
    assert "features_hash" in report.integrity_hash_mismatches
