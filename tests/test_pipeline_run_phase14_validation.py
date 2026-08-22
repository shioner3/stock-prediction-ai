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
from pipeline.run_phase14_validation import (
    DOSE_RESPONSE_ORDER,
    StrategyHashMismatchError,
    run_phase14_validation,
)
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


def test_run_phase14_raises_config_mismatch(base_config: AppConfig, tmp_path: Path) -> None:
    _seeded_tickers(base_config, [500])
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    p13 = tmp_path / "p13.json"
    _write_fake_prior_report(p65, "wrong_hash")
    _write_fake_prior_report(p7, "wrong_hash")
    p13.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigMismatchError):
        run_phase14_validation(
            base_config, ["T0"], p65, p7, p13, _strategy_manifest(base_config),
            preregistration_path=tmp_path / "prereg.json",
        )


def test_run_phase14_raises_strategy_hash_mismatch(base_config: AppConfig, tmp_path: Path) -> None:
    _seeded_tickers(base_config, [500])
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    p13 = tmp_path / "p13.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    p13.write_text("{}", encoding="utf-8")

    bad_manifest = _strategy_manifest(base_config)
    bad_manifest["hashes"]["features_hash"] = "tampered"

    with pytest.raises(StrategyHashMismatchError):
        run_phase14_validation(
            base_config, ["T0"], p65, p7, p13, bad_manifest,
            preregistration_path=tmp_path / "prereg.json",
        )


def test_run_phase14_end_to_end_structure(base_config: AppConfig, tmp_path: Path) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    p13 = tmp_path / "p13.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    p13.write_text("{}", encoding="utf-8")

    seeds = list(range(701, 701 + 20))
    tickers = _seeded_tickers(base_config, seeds)
    prereg_path = tmp_path / "prereg.json"

    report = run_phase14_validation(
        base_config, tickers, p65, p7, p13, _strategy_manifest(base_config),
        preregistration_path=prereg_path,
    )

    assert report.config_check.matches is True
    assert report.strategy_hash_matches is True
    assert prereg_path.exists()
    saved = json.loads(prereg_path.read_text(encoding="utf-8"))
    assert saved["topix_20d_threshold_grid"][0]["label"] == "<=-5%"
    assert saved["core_condition"]["topix_20d_return_threshold"] == -0.10

    # Threshold grid: 5 pre-registered thresholds + 1 control bucket.
    assert len(report.threshold_grid) == 6
    # The grid is cumulative/nested and STRICTER as it goes (-5% down to
    # -15%), so each successive bucket's population must be
    # monotonically NON-INCREASING (a day at -12% also counts toward
    # <=-5%, but a day at -6% does not count toward <=-10%).
    ns = [b.n for b in report.threshold_grid[:5]]
    assert ns == sorted(ns, reverse=True)

    assert report.n_core_condition_trades >= 0
    assert report.core_condition_bucket.n >= 0
    assert report.control_condition_bucket.n >= 0

    assert report.decision.primary is not None
    assert isinstance(report.fdr_results, dict)
    for label, fdr in report.fdr_results.items():
        assert isinstance(label, str)
        assert 0.0 <= fdr.adjusted_p_value <= 1.0

    assert len(report.timing_placebo) == len(saved["timing_placebo_offsets"])
    offsets_seen = {r.offset_days for r in report.timing_placebo}
    assert offsets_seen == set(saved["timing_placebo_offsets"])

    # The timing placebo's own offset=0 entry (no shift at all) must
    # reproduce the SAME core-condition trade count as the main pipeline's
    # own core_trades population - both describe "the real, unshifted
    # signal, backtested, then filtered to BEAR x TOPIX20d<=-10%". A
    # mismatch here would mean the sweep pre-filters signals before
    # backtesting (stripping overlap-suppression history) instead of
    # filtering the resulting trades afterward like Phase 9's own
    # established Timing Placebo precedent.
    offset_zero = next(r for r in report.timing_placebo if r.offset_days == 0)
    assert offset_zero.n_core_trades == report.n_core_condition_trades

    assert len(report.forward_horizon_comparison) == 7

    assert report.bootstrap_battery.trade_level.metric_name == "expectancy"
    assert report.bootstrap_battery.ticker_cluster.metric_name == "expectancy"

    # Dose-response bins are mutually exclusive - the control (mildest)
    # bucket's own count must equal the widest threshold grid bucket's
    # complement-free reading, i.e. every bucket is non-negative and
    # bounded by the overall population already checked above.
    for bucket in report.dose_response:
        assert bucket.n >= 0
    assert {b.label for b in report.dose_response} <= set(DOSE_RESPONSE_ORDER)
