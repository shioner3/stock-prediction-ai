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
from pipeline.run_phase8_analysis import (
    ConfigMismatchError,
    run_phase8_analysis,
    verify_config_hash,
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
                "bootstrap": {"n_resamples": 200, "seed": 1},
                "permutation": {"n_permutations": 200, "seed": 2, "forward_window": 5},
                "min_sample": {"min_oos_trades": 1},
            },
        }
    )


def _seed_ticker(config: AppConfig, ticker: str, seed: int, n: int = 700) -> None:
    ohlcv = make_synthetic_ohlcv(n, seed=seed, ticker=ticker)
    market = make_synthetic_ohlcv(n, seed=9999, ticker="TOPIX")
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


# --- verify_config_hash --------------------------------------------------------


def test_verify_config_hash_matches_when_both_reports_match_current(tmp_path: Path) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    result = verify_config_hash(p65, p7)
    assert result.matches is True
    assert result.current_config_hash == current


def test_verify_config_hash_mismatch_when_report_differs(tmp_path: Path) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, "deadbeef" * 8)

    result = verify_config_hash(p65, p7)
    assert result.matches is False


def test_verify_config_hash_missing_files_gives_mismatch(tmp_path: Path) -> None:
    result = verify_config_hash(tmp_path / "nope1.json", tmp_path / "nope2.json")
    assert result.matches is False
    assert result.phase6_5_config_hash is None
    assert result.phase7_config_hash is None


# --- run_phase8_analysis end-to-end ---------------------------------------------


def test_run_phase8_analysis_raises_config_mismatch_error(
    base_config: AppConfig, tmp_path: Path
) -> None:
    _seed_ticker(base_config, "T0", 500)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, "wrong_hash")
    _write_fake_prior_report(p7, "wrong_hash")

    with pytest.raises(ConfigMismatchError):
        run_phase8_analysis(base_config, ["T0"], p65, p7)


def test_run_phase8_analysis_raises_when_target_signal_never_triggers(
    base_config: AppConfig, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    _seed_ticker(base_config, "T0", 500)

    with pytest.raises(RuntimeError, match="never triggered"):
        run_phase8_analysis(
            base_config, ["T0"], p65, p7, target_signal_name="does_not_exist"
        )


def test_run_phase8_analysis_end_to_end_structure(base_config: AppConfig, tmp_path: Path) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [601, 602, 603, 604, 605, 606]
    tickers = [f"T{i}" for i in range(len(seeds))]
    for ticker, seed in zip(tickers, seeds):
        _seed_ticker(base_config, ticker, seed)

    report = run_phase8_analysis(base_config, tickers, p65, p7)

    assert report.config_check.matches is True
    assert report.target.direction == "LONG"
    assert report.target.signal_name == "long_oversold_rebound"
    assert report.target.combined.signal_name == "long_oversold_rebound"

    # Structural invariants that must hold regardless of whether this
    # particular synthetic random walk happened to include a BEAR period.
    assert len(report.target.bear_by_window) == len(report.windows)
    assert set(report.target.bear_by_year.keys()) == {2022, 2023, 2024, 2025, 2026}
    assert report.target.bear_concentration.n_trades == sum(
        m.n_trades for m in report.target.bear_ticker_metrics.values()
    )
    assert set(report.target.bear_top_k_shares.keys()) == {1, 5, 10, 20}
    assert set(report.target.bear_cost_sensitivity.keys()) == {"zero", "low", "base", "high"}
    assert set(report.target.bear_bootstrap.keys()) == {
        "mean_return", "profit_factor", "expectancy",
    }
    assert report.target.placebo.shift_trading_days > 0
    # 11 other signals (12 total minus the target) - some may not have
    # triggered in this small synthetic sample, that's fine.
    assert len(report.other_signals) <= 11
    assert all(s.signal_name != "long_oversold_rebound" for s in report.other_signals)


def test_run_phase8_analysis_lopo_groups_never_exceed_bear_trade_count(
    base_config: AppConfig, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [701, 702, 703, 704, 705, 706]
    tickers = [f"T{i}" for i in range(len(seeds))]
    for ticker, seed in zip(tickers, seeds):
        _seed_ticker(base_config, ticker, seed)

    report = run_phase8_analysis(base_config, tickers, p65, p7)
    total_bear_trades = sum(m.n_trades for m in report.target.bear_ticker_metrics.values())

    for lopo in report.target.lopo_by_year + report.target.lopo_by_episode:
        assert lopo.n_trades_removed <= total_bear_trades
        assert lopo.classification in ("STABLE", "PERIOD_DEPENDENT")
