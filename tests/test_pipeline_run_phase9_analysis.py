from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

import pytest
from conftest import make_synthetic_ohlcv

from common.hashing import hash_files
from config.loader import AppConfig, Phase9Config, ScoringConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from pipeline.run_phase8_analysis import ConfigMismatchError
from pipeline.run_phase9_analysis import run_phase9_analysis
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


@pytest.fixture
def phase9_config() -> Phase9Config:
    return Phase9Config.model_validate(
        {
            "day_cluster_bootstrap": {"n_resamples": 200, "seed": 44},
            "block_bootstrap": {"block_length_days": 5, "n_resamples": 200, "seed": 45},
            "timing_placebo": {"offsets": [-5, -1, 5]},
            "winsorization": {"lower_percentile": 0.01, "upper_percentile": 0.99},
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


def test_run_phase9_analysis_raises_config_mismatch_error(
    base_config: AppConfig, phase9_config: Phase9Config, tmp_path: Path
) -> None:
    _seed_ticker(base_config, "T0", 800)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, "wrong")
    _write_fake_prior_report(p7, "wrong")

    with pytest.raises(ConfigMismatchError):
        run_phase9_analysis(
            base_config, phase9_config, ["T0"], p65, p7, tmp_path / "no_jpx_master.xls"
        )


def test_run_phase9_analysis_end_to_end_structure(
    base_config: AppConfig, phase9_config: Phase9Config, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [901, 902, 903, 904, 905, 906]
    tickers = [f"T{i}" for i in range(len(seeds))]
    for ticker, seed in zip(tickers, seeds):
        _seed_ticker(base_config, ticker, seed)

    report = run_phase9_analysis(
        base_config, phase9_config, tickers, p65, p7, tmp_path / "no_jpx_master.xls"
    )

    assert report.config_check.matches is True
    assert report.n_signal_trades_total > 0

    # LOPO: every removed-count must not exceed the full trade population
    # it was drawn from.
    for lopo in report.lopo_by_episode + report.lopo_by_year:
        assert lopo.n_trades_removed <= lopo.full_sample_metrics.n_trades

    # Bootstrap dicts cover all 3 requested statistics.
    assert set(report.day_cluster_bootstrap_bear.keys()) == {
        "mean_return", "expectancy", "profit_factor",
    }
    assert set(report.block_bootstrap_combined.keys()) == {
        "mean_return", "expectancy", "profit_factor",
    }

    # Timing sweep covers exactly the configured offsets, in order.
    assert [r.offset_days for r in report.timing_offset_sweep] == [-5, -1, 5]

    # No JPX master file was provided - sector breakdown must gracefully
    # degrade to empty (NOT_AVAILABLE), not crash.
    assert report.sector_breakdown_bear == {}

    # Cost stress dicts cover all 4 tiers for every scope.
    for stress in (
        report.cost_stress_combined, report.cost_stress_bear,
        report.cost_stress_aug2024_episode, report.cost_stress_bear_excl_aug2024,
    ):
        assert set(stress.keys()) == {"zero", "low", "base", "high"}

    # Forward horizon profile covers all 7 targets/forward_returns.py
    # windows (Phase 5's original 5 + Phase 12's 15d/20d addition).
    assert [r.horizon_days for r in report.forward_horizon_profile] == [1, 3, 5, 7, 10, 15, 20]

    # Scenarios A, B, C, E, F are present (D lives in lopo_by_episode).
    assert {s.name for s in report.scenarios} == {"A", "B", "C", "E", "F"}

    # Scenario A uses the full trade set, so its n_trades matches the total.
    scenario_a = next(s for s in report.scenarios if s.name == "A")
    assert scenario_a.metrics.n_trades == report.n_signal_trades_total


def test_run_phase9_analysis_liquidity_breakdown_uses_raw_dir_data(
    base_config: AppConfig, phase9_config: Phase9Config, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [911, 912, 913, 914, 915, 916]
    tickers = [f"T{i}" for i in range(len(seeds))]
    for ticker, seed in zip(tickers, seeds):
        _seed_ticker(base_config, ticker, seed)

    report = run_phase9_analysis(
        base_config, phase9_config, tickers, p65, p7, tmp_path / "no_jpx_master.xls"
    )
    # Liquidity breakdown either has real buckets or gracefully degrades
    # to empty if BEAR trades didn't happen to occur - either way, no crash.
    assert isinstance(report.liquidity_breakdown_bear, dict)


def test_run_phase9_analysis_is_deterministic_given_same_inputs(
    base_config: AppConfig, phase9_config: Phase9Config, tmp_path: Path
) -> None:
    """Phase 9 section 20/21: same config/data/seeds must give identical
    episode list, LOPO results, bootstrap CIs, timing placebo, and
    scenario results across repeated runs.
    """
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [921, 922, 923, 924, 925]
    tickers = [f"T{i}" for i in range(len(seeds))]
    for ticker, seed in zip(tickers, seeds):
        _seed_ticker(base_config, ticker, seed)

    jpx_path = tmp_path / "no_jpx_master.xls"
    report_a = run_phase9_analysis(base_config, phase9_config, tickers, p65, p7, jpx_path)
    report_b = run_phase9_analysis(base_config, phase9_config, tickers, p65, p7, jpx_path)

    assert [e.index for e in report_a.bear_episodes] == [e.index for e in report_b.bear_episodes]
    assert [
        (e.episode.start_date, e.episode.end_date) for e in report_a.rich_episode_metrics
    ] == [(e.episode.start_date, e.episode.end_date) for e in report_b.rich_episode_metrics]

    for la, lb in zip(report_a.lopo_by_year, report_b.lopo_by_year):
        assert la.period_label == lb.period_label
        assert la.remaining_metrics.profit_factor == lb.remaining_metrics.profit_factor
        assert (
            la.remaining_bootstrap_expectancy.ci_low == lb.remaining_bootstrap_expectancy.ci_low
        )

    for ta, tb in zip(report_a.timing_offset_sweep, report_b.timing_offset_sweep):
        assert ta.offset_days == tb.offset_days
        assert ta.n_trades_bear == tb.n_trades_bear
        assert ta.metrics_bear.profit_factor == tb.metrics_bear.profit_factor

    for sa, sb in zip(report_a.scenarios, report_b.scenarios):
        assert sa.name == sb.name
        assert sa.metrics.n_trades == sb.metrics.n_trades
        assert sa.metrics.profit_factor == sb.metrics.profit_factor
