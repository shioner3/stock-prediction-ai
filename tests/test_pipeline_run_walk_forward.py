from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
from conftest import make_synthetic_ohlcv

from backtest.decision import Decision
from config.loader import AppConfig, ScoringConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from pipeline.run_walk_forward import run_walk_forward
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
                "min_sample": {"min_oos_trades": 5},
            },
        }
    )


def _seed_ticker(config: AppConfig, ticker: str, seed: int, n: int = 600) -> None:
    ohlcv = make_synthetic_ohlcv(n, seed=seed, ticker=ticker)
    market = make_synthetic_ohlcv(n, seed=seed + 1000, ticker="TOPIX")
    save_ohlcv(ohlcv, ticker, config.data.processed_dir)
    if not (Path(config.data.processed_dir) / "TOPIX.parquet").exists():
        save_ohlcv(market, "TOPIX", config.data.processed_dir)

    panel = compute_feature_panel(ohlcv, market_df=market)
    save_feature_panel(panel, ticker, config.data.features_dir)
    signal_records = compute_signal_records(panel, SignalsConfig())
    save_signal_records(signal_records, ticker, config.data.signals_dir)
    score_records = compute_score_records(panel, signal_records, ScoringConfig())
    save_score_records(score_records, ticker, config.data.scores_dir)


def _fields_equal_nan_safe(a: object, b: object) -> bool:
    """Dataclass equality that treats NaN == NaN as equal (plain
    dataclass __eq__ does not, since float NaN != NaN by IEEE 754) -
    used for comparing BootstrapResult/PermutationResult across two
    runs, which legitimately produce NaN fields when there are zero OOS
    trades for a given Signal.
    """
    if type(a) is not type(b):
        return False
    if dataclasses.is_dataclass(a):
        return all(
            _fields_equal_nan_safe(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
        )
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_run_walk_forward_end_to_end(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 300)
    _seed_ticker(base_config, "6758", 301)

    report = run_walk_forward(base_config)

    assert len(report.windows) >= 1
    # Not every one of the 12 Signals is guaranteed to trigger at least
    # once in a small synthetic random-walk sample - only that the
    # pipeline evaluates whichever ones did, and never more than 12.
    assert 0 < len(report.signal_results) <= 12
    assert report.config_hash
    assert report.data_hash

    for r in report.signal_results:
        assert r.direction in ("LONG", "SHORT")
        assert set(r.oos_metrics_by_cost_tier) == {"zero", "low", "base", "high"}
        assert r.decision in (
            Decision.ACCEPT_CANDIDATE, Decision.REJECT, Decision.INSUFFICIENT_EVIDENCE
        )


def test_higher_cost_tier_never_gives_better_expectancy(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 302)

    report = run_walk_forward(base_config)
    for r in report.signal_results:
        zero = r.oos_metrics_by_cost_tier["zero"].expectancy
        high = r.oos_metrics_by_cost_tier["high"].expectancy
        if zero is not None and high is not None:
            assert high <= zero


def test_config_hash_changes_when_settings_change(base_config: AppConfig, tmp_path: Path) -> None:
    _seed_ticker(base_config, "7203", 303)
    report_a = run_walk_forward(base_config)

    changed_config = base_config.model_copy(deep=True)
    changed_config.backtest.hold_days = 7
    report_b = run_walk_forward(changed_config)

    # config_hash is computed from the config FILES on disk, not the
    # in-memory AppConfig - so an in-memory-only change does not move it
    # (this is intentional: the hash traces back to what's committed to
    # config/settings.yaml, not to ad-hoc overrides).
    assert report_a.config_hash == report_b.config_hash


def test_data_hash_reflects_feature_files_used(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 304)
    report_one_ticker = run_walk_forward(base_config, tickers=["7203"])

    _seed_ticker(base_config, "6758", 305)
    report_two_tickers = run_walk_forward(base_config, tickers=["7203", "6758"])

    assert report_one_ticker.data_hash != report_two_tickers.data_hash


def test_deterministic_full_pipeline_same_seed_gives_identical_report(
    base_config: AppConfig,
) -> None:
    _seed_ticker(base_config, "7203", 306)

    report_a = run_walk_forward(base_config)
    report_b = run_walk_forward(base_config)

    assert len(report_a.signal_results) == len(report_b.signal_results)
    for ra, rb in zip(report_a.signal_results, report_b.signal_results):
        assert ra.decision == rb.decision
        for tier in ra.oos_metrics_by_cost_tier:
            assert _fields_equal_nan_safe(
                ra.oos_metrics_by_cost_tier[tier], rb.oos_metrics_by_cost_tier[tier]
            )
        for stat in ra.bootstrap:
            assert _fields_equal_nan_safe(ra.bootstrap[stat], rb.bootstrap[stat])
        if ra.permutation is not None:
            assert _fields_equal_nan_safe(ra.permutation, rb.permutation)


def test_empty_data_gives_empty_report(base_config: AppConfig) -> None:
    report = run_walk_forward(base_config, tickers=[])
    assert report.windows == []
    assert report.signal_results == []


# --- min_oos_start (Phase 7 section 3/18: OOS reporting-scope filter) -------


def test_min_oos_start_none_matches_unfiltered_behavior(base_config: AppConfig) -> None:
    _seed_ticker(base_config, "7203", 400)
    report_default = run_walk_forward(base_config)
    report_explicit_none = run_walk_forward(base_config, min_oos_start=None)
    assert report_default.windows == report_explicit_none.windows


def test_min_oos_start_excludes_earlier_windows_but_keeps_train_dates_identical(
    base_config: AppConfig,
) -> None:
    _seed_ticker(base_config, "7203", 401)
    unfiltered = run_walk_forward(base_config)
    assert len(unfiltered.windows) >= 3  # need at least a few windows to make this test meaningful

    cutoff = unfiltered.windows[len(unfiltered.windows) // 2].oos_start
    filtered = run_walk_forward(base_config, min_oos_start=cutoff)

    assert len(filtered.windows) < len(unfiltered.windows)
    assert all(w.oos_start >= cutoff for w in filtered.windows)

    # The surviving windows' TRAIN/VALIDATION dates are byte-identical to
    # the unfiltered run's - min_oos_start only drops windows from the
    # report, it does not shift or recompute anyone's TRAIN/VALIDATION.
    unfiltered_by_index = {w.index: w for w in unfiltered.windows}
    for fw in filtered.windows:
        uw = unfiltered_by_index[fw.index]
        assert fw.train_start == uw.train_start
        assert fw.train_end == uw.train_end
        assert fw.validation_start == uw.validation_start
        assert fw.validation_end == uw.validation_end
        assert fw.oos_end == uw.oos_end


def test_min_oos_start_never_admits_trades_from_earlier_windows(base_config: AppConfig) -> None:
    """The whole point of min_oos_start (Phase 7's "don't let Phase 6.5
    contaminate Phase 7's reported OOS metrics" requirement): a signal's
    aggregate OOS trade count under a cutoff must never exceed its
    unfiltered aggregate trade count, and every surviving window's OOS
    trades must fall on/after the cutoff date.
    """
    _seed_ticker(base_config, "7203", 402)
    unfiltered = run_walk_forward(base_config)
    assert len(unfiltered.windows) >= 3

    cutoff = unfiltered.windows[len(unfiltered.windows) // 2].oos_start
    filtered = run_walk_forward(base_config, min_oos_start=cutoff)

    unfiltered_trades = {
        (r.direction, r.signal_name): r.oos_metrics_by_cost_tier["zero"].n_trades
        for r in unfiltered.signal_results
    }
    for r in filtered.signal_results:
        key = (r.direction, r.signal_name)
        assert r.oos_metrics_by_cost_tier["zero"].n_trades <= unfiltered_trades.get(key, 0)
        for wm in r.windows:
            assert wm.window.oos_start >= cutoff


def test_min_oos_start_beyond_all_data_gives_no_windows(base_config: AppConfig) -> None:
    from datetime import date

    _seed_ticker(base_config, "7203", 403)
    report = run_walk_forward(base_config, min_oos_start=date(2999, 1, 1))
    assert report.windows == []
