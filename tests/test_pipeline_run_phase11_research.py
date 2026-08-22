from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import make_synthetic_ohlcv

from backtest.decision import Decision
from common.hashing import hash_files
from config.loader import AppConfig, ScoringConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from forward_test.manifest import build_manifest, save_manifest
from pipeline.run_phase8_analysis import ConfigMismatchError
from pipeline.run_phase11_research import (
    FROZEN_STRATEGY_SIGNAL,
    REMAINING_SIGNALS,
    run_phase11_research,
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


# --- REMAINING_SIGNALS static invariants ----------------------------------------


def test_remaining_signals_excludes_the_frozen_strategy_signal() -> None:
    assert FROZEN_STRATEGY_SIGNAL not in REMAINING_SIGNALS
    assert FROZEN_STRATEGY_SIGNAL == ("LONG", "long_oversold_rebound")


def test_remaining_signals_has_exactly_11_unique_entries() -> None:
    assert len(REMAINING_SIGNALS) == 11
    assert len(set(REMAINING_SIGNALS)) == 11


def test_remaining_signals_matches_the_known_signal_set() -> None:
    expected = {
        ("LONG", "long_breakout"), ("LONG", "long_ma_rebound"),
        ("LONG", "long_momentum_continuation"), ("LONG", "long_pullback"),
        ("LONG", "long_volume_breakout"), ("SHORT", "short_breakdown"),
        ("SHORT", "short_ma_rejection"), ("SHORT", "short_momentum_continuation"),
        ("SHORT", "short_overbought_reversal"), ("SHORT", "short_pullback"),
        ("SHORT", "short_volume_breakdown"),
    }
    assert set(REMAINING_SIGNALS) == expected


# --- run_phase11_research end-to-end ---------------------------------------------


def test_run_phase11_research_raises_config_mismatch_error(
    base_config: AppConfig, tmp_path: Path
) -> None:
    _seeded_tickers(base_config, [500])
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, "wrong_hash")
    _write_fake_prior_report(p7, "wrong_hash")

    with pytest.raises(ConfigMismatchError):
        run_phase11_research(
            base_config, ["T0"], p65, p7, tmp_path / "jpx.xls",
            target_signals=[("LONG", "long_breakout")],
        )


def test_run_phase11_research_never_triggered_signal_recorded_with_note(
    base_config: AppConfig, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    tickers = _seeded_tickers(base_config, [500])

    report = run_phase11_research(
        base_config, tickers, p65, p7, tmp_path / "jpx.xls",
        target_signals=[("LONG", "does_not_exist")],
    )

    assert len(report.signals) == 1
    s = report.signals[0]
    assert s.direction == "LONG"
    assert s.signal_name == "does_not_exist"
    assert s.combined is None
    assert s.decision is None
    assert s.phase8 is None
    assert s.phase9 is None
    assert s.note is not None and "never triggered" in s.note


def test_run_phase11_research_end_to_end_structure(base_config: AppConfig, tmp_path: Path) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [601, 602, 603, 604, 605, 606, 607, 608]
    tickers = _seeded_tickers(base_config, seeds)

    # Restrict to 2 remaining signals - the full 11-signal sweep is
    # exercised by the real Phase 11 Research run against data/phase7/,
    # not by this fast synthetic smoke test.
    target_signals = [("LONG", "long_breakout"), ("SHORT", "short_pullback")]
    report = run_phase11_research(
        base_config, tickers, p65, p7, tmp_path / "jpx.xls", target_signals=target_signals,
    )

    assert report.config_check.matches is True
    assert len(report.signals) == 2
    for s, (direction, name) in zip(report.signals, target_signals):
        assert s.direction == direction
        assert s.signal_name == name
        if s.combined is None:
            assert s.note is not None  # never triggered on this small synthetic sample
            continue
        assert isinstance(s.decision, Decision)
        assert s.phase8 is not None
        assert s.phase8.target.direction == direction
        assert s.phase8.target.signal_name == name
        assert s.phase9 is not None
        assert s.combined_day_concentration is not None
        # Phase 11's offsets (incl. 0/+1/+3) must be exactly what Phase 9
        # sub-analysis actually swept - not Phase 9's own persisted default.
        swept_offsets = {r.offset_days for r in s.phase9.timing_offset_sweep}
        assert swept_offsets == {-15, -10, -5, -3, -1, 0, 1, 3, 5, 10}


def test_run_phase11_research_calls_walk_forward_exactly_once(
    base_config: AppConfig, tmp_path: Path
) -> None:
    """The whole point of injecting `combined_report` into
    run_phase8_analysis (pipeline/run_phase8_analysis.py) is to avoid
    recomputing the expensive Combined WFO/permutation once per signal.
    This directly proves that invariant: run_walk_forward must be called
    exactly once no matter how many target_signals are requested.
    """
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    tickers = _seeded_tickers(base_config, [701, 702, 703, 704])

    target_signals = [
        ("LONG", "long_breakout"), ("LONG", "long_ma_rebound"), ("SHORT", "short_pullback"),
    ]

    from pipeline.run_walk_forward import run_walk_forward as real_run_walk_forward

    call_count = 0

    def _counting_run_walk_forward(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        return real_run_walk_forward(*args, **kwargs)  # type: ignore[arg-type]

    with patch(
        "pipeline.run_phase11_research.run_walk_forward", side_effect=_counting_run_walk_forward
    ):
        run_phase11_research(
            base_config, tickers, p65, p7, tmp_path / "jpx.xls", target_signals=target_signals,
        )

    assert call_count == 1


def test_run_phase11_research_does_not_alter_forward_test_strategy_hash(
    base_config: AppConfig, tmp_path: Path
) -> None:
    """Phase 11 section 26: running Track B Research must never touch
    Track A's Forward Test Strategy manifest/hash - the two tracks share
    no mutable state at all (Research reads Signal/Backtest/WFO code the
    SAME way any other analysis phase does; it never writes to
    data/forward_test/).
    """
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    tickers = _seeded_tickers(base_config, [801, 802])

    manifest_path = tmp_path / "forward_test_manifest.json"
    manifest = build_manifest(
        base_config, "v1", datetime.date(2026, 8, 20), "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    save_manifest(manifest, manifest_path)
    before = manifest_path.read_text(encoding="utf-8")

    run_phase11_research(
        base_config, tickers, p65, p7, tmp_path / "jpx.xls",
        target_signals=[("LONG", "long_breakout")],
    )

    after = manifest_path.read_text(encoding="utf-8")
    assert after == before
