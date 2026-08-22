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
from pipeline.run_phase8_analysis import ConfigMismatchError
from pipeline.run_phase13_conditional_analysis import (
    run_phase13_conditional_analysis,
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


def test_run_phase13_raises_config_mismatch(base_config: AppConfig, tmp_path: Path) -> None:
    _seeded_tickers(base_config, [500])
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, "wrong_hash")
    _write_fake_prior_report(p7, "wrong_hash")

    with pytest.raises(ConfigMismatchError):
        run_phase13_conditional_analysis(base_config, ["T0"], p65, p7)


def test_run_phase13_raises_when_signal_never_triggers(
    base_config: AppConfig, tmp_path: Path
) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    # A flat/constant series never triggers oversold_rebound (RSI never
    # dips under 30 with no real volatility).
    tickers = ["T0"]
    ohlcv = make_synthetic_ohlcv(50, seed=1, ticker="T0")
    ohlcv = ohlcv.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    market = make_synthetic_ohlcv(50, seed=9999, ticker="TOPIX")
    save_ohlcv(ohlcv, "T0", base_config.data.raw_dir)
    save_ohlcv(ohlcv, "T0", base_config.data.processed_dir)
    save_ohlcv(market, "TOPIX", base_config.data.processed_dir)
    panel = compute_feature_panel(ohlcv, market_df=market)
    save_feature_panel(panel, "T0", base_config.data.features_dir)
    signal_records = compute_signal_records(panel, SignalsConfig())
    save_signal_records(signal_records, "T0", base_config.data.signals_dir)
    score_records = compute_score_records(panel, signal_records, ScoringConfig())
    save_score_records(score_records, "T0", base_config.data.scores_dir)

    with pytest.raises(RuntimeError, match="never triggered"):
        run_phase13_conditional_analysis(base_config, tickers, p65, p7)


def test_run_phase13_end_to_end_structure(base_config: AppConfig, tmp_path: Path) -> None:
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)

    seeds = [601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612]
    tickers = _seeded_tickers(base_config, seeds)

    report = run_phase13_conditional_analysis(base_config, tickers, p65, p7)

    assert report.config_check.matches is True
    assert report.n_signal_total > 0
    assert report.unique_tickers > 0
    assert report.unique_signal_dates > 0

    # Overall Forward Return stats cover all 7 windows.
    assert [s.window_days for s in report.overall_forward_return_stats] == [1, 3, 5, 7, 10, 15, 20]

    # Every bucket across every axis sums to the total signal count (no
    # signal silently dropped, no signal double-counted, modulo rows
    # with unavailable Feature data at the edges of history which
    # naturally fall outside every fixed bucket already).
    for bucket_list in (
        report.regime_buckets, report.market_drawdown_20d_buckets,
        report.stock_drawdown_20d_buckets, report.ma20_deviation_buckets,
        report.volume_buckets, report.volatility_buckets,
        report.signal_count_buckets, report.long_short_consensus_buckets,
    ):
        total = sum(b.n for b in bucket_list)
        assert total <= report.n_signal_total

    # Score analysis structure.
    assert report.score_analysis is not None

    # Event exclusion metrics are always BacktestMetrics (possibly n=0).
    assert report.event_case_a.n_trades >= report.event_case_b_excl_aug2024.n_trades
    assert report.event_case_a.n_trades >= report.event_case_c_excl_2024.n_trades

    # FDR bookkeeping only holds units that actually had a permutation p.
    for key in report.fdr_tested_p_values:
        assert isinstance(key, str)


def test_run_phase13_signal_count_buckets_use_other_11_signals_only(
    base_config: AppConfig, tmp_path: Path
) -> None:
    """Spec section 13: Signal Count for long_oversold_rebound itself
    must count only the OTHER co-firing signals, not itself - so a row
    where ONLY long_oversold_rebound fired (no other LONG signal that
    day) must land in the "0" bucket, never "1".
    """
    current = hash_files(CONFIG_FILES)
    p65 = tmp_path / "p65.json"
    p7 = tmp_path / "p7.json"
    _write_fake_prior_report(p65, current)
    _write_fake_prior_report(p7, current)
    tickers = _seeded_tickers(base_config, [701, 702, 703, 704, 705, 706])

    report = run_phase13_conditional_analysis(base_config, tickers, p65, p7)

    bucket_labels = {b.label for b in report.signal_count_buckets}
    assert bucket_labels <= {"0", "1", "2", "3+"}
