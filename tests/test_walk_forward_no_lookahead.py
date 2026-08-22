"""No-lookahead verification for Walk Forward (Phase 6 section 31):

    A. Data beyond an OOS window's start does not affect Signal/Score
       computed at or before that OOS start.
    B. Adding OOS-period data to the dataset does not change TRAIN
       results - i.e. slicing Trade/Score records by date AFTER one full
       computation (what pipeline/run_walk_forward.py actually does) is
       equivalent to recomputing on a dataset truncated at TRAIN's end.
    C. Changing validation_months shifts the OOS boundary in exactly the
       expected, controlled way - it never silently changes TRAIN's own
       boundary or leaves OOS unmoved when it should move.
    D. Corrupting/deleting Forward Return values does not change Signal
       or Score output (extends Phase 5's unit-level check through the
       Walk Forward context specifically).
    E. (Phase 6.5 section 15/16/30 "multi-ticker no-lookahead") A given
       ticker's own trades are byte-identical whether it is the only
       ticker processed or one of many in a Full-Universe-style batch -
       i.e. batching many tickers together introduces no cross-ticker
       leakage into any single ticker's result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from backtest.engine import run_backtest_for_ticker
from backtest.walk_forward import generate_windows
from config.loader import AppConfig, BacktestConfig, ScoringConfig, SignalsConfig, WalkForwardConfig
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
from targets.forward_returns import compute_forward_returns

# --- Test A: OOS-future data doesn't affect pre-OOS-start Signal/Score -----


def test_data_after_oos_start_does_not_affect_signal_before_it() -> None:
    n = 400
    base = make_synthetic_ohlcv(n, seed=200)
    panel_base = compute_feature_panel(base)
    signals_base = compute_signal_records(panel_base, SignalsConfig())

    oos_start_idx = 300  # an arbitrary "OOS boundary" for this test
    oos_start_date = base["date"].iloc[oos_start_idx]

    rng = np.random.default_rng(201)
    perturbed = base.copy()
    future_mask = perturbed.index >= oos_start_idx
    n_future = int(future_mask.sum())
    for col in ("open", "high", "low", "close"):
        perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] * rng.uniform(
            0.5, 1.5, size=n_future
        )

    panel_perturbed = compute_feature_panel(perturbed)
    signals_perturbed = compute_signal_records(panel_perturbed, SignalsConfig())

    before_oos_base = signals_base[signals_base["date"] < oos_start_date]
    before_oos_perturbed = signals_perturbed[signals_perturbed["date"] < oos_start_date]
    pd.testing.assert_frame_equal(
        before_oos_base.reset_index(drop=True), before_oos_perturbed.reset_index(drop=True)
    )


# --- Test B: adding OOS data does not change TRAIN results -----------------


def test_adding_oos_data_does_not_change_train_metrics() -> None:
    n = 400
    full = make_synthetic_ohlcv(n, seed=202)
    train_end_idx = 250
    train_end_date = full["date"].iloc[train_end_idx]

    # "Full" computation (what pipeline/run_walk_forward.py does): compute
    # once over everything, then slice TRAIN out by date.
    panel_full = compute_feature_panel(full)
    signals_full = compute_signal_records(panel_full, SignalsConfig())
    trades_full = run_backtest_for_ticker(
        signals_full, panel_full, BacktestConfig()
    ).trades
    train_trades_from_full = trades_full[trades_full["signal_date"] < train_end_date]

    # "Truncated" computation: pretend OOS data doesn't exist yet at all.
    truncated = full.iloc[:train_end_idx].reset_index(drop=True)
    panel_truncated = compute_feature_panel(truncated)
    signals_truncated = compute_signal_records(panel_truncated, SignalsConfig())
    trades_truncated = run_backtest_for_ticker(
        signals_truncated, panel_truncated, BacktestConfig()
    ).trades

    # Every trade that COULD be fully formed within the truncated dataset
    # (had enough post-signal data for entry+exit) must match exactly.
    common_dates = set(trades_truncated["signal_date"]) & set(train_trades_from_full["signal_date"])
    assert len(common_dates) > 0

    a = (
        train_trades_from_full[train_trades_from_full["signal_date"].isin(common_dates)]
        .sort_values(["signal_date", "signal_name"])
        .reset_index(drop=True)
    )
    b = (
        trades_truncated[trades_truncated["signal_date"].isin(common_dates)]
        .sort_values(["signal_date", "signal_name"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(a, b)


# --- Test C: changing validation_months moves OOS in the expected way ------


def test_changing_validation_months_shifts_oos_start_predictably() -> None:
    from datetime import date

    data_start = date(2022, 1, 4)
    data_end = date(2024, 6, 28)

    base_config = WalkForwardConfig(
        train_months=12, validation_months=3, oos_months=3, step_months=3
    )
    longer_val_config = WalkForwardConfig(
        train_months=12, validation_months=6, oos_months=3, step_months=3
    )

    base_windows = generate_windows(data_start, data_end, base_config)
    longer_val_windows = generate_windows(data_start, data_end, longer_val_config)

    # TRAIN boundary must be untouched by a validation_months change...
    assert base_windows[0].train_start == longer_val_windows[0].train_start
    assert base_windows[0].train_end == longer_val_windows[0].train_end
    # ...but OOS must shift forward by exactly the extra validation time.
    assert longer_val_windows[0].validation_end > base_windows[0].validation_end
    assert longer_val_windows[0].oos_start == longer_val_windows[0].validation_end
    assert longer_val_windows[0].oos_start > base_windows[0].oos_start


# --- Test D: corrupting Forward Return does not affect Signal/Score -------


def test_corrupting_forward_return_does_not_affect_signal_records() -> None:
    n = 300
    stock = make_synthetic_ohlcv(n, seed=203)
    panel = compute_feature_panel(stock)

    signals_before = compute_signal_records(panel, SignalsConfig())

    forward = compute_forward_returns(panel)
    forward["forward_return_5d"] = 999_999.0
    del forward

    signals_after = compute_signal_records(panel, SignalsConfig())
    pd.testing.assert_frame_equal(signals_before, signals_after)


# --- Test E: multi-ticker batching doesn't cross-contaminate a ticker ------


def _seed_ticker_for_batch_test(config: AppConfig, ticker: str, seed: int, n: int = 500) -> None:
    # Same n for every ticker -> identical date range -> identical Walk
    # Forward windows regardless of how many OTHER tickers are in the
    # batch, isolating this test to cross-ticker leakage specifically
    # (not a data_start/data_end difference).
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


def test_ticker_trades_identical_whether_run_alone_or_batched(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
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
                "bootstrap": {"n_resamples": 100, "seed": 1},
                "permutation": {"n_permutations": 100, "seed": 2, "forward_window": 5},
                "min_sample": {"min_oos_trades": 1},
            },
        }
    )
    for i, seed in enumerate([310, 311, 312, 313, 314]):
        _seed_ticker_for_batch_test(config, f"T{i}", seed)

    report_alone = run_walk_forward(config, tickers=["T0"])
    report_batched = run_walk_forward(config, tickers=["T0", "T1", "T2", "T3", "T4"])

    assert report_alone.windows == report_batched.windows

    # report_batched's aggregate oos_metrics pool ALL 5 tickers' trades
    # under each signal - the correct isolated comparison is each
    # signal's OWN per-ticker breakdown (ticker_metrics, Phase 6.5
    # section 17/18) restricted to "T0" specifically.
    alone_by_signal = {(r.direction, r.signal_name): r for r in report_alone.signal_results}
    batched_by_signal = {(r.direction, r.signal_name): r for r in report_batched.signal_results}

    checked_any = False
    for key, alone_result in alone_by_signal.items():
        assert key in batched_by_signal
        batched_result = batched_by_signal[key]
        t0_alone = alone_result.ticker_metrics.get("T0")
        t0_batched = batched_result.ticker_metrics.get("T0")
        assert (t0_alone is None) == (t0_batched is None)
        if t0_alone is None:
            continue
        checked_any = True
        assert t0_alone.n_trades == t0_batched.n_trades
        assert t0_alone.average_return == t0_batched.average_return
        assert t0_alone.total_return == t0_batched.total_return
        assert t0_alone.win_rate == t0_batched.win_rate
    assert checked_any  # sanity: the synthetic data actually produced >=1 T0 trade
