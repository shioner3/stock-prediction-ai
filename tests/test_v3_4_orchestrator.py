"""Phase V3-4 orchestrator integration smoke test - end-to-end on a small
synthetic Universe (mirrors `tests/v3_3_test_helpers.py`'s pattern),
verifying `run_v3_4_analysis()` executes without error and produces a
structurally sane `V3_4Report`. Not a check on the NUMBERS (synthetic
data has no real market-timing-vs-stock-selection signal to detect) -
that judgment only happens in `research/phase_v3_4_report.md` against the
real Full Universe run.
"""

from __future__ import annotations

from pathlib import Path

from v3_3_test_helpers import build_small_dataset_and_windows

from config.loader import TransactionCostConfig
from storage.parquet_store import load_ohlcv
from v3.robustness.orchestrator import run_v3_4_analysis
from v3.robustness.reproduce import reproduce_predictions
from v3.validation.wfo_config import MARKET_REGIME_CONFIG


def test_v3_4_analysis_runs_end_to_end(tmp_path: Path) -> None:
    config, tickers, dataset, windows = build_small_dataset_and_windows(
        tmp_path, n_tickers=10, n_days=700
    )
    assert len(windows) >= 2, "need >=2 WFO windows for this smoke test to be meaningful"

    reproduced = reproduce_predictions(dataset, windows)
    assert set(reproduced.keys()) == {
        "target_raw_5d", "target_raw_10d", "target_raw_15d", "target_raw_20d",
    }
    for target_col, df in reproduced.items():
        assert {"date", "ticker", "actual", "prediction", "window_index"}.issubset(df.columns)
        assert len(df) > 0, target_col

    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    cost_tiers = TransactionCostConfig().tiers

    report = run_v3_4_analysis(
        dataset=dataset, reproduced_predictions=reproduced, tickers=tickers,
        topix_ohlcv=topix, market_regime_config=MARKET_REGIME_CONFIG, cost_tiers=cost_tiers,
        v3_config=config,
    )

    assert report.primary_ranking.n > 0
    assert set(report.market_decomposition.keys()) == {
        "raw", "topix_relative", "beta_adjusted", "sector_relative", "market_neutralized",
    }
    assert set(report.cross_sectional_decomposition.keys()) == {
        "original", "demeaned_mean", "demeaned_median",
    }
    assert report.structural_invariance.rank_ic_identical is True
    assert set(report.regime_robustness.breakdown.keys()) == {"BULL", "NEUTRAL", "BEAR"}
    assert report.day_concentration.n_unique_q5_days >= 0
    assert report.stock_concentration.n_unique_q5_tickers >= 0
    assert set(report.holding_period.keys()) == {5, 10, 15, 20}
    assert set(report.cost_sensitivity.keys()) == {"zero", "low", "base", "high"}
    assert set(report.economic_significance.keys()) == {5, 10, 20}
    assert report.v3_3_decision is not None
    assert report.edge_classification is not None
    # matched control may legitimately find 0 matches on synthetic data
    # (this repo's JPX master cache has no entries for synthetic tickers,
    # so scale/turnover/close all come back NaN and every row gets
    # dropped before matching is attempted) - the important thing is that
    # it runs without raising, not that it finds matches.
    assert report.matched_control.n_q5_rows >= 0
