"""Benchmark rankings (spec section 14): Random / Simple Momentum / V2
Initial Research Score - each produces a "prediction"-shaped column so
`v3/validation/ranking_metrics.py::evaluate_ranking()` and
`v3/validation/topn_portfolio.py::compute_topn_portfolio_metrics()` can
be reused UNCHANGED against them, exactly as against a real Model's
predictions.

Simple Momentum uses `return_20d` - one of V1's own existing Momentum
Feature columns (already in `v3/features/registry.py::CORE_FEATURE_NAMES`,
fixed here BEFORE running, not chosen after seeing which feature looked
best) - as the ranking score directly, no model involved.

V2 Score reuses `v2/pipeline.py::run_v2_ranking()` (frozen Phase V2-1
code, unmodified) to compute V2's `total_score` for the SAME Universe,
then joins it onto V3's OOS rows by (date, ticker). Computed ONCE for the
full date range (not once per WFO window) since it does not depend on
any V3 train/test split.
"""

from __future__ import annotations

import pandas as pd

from v2.config.loader import V2Config
from v2.pipeline import run_v2_ranking
from v3.models.cross_sectional import RANDOM_BASELINE_SEED, add_random_baseline_column
from v3.validation.data_integrity import clean_predictions

SIMPLE_MOMENTUM_FEATURE = "return_20d"


def random_benchmark_predictions(
    oos_predictions: pd.DataFrame, seed: int = RANDOM_BASELINE_SEED
) -> pd.DataFrame:
    with_random = add_random_baseline_column(oos_predictions, seed=seed, column_name="prediction")
    return with_random[["date", "ticker", "actual", "prediction"]]


def momentum_benchmark_predictions(
    oos_dataset_slice: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    """oos_dataset_slice: rows from the FULL V3 Dataset (has return_20d
    and every target column) restricted to the OOS date range being
    compared - NOT the model's own `predictions` DataFrame, since that
    one doesn't carry raw Feature columns.
    """
    out = oos_dataset_slice[["date", "ticker", SIMPLE_MOMENTUM_FEATURE, target_col]].copy()
    out = out.rename(columns={SIMPLE_MOMENTUM_FEATURE: "prediction", target_col: "actual"})
    return clean_predictions(out, label=f"(momentum benchmark, {target_col})")


def compute_v2_score_panel(v2_config: V2Config, tickers: list[str]) -> pd.DataFrame:
    """V2's total_score for every (date, ticker) in the Universe, computed
    ONCE. Callers merge this onto their own (date, ticker, actual) frame.
    """
    ranked = run_v2_ranking(v2_config, tickers=tickers)
    return ranked[["date", "ticker", "total_score"]].rename(columns={"total_score": "prediction"})


def v2_score_benchmark_predictions(
    oos_dataset_slice: pd.DataFrame, v2_score_panel: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    merged = oos_dataset_slice[["date", "ticker", target_col]].merge(
        v2_score_panel, on=["date", "ticker"], how="inner"
    )
    merged = merged.rename(columns={target_col: "actual"})
    return clean_predictions(merged, label=f"(V2 Score benchmark, {target_col})")
