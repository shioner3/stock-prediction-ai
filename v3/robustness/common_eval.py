"""Shared "evaluate this filtered/transformed slice of Primary predictions"
helper (spec sections 6-11's Regime/Year/Event/Stock/Sector leave-one-out
analyses all reduce to the same mechanical pattern: filter rows, re-bucket
via the frozen `assign_quantile_buckets()`, compute spread/IC/Top-N/PF/
Expectancy/Bootstrap CI). One shared function avoids re-deriving this
5-times over.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from v3.validation.ranking_metrics import RankingResult, evaluate_ranking
from v3.validation.robustness import SpreadBootstrapBattery, bootstrap_q5_q1_spread
from v3.validation.topn_portfolio import TopNPortfolioMetrics, compute_topn_portfolio_metrics
from v3.validation.wfo_config import TOP_N_VALUES

MIN_ROWS_FOR_BOOTSTRAP = 200  # below this, day-cluster/block resampling is too noisy to report


@dataclass(frozen=True)
class SliceEvaluation:
    label: str
    n: int
    ranking: RankingResult
    topn: dict[int, TopNPortfolioMetrics]
    spread_bootstrap: SpreadBootstrapBattery | None


def evaluate_slice(
    predictions: pd.DataFrame, label: str, window_days: int, prediction_col: str = "prediction",
    actual_col: str = "actual",
) -> SliceEvaluation:
    ranking = evaluate_ranking(predictions, window_days, prediction_col, actual_col)
    topn = {
        n: compute_topn_portfolio_metrics(
            predictions, n, actual_col, window_days, score_col=prediction_col
        )
        for n in TOP_N_VALUES
    }
    spread_bootstrap = (
        bootstrap_q5_q1_spread(predictions, prediction_col=prediction_col, actual_col=actual_col)
        if ranking.n >= MIN_ROWS_FOR_BOOTSTRAP
        else None
    )
    return SliceEvaluation(
        label=label, n=ranking.n, ranking=ranking, topn=topn, spread_bootstrap=spread_bootstrap
    )
