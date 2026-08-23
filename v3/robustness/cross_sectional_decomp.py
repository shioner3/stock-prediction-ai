"""Spec section 4/5: Cross-sectional vs Time-series decomposition of the
PREDICTION itself (as opposed to `market_decomposition.py`, which varies
the OUTCOME while holding the prediction/ranking fixed).

Spec section 4 lists 4 "prediction variants" (original / same-day
cross-sectional demeaned / market-component-removed / stock-specific
residual) and section 5 asks for BOTH mean- and median-based cross-
sectional neutralization. Operationalized here as exactly 3 variants -
`original`, `demeaned` (subtract the day's MEAN prediction), `demedianed`
(subtract the day's MEDIAN prediction) - since "market component removed"
and "stock-specific residual" are, applied to a single scalar prediction
per ticker per day, the SAME operation as cross-sectional demeaning: the
day's mean/median IS the prediction's market component, and what remains
after subtracting it IS the stock-specific residual. This is stated
explicitly rather than building near-duplicate machinery for a
distinction spec section 5's own formula (`prediction = market_component
+ stock_specific_component`) already collapses into one subtraction.

A key STRUCTURAL fact this module verifies empirically before reporting
any numbers: `v3.validation.wfo_config`'s own bucket/ranking primitives
compute Rank IC and Top-N PER DAY (`v2.validation.ic.
compute_daily_spearman_ic` groups by date; `v3.validation.topn_portfolio.
compute_daily_top_n_tickers` groups by date then takes `nlargest` within
the group) - both are therefore mathematically INVARIANT to subtracting
a per-day constant (mean or median) from every ticker's prediction that
day, since that operation never changes the WITHIN-day rank order. Only
Q1-Q5 bucket assignment (`scoring.validation.assign_quantile_buckets`) is
a single GLOBAL quantile split over the whole pooled OOS prediction
column, POOLING across days - THIS is the one place a day-level shift in
the prediction can move rows between buckets, and therefore the one place
this decomposition can show a real difference. This asymmetry - not an
assumption - is why V3-3's Q1-Q5 spread can be sensitive to day-level
prediction drift even though its Rank IC/Top-N cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from v3.validation.ranking_metrics import RankingResult, evaluate_ranking
from v3.validation.robustness import SpreadBootstrapBattery, bootstrap_q5_q1_spread
from v3.validation.topn_portfolio import TopNPortfolioMetrics, compute_topn_portfolio_metrics
from v3.validation.wfo_config import TOP_N_VALUES

VARIANT_ORIGINAL = "original"
VARIANT_DEMEANED = "demeaned_mean"
VARIANT_DEMEDIANED = "demeaned_median"
PREDICTION_VARIANTS = (VARIANT_ORIGINAL, VARIANT_DEMEANED, VARIANT_DEMEDIANED)


def add_demeaned_prediction_columns(
    predictions: pd.DataFrame, prediction_col: str = "prediction", date_col: str = "date"
) -> pd.DataFrame:
    out = predictions.copy()
    day_mean = out.groupby(date_col)[prediction_col].transform("mean")
    day_median = out.groupby(date_col)[prediction_col].transform("median")
    out[f"prediction_{VARIANT_DEMEANED}"] = out[prediction_col] - day_mean
    out[f"prediction_{VARIANT_DEMEDIANED}"] = out[prediction_col] - day_median
    out[f"prediction_{VARIANT_ORIGINAL}"] = out[prediction_col]
    return out


@dataclass(frozen=True)
class PredictionVariantResult:
    variant: str
    ranking: RankingResult
    topn: dict[int, TopNPortfolioMetrics]
    spread_bootstrap: SpreadBootstrapBattery


def evaluate_prediction_variant(
    predictions_with_variants: pd.DataFrame, variant: str, window_days: int,
    actual_col: str = "actual",
) -> PredictionVariantResult:
    prediction_col = f"prediction_{variant}"
    ranking = evaluate_ranking(predictions_with_variants, window_days, prediction_col, actual_col)
    topn = {
        n: compute_topn_portfolio_metrics(
            predictions_with_variants, n, actual_col, window_days, score_col=prediction_col
        )
        for n in TOP_N_VALUES
    }
    spread_bootstrap = bootstrap_q5_q1_spread(
        predictions_with_variants, prediction_col=prediction_col, actual_col=actual_col
    )
    return PredictionVariantResult(
        variant=variant, ranking=ranking, topn=topn, spread_bootstrap=spread_bootstrap
    )


def run_cross_sectional_decomposition(
    predictions: pd.DataFrame, window_days: int, prediction_col: str = "prediction",
    actual_col: str = "actual", date_col: str = "date",
) -> dict[str, PredictionVariantResult]:
    with_variants = add_demeaned_prediction_columns(predictions, prediction_col, date_col)
    return {
        variant: evaluate_prediction_variant(with_variants, variant, window_days, actual_col)
        for variant in PREDICTION_VARIANTS
    }


@dataclass(frozen=True)
class StructuralInvarianceCheck:
    rank_ic_identical: bool
    top5_selection_identical: bool
    top10_selection_identical: bool
    top20_selection_identical: bool
    q5_q1_spread_original: float | None
    q5_q1_spread_demeaned: float | None
    q5_q1_spread_demedianed: float | None


def check_structural_invariance(
    results: dict[str, PredictionVariantResult], tolerance: float = 1e-9
) -> StructuralInvarianceCheck:
    orig = results[VARIANT_ORIGINAL]
    dem = results[VARIANT_DEMEANED]
    demed = results[VARIANT_DEMEDIANED]

    def _ic_close(a: RankingResult, b: RankingResult) -> bool:
        if a.ic_summary.mean_ic is None or b.ic_summary.mean_ic is None:
            return a.ic_summary.mean_ic == b.ic_summary.mean_ic
        return bool(np.isclose(a.ic_summary.mean_ic, b.ic_summary.mean_ic, atol=tolerance))

    def _topn_identical(n: int) -> bool:
        a = {d.date for d in orig.topn[n].base.daily_returns if d.equal_weight_return is not None}
        b = {d.date for d in dem.topn[n].base.daily_returns if d.equal_weight_return is not None}
        c = {d.date for d in demed.topn[n].base.daily_returns if d.equal_weight_return is not None}
        ra = [d.equal_weight_return for d in orig.topn[n].base.daily_returns]
        rb = [d.equal_weight_return for d in dem.topn[n].base.daily_returns]
        rc = [d.equal_weight_return for d in demed.topn[n].base.daily_returns]
        return a == b == c and ra == rb == rc

    ic_identical = (
        _ic_close(orig.ranking, dem.ranking) and _ic_close(orig.ranking, demed.ranking)
    )
    return StructuralInvarianceCheck(
        rank_ic_identical=ic_identical,
        top5_selection_identical=_topn_identical(5),
        top10_selection_identical=_topn_identical(10),
        top20_selection_identical=_topn_identical(20),
        q5_q1_spread_original=orig.ranking.q5_q1_spread,
        q5_q1_spread_demeaned=dem.ranking.q5_q1_spread,
        q5_q1_spread_demedianed=demed.ranking.q5_q1_spread,
    )


def daily_market_component_correlation(
    predictions: pd.DataFrame, prediction_col: str = "prediction", actual_col: str = "actual",
    date_col: str = "date",
) -> float | None:
    """Correlation between each day's MEAN prediction (the "market
    component" section 4/5 refer to) and that day's mean realized return
    across the same rows - if the model's day-level prediction level
    genuinely tracks market direction, this should be positive and
    non-trivial; if near zero, the day-level component is closer to noise
    than a real market-timing signal.
    """
    daily = predictions.groupby(date_col).agg(
        mean_prediction=(prediction_col, "mean"), mean_actual=(actual_col, "mean")
    ).dropna()
    if (
        len(daily) < 2
        or daily["mean_prediction"].nunique() < 2
        or daily["mean_actual"].nunique() < 2
    ):
        return None
    corr = np.corrcoef(
        daily["mean_prediction"].to_numpy(), daily["mean_actual"].to_numpy()
    )[0, 1]
    return float(corr) if corr == corr else None
