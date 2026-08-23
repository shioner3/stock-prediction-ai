"""Phase V3-3 orchestrator: Full Universe Dataset -> WFO windows -> per-
window train/predict (frozen Model A/B/C) -> pooled-OOS ranking/Top-N/
robustness battery. Depth is TIERED (spec section 3/4's Primary vs
Secondary framing, mirroring Phase V2-2's own "tiered analysis depth"
precedent - a pre-run design decision, not a runtime shortcut):

  TIER 1 (full battery): Model A on the Primary combination
  (target_raw_5d) - every WFO window, Q1-Q5/Rank IC/Top-N per window AND
  pooled, Regime/Year/Event breakdown, 3-method Bootstrap, Permutation,
  Concentration, Stability, Feature Importance, all 3 Benchmarks.

  TIER 2 (light battery: pooled Q1-Q5/Rank IC/Top-N/basic metrics only):
  Model A on the 3 secondary Horizons (10/15/20d) and 3 secondary Target
  Variants (TOPIX-relative/Vol-adjusted/Risk-adjusted, at 5d); Model B and
  Model C on the Primary target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtest.market_regime import compute_market_regime
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import permutation_test
from backtest.walk_forward import WalkForwardWindow
from scoring.validation import assign_quantile_buckets
from v2.config.loader import V2Config
from v3.features.registry import CORE_FEATURE_NAMES
from v3.models.data_prep import prepare_training_set
from v3.models.evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classification,
    evaluate_regression,
)
from v3.models.importance import FeatureImportance, extract_gain_and_split_importance
from v3.models.regression import fit_regression_model
from v3.targets.registry import HORIZONS
from v3.validation.benchmarks import (
    compute_v2_score_panel,
    momentum_benchmark_predictions,
    random_benchmark_predictions,
    v2_score_benchmark_predictions,
)
from v3.validation.concentration import compute_prediction_concentration
from v3.validation.feature_importance_agg import (
    AggregatedImportance,
    aggregate_importance_across_windows,
)
from v3.validation.quantile_eval import QuantileCalibrationResult, evaluate_quantile_calibration
from v3.validation.ranking_metrics import RankingResult, evaluate_ranking
from v3.validation.regime_year_event import (
    SliceResult,
    analyze_by_regime,
    analyze_by_year,
    analyze_event_exclusion,
)
from v3.validation.robustness import (
    BucketPermutationResult,
    SingleGroupBootstrapBattery,
    SpreadBootstrapBattery,
    bootstrap_daily_return_series,
    bootstrap_q5_q1_spread,
    run_bucket_permutation_tests,
)
from v3.validation.stability import StabilitySummary, summarize_stability
from v3.validation.topn_portfolio import (
    TopNPortfolioMetrics,
    compute_topn_daily_returns,
    compute_topn_portfolio_metrics,
)
from v3.validation.train_predict import (
    WindowPrediction,
    run_model_a_on_window,
    run_model_b_on_window,
    run_model_c_on_window,
    slice_window,
)
from v3.validation.wfo_config import (
    FDR_SWEEP_PERMUTATION_CONFIG,
    PERMUTATION_CONFIG,
    PRIMARY_TARGET_COL,
    SECONDARY_HORIZON_TARGETS,
    SECONDARY_VARIANT_TARGETS,
    TOP_N_VALUES,
)
from v3.validation.windows import get_v3_3_windows

logger = logging.getLogger(__name__)


def _pool(window_predictions: list[WindowPrediction]) -> pd.DataFrame:
    return pd.concat([wp.predictions for wp in window_predictions], ignore_index=True)


@dataclass(frozen=True)
class LightTargetResult:
    target_col: str
    model_type: str
    n_windows: int
    pooled_predictions: pd.DataFrame = field(repr=False)
    pooled_ranking: RankingResult
    topn: dict[int, TopNPortfolioMetrics]
    regression_metrics: RegressionMetrics | None = None
    classification_metrics: ClassificationMetrics | None = None


@dataclass(frozen=True)
class PrimaryResult:
    target_col: str
    per_window: list[WindowPrediction]
    per_window_ranking: list[RankingResult]
    pooled_ranking: RankingResult
    topn: dict[int, TopNPortfolioMetrics]
    regression_metrics: RegressionMetrics
    regime: list[SliceResult]
    year: dict[int, SliceResult]
    event: dict[str, SliceResult]
    spread_bootstrap: SpreadBootstrapBattery
    top5_bootstrap: SingleGroupBootstrapBattery
    permutation: list[BucketPermutationResult]
    concentration: object
    feature_importance: list[AggregatedImportance]
    window_stability: StabilitySummary
    year_stability: StabilitySummary
    regime_stability: StabilitySummary


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    pooled_ranking: RankingResult
    topn: dict[int, TopNPortfolioMetrics]


@dataclass(frozen=True)
class V3_3Report:
    n_tickers: int
    windows: list[WalkForwardWindow]
    primary: PrimaryResult
    secondary: dict[str, LightTargetResult]
    quantile_calibration: QuantileCalibrationResult
    benchmarks: dict[str, BenchmarkResult]
    fdr_results: dict[str, FDRResult] = field(default_factory=dict)


def _target_horizon(target_col: str) -> int:
    for h in HORIZONS:
        if target_col.endswith(f"_{h}d"):
            return h
    raise ValueError(f"cannot infer horizon from {target_col}")


def _run_topn_battery(
    predictions: pd.DataFrame, target_col: str
) -> dict[int, TopNPortfolioMetrics]:
    window_days = _target_horizon(target_col)
    return {
        n: compute_topn_portfolio_metrics(predictions, n, "actual", window_days)
        for n in TOP_N_VALUES
    }


def run_primary_analysis(
    dataset: pd.DataFrame, windows: list[WalkForwardWindow], topix_ohlcv: pd.DataFrame,
    market_regime_config,
) -> PrimaryResult:
    target_col = PRIMARY_TARGET_COL
    horizon = _target_horizon(target_col)

    per_window: list[WindowPrediction] = []
    per_window_ranking: list[RankingResult] = []
    per_window_importance: list[list[FeatureImportance]] = []
    for window in windows:
        logger.info("V3-3 primary: window %d train/predict (Model A, %s)", window.index, target_col)
        wp = run_model_a_on_window(dataset, window, target_col)
        per_window.append(wp)
        per_window_ranking.append(evaluate_ranking(wp.predictions, horizon))

        # Feature importance re-fit per window (train_predict.py doesn't
        # expose the fitted model - refit once more here on the SAME
        # window/target, deterministic given the same seed, so this is
        # NOT a second independent training run's worth of new
        # information, just re-exposing what fit_regression_model()
        # already computed).
        train, _oos = slice_window(dataset, window)
        train_set = prepare_training_set(train, target_col)
        model = fit_regression_model(train_set.X, train_set.y)
        per_window_importance.append(extract_gain_and_split_importance(model, CORE_FEATURE_NAMES))

    pooled = _pool(per_window)
    pooled_ranking = evaluate_ranking(pooled, horizon)
    topn = _run_topn_battery(pooled, target_col)
    regression_metrics = evaluate_regression(
        pooled["actual"].to_numpy(), pooled["prediction"].to_numpy()
    )

    regime_df = compute_market_regime(topix_ohlcv, market_regime_config)
    regime = analyze_by_regime(pooled, regime_df, horizon)
    year = analyze_by_year(pooled, horizon)
    event = analyze_event_exclusion(pooled, horizon)

    spread_bootstrap = bootstrap_q5_q1_spread(pooled)
    top5_daily = pooled.dropna(subset=["prediction", "actual"])
    top5_series = compute_topn_daily_returns(top5_daily, 5, "actual", score_col="prediction")
    top5_df = pd.DataFrame(
        [(d.date, d.equal_weight_return) for d in top5_series if d.equal_weight_return is not None],
        columns=["date", "return"],
    )
    top5_bootstrap = bootstrap_daily_return_series(top5_df["return"], top5_df["date"])

    permutation = run_bucket_permutation_tests(pooled, config=PERMUTATION_CONFIG)
    concentration = compute_prediction_concentration(pooled, horizon)
    feature_importance = aggregate_importance_across_windows(per_window_importance)

    window_stability = summarize_stability(
        "window_q5_q1_spread", [r.q5_q1_spread for r in per_window_ranking]
    )
    year_stability = summarize_stability(
        "year_q5_q1_spread", [s.q5_q1_spread for s in year.values()]
    )
    regime_stability = summarize_stability(
        "regime_q5_q1_spread", [s.q5_q1_spread for s in regime]
    )

    return PrimaryResult(
        target_col=target_col, per_window=per_window, per_window_ranking=per_window_ranking,
        pooled_ranking=pooled_ranking, topn=topn, regression_metrics=regression_metrics,
        regime=regime, year=year, event=event, spread_bootstrap=spread_bootstrap,
        top5_bootstrap=top5_bootstrap, permutation=permutation, concentration=concentration,
        feature_importance=feature_importance, window_stability=window_stability,
        year_stability=year_stability, regime_stability=regime_stability,
    )


def run_secondary_target(
    dataset: pd.DataFrame, windows: list[WalkForwardWindow], target_col: str
) -> LightTargetResult:
    horizon = _target_horizon(target_col)
    per_window = []
    for window in windows:
        logger.info("V3-3 secondary A: window %d/%s", window.index, target_col)
        per_window.append(run_model_a_on_window(dataset, window, target_col))
    pooled = _pool(per_window)
    pooled_ranking = evaluate_ranking(pooled, horizon)
    topn = _run_topn_battery(pooled, target_col)
    regression_metrics = evaluate_regression(
        pooled["actual"].to_numpy(), pooled["prediction"].to_numpy()
    )
    return LightTargetResult(
        target_col=target_col, model_type="A", n_windows=len(per_window),
        pooled_predictions=pooled, pooled_ranking=pooled_ranking, topn=topn,
        regression_metrics=regression_metrics,
    )


def run_model_b_light(
    dataset: pd.DataFrame, windows: list[WalkForwardWindow]
) -> LightTargetResult:
    target_col = PRIMARY_TARGET_COL
    horizon = _target_horizon(target_col)
    per_window = []
    for window in windows:
        logger.info("V3-3 secondary B: window %d/%s", window.index, target_col)
        per_window.append(run_model_b_on_window(dataset, window, target_col))
    pooled = _pool(per_window)
    pooled_ranking = evaluate_ranking(pooled, horizon)
    topn = _run_topn_battery(pooled, target_col)
    classification_metrics = evaluate_classification(
        pooled["actual_binary"].to_numpy(), pooled["prediction"].to_numpy()
    )
    return LightTargetResult(
        target_col=target_col, model_type="B", n_windows=len(per_window),
        pooled_predictions=pooled, pooled_ranking=pooled_ranking, topn=topn,
        classification_metrics=classification_metrics,
    )


def run_model_c_light(
    dataset: pd.DataFrame, windows: list[WalkForwardWindow]
) -> tuple[LightTargetResult, QuantileCalibrationResult]:
    target_col = PRIMARY_TARGET_COL
    horizon = _target_horizon(target_col)
    per_window = []
    for window in windows:
        logger.info("V3-3 secondary C: window %d/%s", window.index, target_col)
        per_window.append(run_model_c_on_window(dataset, window, target_col))
    pooled = _pool(per_window)
    pooled_ranking = evaluate_ranking(pooled, horizon)
    topn = _run_topn_battery(pooled, target_col)
    regression_metrics = evaluate_regression(
        pooled["actual"].to_numpy(), pooled["prediction"].to_numpy()
    )
    calibration = evaluate_quantile_calibration(pooled)
    result = LightTargetResult(
        target_col=target_col, model_type="C", n_windows=len(per_window),
        pooled_predictions=pooled, pooled_ranking=pooled_ranking, topn=topn,
        regression_metrics=regression_metrics,
    )
    return result, calibration


def _q5_permutation_p(predictions: pd.DataFrame, config) -> float | None:
    valid = predictions.dropna(subset=["prediction", "actual"]).copy()
    if valid.empty:
        return None
    valid["_bucket"] = assign_quantile_buckets(valid["prediction"])
    population = valid["actual"].dropna().to_numpy()
    q5_values = valid.loc[valid["_bucket"] == "Q5", "actual"].dropna().to_numpy()
    p = permutation_test(q5_values, population, config).p_value
    return p if p == p else None  # NaN check


def run_v3_3_validation(
    dataset: pd.DataFrame, tickers: list[str], v2_config: V2Config, topix_ohlcv: pd.DataFrame,
    market_regime_config, windows: list[WalkForwardWindow] | None = None,
) -> V3_3Report:
    """windows defaults to get_v3_3_windows() (this Phase's pre-registered
    Full Universe WFO windows) - the override exists only so tests can
    pass a smaller synthetic-data-compatible window list; the real Full
    Universe run never passes it.
    """
    if windows is None:
        windows = get_v3_3_windows()
    n_tickers = int(dataset["ticker"].nunique())
    horizon = _target_horizon(PRIMARY_TARGET_COL)

    logger.info("V3-3: primary analysis (Model A, %s)", PRIMARY_TARGET_COL)
    primary = run_primary_analysis(dataset, windows, topix_ohlcv, market_regime_config)
    primary_pooled = _pool(primary.per_window)

    secondary: dict[str, LightTargetResult] = {}
    for target_col in (*SECONDARY_HORIZON_TARGETS, *SECONDARY_VARIANT_TARGETS):
        logger.info("V3-3: secondary target %s (Model A)", target_col)
        secondary[target_col] = run_secondary_target(dataset, windows, target_col)

    logger.info("V3-3: Model B (classification)")
    secondary["model_b"] = run_model_b_light(dataset, windows)
    logger.info("V3-3: Model C (quantile)")
    model_c_result, quantile_calibration = run_model_c_light(dataset, windows)
    secondary["model_c"] = model_c_result

    logger.info("V3-3: benchmarks (Random / Momentum / V2 Score)")
    oos_key = primary_pooled[["date", "ticker"]].drop_duplicates()
    oos_dataset_slice = dataset.merge(oos_key, on=["date", "ticker"], how="inner")

    random_bench = random_benchmark_predictions(primary_pooled)
    momentum_bench = momentum_benchmark_predictions(oos_dataset_slice, PRIMARY_TARGET_COL)
    v2_score_panel = compute_v2_score_panel(v2_config, tickers)
    v2_bench = v2_score_benchmark_predictions(oos_dataset_slice, v2_score_panel, PRIMARY_TARGET_COL)

    benchmarks: dict[str, BenchmarkResult] = {}
    for label, bench_predictions in (
        ("random", random_bench), ("momentum", momentum_bench), ("v2_score", v2_bench),
    ):
        benchmarks[label] = BenchmarkResult(
            label=label,
            pooled_ranking=evaluate_ranking(bench_predictions, horizon),
            topn={
                n: compute_topn_portfolio_metrics(bench_predictions, n, "actual", horizon)
                for n in TOP_N_VALUES
            },
        )

    logger.info("V3-3: FDR family (Horizon/Variant/Model/Regime/Top-N)")
    p_values: dict[str, float] = {}
    for bp in primary.permutation:
        p_values[f"horizon:{PRIMARY_TARGET_COL}:{bp.bucket_label}"] = bp.result.p_value

    for target_col, result in secondary.items():
        if target_col in ("model_b", "model_c"):
            continue
        p = _q5_permutation_p(result.pooled_predictions, FDR_SWEEP_PERMUTATION_CONFIG)
        if p is not None:
            p_values[f"target:{target_col}:Q5"] = p

    for model_label in ("model_b", "model_c"):
        model_predictions = secondary[model_label].pooled_predictions
        p = _q5_permutation_p(model_predictions, FDR_SWEEP_PERMUTATION_CONFIG)
        if p is not None:
            p_values[f"model:{model_label}:Q5"] = p

    regime_df = compute_market_regime(topix_ohlcv, market_regime_config)
    primary_with_regime = primary_pooled.merge(regime_df, on="date", how="left")
    for regime_name in ("BULL", "NEUTRAL", "BEAR"):
        subset = primary_with_regime[primary_with_regime["regime"] == regime_name]
        p = _q5_permutation_p(subset, FDR_SWEEP_PERMUTATION_CONFIG)
        if p is not None:
            p_values[f"regime:{regime_name}:Q5"] = p

    full_population = primary_pooled["actual"].dropna().to_numpy()
    for n in TOP_N_VALUES:
        topn_returns = pd.Series(
            [d.equal_weight_return for d in primary.topn[n].base.daily_returns], dtype=float
        ).dropna().to_numpy()
        p = permutation_test(topn_returns, full_population, FDR_SWEEP_PERMUTATION_CONFIG).p_value
        if p == p:
            p_values[f"topn:{n}"] = p

    fdr_results = benjamini_hochberg_correction(p_values)

    return V3_3Report(
        n_tickers=n_tickers, windows=windows, primary=primary, secondary=secondary,
        quantile_calibration=quantile_calibration, benchmarks=benchmarks, fdr_results=fdr_results,
    )
