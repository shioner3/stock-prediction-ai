"""Phase 6 Walk Forward + Statistical Validation orchestration.

Ties together backtest/walk_forward.py (window generation),
backtest/costs.py, backtest/bootstrap.py, backtest/permutation.py,
backtest/market_regime.py, backtest/decision.py, and scoring/validation.py
(reused unmodified from Phase 5).

Every Signal/Score/Backtest computation is done EXACTLY ONCE, over each
ticker's FULL date range - by reusing pipeline.run_backtest.run_backtest()
and pipeline.run_score_validation.build_scored_with_targets() verbatim.
Walk Forward windows are applied only as a DATE FILTER on the resulting
Trade/Score records afterward; no Signal/Feature/Score is ever
recomputed per-window. This is deliberate: Phase 2-5's no-lookahead
tests already prove Feature(t)/Signal(t)/Score(t) depend only on data
<= t, so slicing by date after one full computation is guaranteed
identical to recomputing on a truncated dataset - see
tests/test_walk_forward_no_lookahead.py's Test B for the direct check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.bootstrap import (
    STATISTIC_NAMES,
    BootstrapDiffResult,
    BootstrapResult,
    bootstrap_ci,
    bootstrap_diff_ci,
)
from backtest.costs import apply_cost
from backtest.cross_sectional import ConcentrationMetrics, compute_concentration
from backtest.decision import Decision, DecisionInputs, classify
from backtest.market_regime import compute_market_regime
from backtest.metrics import BacktestMetrics, compute_metrics, compute_metrics_by_ticker
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import PermutationResult, permutation_test
from backtest.walk_forward import WalkForwardWindow, generate_windows
from common.hashing import hash_files
from config.loader import AppConfig
from pipeline.run_backtest import run_backtest
from pipeline.run_score_validation import build_scored_with_targets
from scoring.validation import (
    FIXED_BUCKET_LABELS,
    QUANTILE_BUCKET_LABELS,
    assign_fixed_buckets,
    assign_quantile_buckets,
    bucket_ratio,
    bucket_spread,
    check_monotonicity,
    compute_bucket_metrics,
    extract_bucket_returns,
    monotonicity_correlation,
)
from storage.parquet_store import load_feature_panel, load_ohlcv
from targets.forward_returns import FORWARD_WINDOWS, compute_forward_returns

logger = logging.getLogger(__name__)

CONFIG_FILES = [Path("config/settings.yaml"), Path("config/universe_filters.yaml")]


@dataclass
class WindowMetrics:
    window: WalkForwardWindow
    train_metrics: BacktestMetrics
    validation_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    expectancy_degradation_train_to_oos: float | None


@dataclass
class SignalWalkForwardResult:
    direction: str
    signal_name: str
    windows: list[WindowMetrics]
    oos_metrics_by_cost_tier: dict[str, BacktestMetrics]
    bootstrap: dict[str, BootstrapResult]
    permutation: PermutationResult | None
    regime_metrics: dict[str, BacktestMetrics]
    windows_with_positive_pf: int
    windows_with_positive_expectancy: int
    windows_evaluated: int
    decision: Decision
    # Phase 6.5 section 17/18 - descriptive-only breakdown of the SAME
    # aggregate_oos_trades already used above; does not affect `decision`.
    ticker_metrics: dict[str, BacktestMetrics]
    concentration: ConcentrationMetrics


@dataclass
class ScoreOOSResult:
    direction: str
    signal_name: str
    forward_window: int
    bucket_scheme: str  # "fixed" or "quantile"
    n: int
    bucket_metrics: dict[str, BacktestMetrics]
    monotonic: bool | None
    monotonicity_corr: float | None
    # Phase 6.5 section 21 - only populated for bucket_scheme="quantile"
    # (the "Q5-Q1" naming is quantile-bucket-specific; a Fixed-bucket
    # spread would compare arbitrary-width score ranges, not population
    # quintiles). None on the "fixed" rows.
    q5_q1_spread: float | None = None
    q5_q1_ratio: float | None = None
    q5_bootstrap: BootstrapResult | None = None
    q5_q1_bootstrap_diff: BootstrapDiffResult | None = None


@dataclass
class WalkForwardReport:
    windows: list[WalkForwardWindow]
    signal_results: list[SignalWalkForwardResult]
    score_results: list[ScoreOOSResult]
    config_hash: str
    data_hash: str
    tickers: list[str]
    # Phase 6.5 section 14 - INFORMATIONAL ONLY: keyed "{direction}:
    # {signal_name}", built from each signal's raw permutation p-value.
    # Never read by backtest/decision.py's classify() - each
    # SignalWalkForwardResult.decision above is computed independently,
    # using only the raw (uncorrected) p-value, exactly as Phase 6 did.
    fdr_results: dict[str, FDRResult] = field(default_factory=dict)


def _oos_mask(dates: pd.Series, windows: list[WalkForwardWindow]) -> pd.Series:
    mask = pd.Series(False, index=dates.index)
    for w in windows:
        mask |= (dates >= w.oos_start) & (dates <= w.oos_end)
    return mask


def _forward_return_population(config: AppConfig, tickers: list[str]) -> pd.DataFrame:
    """ticker/date/forward_return_{n}d for EVERY row of every ticker's
    Feature panel (not just Signal-triggered rows) - the sampling pool
    the Permutation Test draws its null distribution from.
    """
    features_dir = Path(config.data.features_dir)
    frames = []
    for ticker in tickers:
        panel = load_feature_panel(ticker, features_dir)
        forward = compute_forward_returns(panel)
        frame = panel[["date"]].join(forward)
        frame.insert(0, "ticker", ticker)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compute_window_metrics(
    trades: pd.DataFrame, window: WalkForwardWindow, base_cost_bps: float
) -> WindowMetrics:
    train = trades[
        (trades["signal_date"] >= window.train_start) & (trades["signal_date"] < window.train_end)
    ]
    val = trades[
        (trades["signal_date"] >= window.validation_start)
        & (trades["signal_date"] < window.validation_end)
    ]
    oos = trades[
        (trades["signal_date"] >= window.oos_start) & (trades["signal_date"] <= window.oos_end)
    ]

    def _metrics(subset: pd.DataFrame) -> BacktestMetrics:
        costed = subset.assign(**{"return": subset["return"] - base_cost_bps / 10_000.0})
        return compute_metrics(costed)

    train_metrics = _metrics(train)
    val_metrics = _metrics(val)
    oos_metrics = _metrics(oos)

    degradation = None
    if train_metrics.expectancy is not None and oos_metrics.expectancy is not None:
        degradation = oos_metrics.expectancy - train_metrics.expectancy

    return WindowMetrics(
        window=window,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        oos_metrics=oos_metrics,
        expectancy_degradation_train_to_oos=degradation,
    )


def _compute_signal_result(
    direction: str,
    signal_name: str,
    trades: pd.DataFrame,
    windows: list[WalkForwardWindow],
    regime_df: pd.DataFrame,
    permutation_signal_returns: pd.Series,
    permutation_population: np.ndarray,
    config: AppConfig,
) -> SignalWalkForwardResult:
    signal_trades = trades[
        (trades["direction"] == direction) & (trades["signal_name"] == signal_name)
    ]

    base_tier = next(
        t for t in config.validation.transaction_cost.tiers if t.name == "base"
    )
    window_metrics = [
        _compute_window_metrics(signal_trades, w, base_tier.round_trip_bps) for w in windows
    ]

    windows_with_trades = [
        wm for wm in window_metrics if wm.oos_metrics.n_trades > 0
    ]
    positive_pf = sum(
        1 for wm in windows_with_trades
        if wm.oos_metrics.profit_factor is not None and wm.oos_metrics.profit_factor > 1.0
    )
    positive_expectancy = sum(
        1 for wm in windows_with_trades
        if wm.oos_metrics.expectancy is not None and wm.oos_metrics.expectancy > 0
    )

    oos_mask = _oos_mask(signal_trades["signal_date"], windows)
    aggregate_oos_trades = signal_trades[oos_mask]

    metrics_by_tier: dict[str, BacktestMetrics] = {}
    for tier in config.validation.transaction_cost.tiers:
        costed_returns = apply_cost(aggregate_oos_trades["return"], tier)
        costed = aggregate_oos_trades.assign(**{"return": costed_returns})
        metrics_by_tier[tier.name] = compute_metrics(costed)

    base_returns = apply_cost(aggregate_oos_trades["return"], base_tier)
    bootstrap_results = {
        name: bootstrap_ci(base_returns.to_numpy(), name, config.validation.bootstrap)
        for name in STATISTIC_NAMES
    }

    permutation_result: PermutationResult | None = None
    if len(permutation_signal_returns) > 0:
        permutation_result = permutation_test(
            permutation_signal_returns.to_numpy(),
            permutation_population,
            config.validation.permutation,
        )

    regime_metrics: dict[str, BacktestMetrics] = {}
    if not aggregate_oos_trades.empty:
        merged = aggregate_oos_trades.merge(
            regime_df, left_on="signal_date", right_on="date", how="left"
        )
        for regime_name, group in merged.groupby("regime", dropna=True):
            regime_metrics[str(regime_name)] = compute_metrics(group)

    aggregate_metrics = metrics_by_tier["base"]
    decision_inputs = DecisionInputs(
        oos_trade_count=aggregate_metrics.n_trades,
        min_oos_trades=config.validation.min_sample.min_oos_trades,
        total_windows=len(windows_with_trades),
        windows_with_positive_pf=positive_pf,
        aggregate_oos_expectancy=aggregate_metrics.expectancy,
        aggregate_oos_profit_factor=aggregate_metrics.profit_factor,
        expectancy_ci_low=bootstrap_results["expectancy"].ci_low
        if bootstrap_results["expectancy"].n_observations > 0
        else None,
        permutation_p_value=permutation_result.p_value if permutation_result else None,
        high_cost_expectancy=metrics_by_tier["high"].expectancy,
    )
    decision = classify(decision_inputs)

    ticker_metrics = compute_metrics_by_ticker(aggregate_oos_trades)
    concentration = compute_concentration(aggregate_oos_trades)

    return SignalWalkForwardResult(
        direction=direction,
        signal_name=signal_name,
        windows=window_metrics,
        oos_metrics_by_cost_tier=metrics_by_tier,
        bootstrap=bootstrap_results,
        permutation=permutation_result,
        regime_metrics=regime_metrics,
        windows_with_positive_pf=positive_pf,
        windows_with_positive_expectancy=positive_expectancy,
        windows_evaluated=len(windows_with_trades),
        decision=decision,
        ticker_metrics=ticker_metrics,
        concentration=concentration,
    )


def _compute_score_oos_results(
    scored_with_targets: pd.DataFrame, windows: list[WalkForwardWindow], config: AppConfig
) -> list[ScoreOOSResult]:
    if scored_with_targets.empty:
        return []

    oos_mask = _oos_mask(scored_with_targets["date"], windows)
    oos_scored = scored_with_targets[oos_mask].copy()
    if oos_scored.empty:
        return []

    oos_scored["score_bucket_fixed"] = assign_fixed_buckets(oos_scored["total_score"])
    oos_scored["score_bucket_quantile"] = assign_quantile_buckets(oos_scored["total_score"])

    # (scheme name, bucket column, bucket order low->high, high label, low label)
    bucket_schemes = [
        ("fixed", "score_bucket_fixed", FIXED_BUCKET_LABELS),
        ("quantile", "score_bucket_quantile", QUANTILE_BUCKET_LABELS),
    ]

    results = []
    for (direction, signal_name), group in oos_scored.groupby(["direction", "signal_name"]):
        for window in FORWARD_WINDOWS:
            return_col = f"forward_return_{window}d"
            for scheme_name, bucket_col, bucket_order in bucket_schemes:
                bucket_metrics = compute_bucket_metrics(group, bucket_col, return_col)
                high_label, low_label = bucket_order[-1], bucket_order[0]

                q5_q1_spread = None
                q5_q1_ratio = None
                q5_bootstrap = None
                q5_q1_bootstrap_diff = None
                if scheme_name == "quantile":
                    q5_q1_spread = bucket_spread(bucket_metrics, high_label, low_label)
                    q5_q1_ratio = bucket_ratio(bucket_metrics, high_label, low_label)
                    high_returns = extract_bucket_returns(
                        group, bucket_col, return_col, high_label
                    )
                    low_returns = extract_bucket_returns(group, bucket_col, return_col, low_label)
                    if len(high_returns) > 0:
                        q5_bootstrap = bootstrap_ci(
                            high_returns, "mean_return", config.validation.bootstrap
                        )
                    if len(high_returns) > 0 and len(low_returns) > 0:
                        q5_q1_bootstrap_diff = bootstrap_diff_ci(
                            high_returns, low_returns, config.validation.bootstrap
                        )

                results.append(
                    ScoreOOSResult(
                        direction=direction,
                        signal_name=signal_name,
                        forward_window=window,
                        bucket_scheme=scheme_name,
                        n=len(group),
                        bucket_metrics=bucket_metrics,
                        monotonic=check_monotonicity(bucket_metrics, bucket_order),
                        monotonicity_corr=monotonicity_correlation(bucket_metrics, bucket_order),
                        q5_q1_spread=q5_q1_spread,
                        q5_q1_ratio=q5_q1_ratio,
                        q5_bootstrap=q5_bootstrap,
                        q5_q1_bootstrap_diff=q5_q1_bootstrap_diff,
                    )
                )
    return results


def run_walk_forward(
    config: AppConfig, tickers: list[str] | None = None, min_oos_start: date | None = None
) -> WalkForwardReport:
    """min_oos_start (Phase 7 section 3/18): an optional REPORTING-SCOPE
    filter, applied AFTER generate_windows() - the pure WFO generator
    itself is completely unchanged and still walks forward from
    data_start across the FULL fetched range, so a window whose OOS
    starts on/after min_oos_start still gets a real, non-look-ahead
    TRAIN/VALIDATION built from genuine prior data. Windows with
    oos_start before min_oos_start are simply excluded from this
    report's aggregate signal/score metrics (see
    tests/test_pipeline_run_walk_forward.py for the direct proof that
    TRAIN/VALIDATION construction and window DATES are unaffected -
    only which windows get aggregated into the OOS report changes).
    Default None reproduces Phase 6's/6.5's original behavior exactly
    (no filtering).
    """
    features_dir = Path(config.data.features_dir)

    if tickers is None:
        tickers = sorted(p.stem for p in features_dir.glob("*.parquet"))

    data_start: date | None = None
    data_end: date | None = None
    data_files: list[Path] = []
    for ticker in tickers:
        panel = load_feature_panel(ticker, features_dir)
        ticker_start, ticker_end = panel["date"].min(), panel["date"].max()
        data_start = ticker_start if data_start is None else min(data_start, ticker_start)
        data_end = ticker_end if data_end is None else max(data_end, ticker_end)
        data_files.append(features_dir / f"{ticker}.parquet")

    if data_start is None or data_end is None:
        return WalkForwardReport([], [], [], "", "", tickers)

    windows = generate_windows(data_start, data_end, config.validation.walk_forward)
    if min_oos_start is not None:
        windows = [w for w in windows if w.oos_start >= min_oos_start]
    logger.info("walk forward: %d windows generated (%s .. %s)", len(windows), data_start, data_end)

    backtest_summary = run_backtest(config, tickers=tickers)
    trades = backtest_summary.trades

    scored_with_targets = build_scored_with_targets(config, tickers=tickers)

    topix = load_ohlcv("TOPIX", config.data.processed_dir)
    regime_df = compute_market_regime(topix, config.validation.market_regime)

    population_df = _forward_return_population(config, tickers)
    perm_window = config.validation.permutation.forward_window
    population_col = f"forward_return_{perm_window}d"
    if population_df.empty:
        population_values = np.array([])
    else:
        oos_population_mask = _oos_mask(population_df["date"], windows)
        population_values = (
            population_df.loc[oos_population_mask, population_col].dropna().to_numpy()
        )

    signal_pairs = trades[["direction", "signal_name"]].drop_duplicates()
    signal_names = sorted(signal_pairs.itertuples(index=False))
    signal_results = []
    for direction, signal_name in signal_names:
        if not scored_with_targets.empty:
            sig_scored = scored_with_targets[
                (scored_with_targets["direction"] == direction)
                & (scored_with_targets["signal_name"] == signal_name)
            ]
            oos_sig_mask = _oos_mask(sig_scored["date"], windows)
            permutation_signal_returns = sig_scored.loc[oos_sig_mask, population_col].dropna()
        else:
            permutation_signal_returns = pd.Series(dtype=float)

        result = _compute_signal_result(
            direction, signal_name, trades, windows, regime_df,
            permutation_signal_returns, population_values, config,
        )
        signal_results.append(result)

    score_results = _compute_score_oos_results(scored_with_targets, windows, config)

    permutation_p_values = {
        f"{r.direction}:{r.signal_name}": r.permutation.p_value
        for r in signal_results
        if r.permutation is not None
    }
    fdr_results = benjamini_hochberg_correction(permutation_p_values)

    config_hash = hash_files(CONFIG_FILES)
    data_hash = hash_files(data_files)

    return WalkForwardReport(
        windows=windows,
        signal_results=signal_results,
        score_results=score_results,
        config_hash=config_hash,
        data_hash=data_hash,
        tickers=tickers,
        fdr_results=fdr_results,
    )
