"""Phase 13: long_oversold_rebound Conditional Analysis.

Answers ONE question: under what market/stock/Score conditions does
long_oversold_rebound's Forward Return look strongest, and does that
hold up once the 2024-08 event and BEAR-episode concentration Phase 8/9
already found are accounted for? This is exploratory sub-group analysis
of an already-frozen Signal's own historical occurrences - never a
Signal/Score/Backtest change, never a new Signal, never a threshold
decision (spec section 1/2/29).

Every bucket boundary in this module is FIXED here, before any real run
against Full Universe data, and never revisited after seeing results
(spec section 7-11's "事前固定"):

- Market Drawdown (TOPIX 20d return): >=0% / -5%~0% / -10%~-5% / <-10%
  (spec section 7's own example, adopted verbatim).
- Individual Stock Drawdown (return_20d): >0% / 0%~-5% / -5%~-10% /
  -10%~-20% / <-20% (spec section 8's own example, adopted verbatim).
- MA20 deviation (close_to_sma_20): >0% / 0%~-5% / -5%~-10% / <=-10%
  (spec section 9's own example, adopted verbatim).
- Volume (volume_ratio_20d): <0.8x below-normal / 0.8-1.2x normal /
  1.2-2.0x increased / >=2.0x surge - a concrete operationalization of
  spec section 10's qualitative categories, fixed here.
- Volatility (volatility_20d): tercile quantile split (bottom/middle/
  top third) of the SIGNAL population's own distribution - same
  "quantile-bucket" convention scoring/validation.py's Score Q1-Q5
  buckets already use elsewhere in this project (the RULE - a 3-way
  quantile split - is fixed before running; only the resulting cutoff
  VALUES are data-derived, exactly like every existing Score quantile
  bucket in this codebase).

TOPIX N-day return (spec section 7's own example dimension) is NOT a
stored Feature-panel column (features/pipeline.py deliberately never
runs compute_feature_panel() on the market index itself - see
pipeline/build_features.py's _NON_TICKER_FILES). It is computed here by
calling features/momentum.py::compute_return() - the exact same,
unmodified formula every ticker's own return_Nd Feature already uses -
on TOPIX's raw Close series. This is reuse of an existing function for
analysis purposes, not a new Feature or new calculation logic.

Analysis depth is tiered to keep runtime bounded (mirroring Phase 12's
precedent): the PRIMARY bucketing dimension named in each spec example
(TOPIX 20d return, stock 20d return, MA20 deviation) gets the full
statistical battery (Forward Return stats, MFE/MAE, trade-level cost/
PF/expectancy, Bootstrap, Block Bootstrap). Secondary windows for the
same underlying concept (e.g. TOPIX 5d/10d/60d return, stock 5d/10d/60d
return, MA5/MA50 deviation) get descriptive-only stats (n, mean, median,
win rate) - this tiering is fixed here, before running, not chosen
after seeing which windows looked interesting.

Reuse: run_backtest() and the Forward Return/MFE-MAE panel are each
computed ONCE for the full Universe and reused across every axis below.
Bootstrap/Block Bootstrap/Permutation/FDR/Regime/Episode/Cost primitives
are the exact same Phase 6-9/12 code, unmodified. Signal Count (spec
section 13) reuses Phase 12's ensemble.signal_count.aggregate_signal_counts()
directly rather than recomputing co-occurrence.

Dependency direction: this module imports FROM signals/backtest/targets/
ensemble, and must never be imported BY them - see
tests/test_phase9_no_lookahead.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.block_bootstrap import BlockBootstrapResult, block_bootstrap
from backtest.bootstrap import BootstrapResult, bootstrap_ci
from backtest.costs import apply_cost
from backtest.episode_analysis import (
    RichEpisodeMetrics,
    compute_rich_episode_metrics,
    identify_regime_episodes,
)
from backtest.market_regime import compute_market_regime
from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.permutation import PermutationResult, permutation_test
from config.loader import (
    AppConfig,
    BootstrapConfig,
    PermutationConfig,
    Phase9Config,
    TransactionCostTierConfig,
    load_phase9_config,
)
from ensemble.signal_count import aggregate_signal_counts
from features.momentum import compute_return
from pipeline.run_backtest import run_backtest
from pipeline.run_phase8_analysis import ConfigCheckResult, ConfigMismatchError, verify_config_hash
from pipeline.run_phase9_analysis import AUG_2024_EVENT_END, AUG_2024_EVENT_START
from pipeline.run_phase12_ensemble import load_all_signal_records, load_forward_return_panel
from scoring.validation import assign_quantile_buckets
from storage.parquet_store import (
    load_feature_panel,
    load_ohlcv,
    load_score_records,
)
from targets.forward_returns import FORWARD_WINDOWS, compute_mfe_mae

logger = logging.getLogger(__name__)

TARGET_DIRECTION = "LONG"
TARGET_SIGNAL_NAME = "long_oversold_rebound"
PERMUTATION_FORWARD_WINDOW = 5
_PERMUTATION_COL = f"forward_return_{PERMUTATION_FORWARD_WINDOW}d"

# --- Fixed bucket edges (spec section 7-11 - never revisited) ------------------

MARKET_DRAWDOWN_20D_EDGES: list[tuple[float, float, str]] = [
    (float("-inf"), -0.10, "<-10%"),
    (-0.10, -0.05, "-10%~-5%"),
    (-0.05, 0.0, "-5%~0%"),
    (0.0, float("inf"), ">=0%"),
]
MARKET_DRAWDOWN_20D_ORDER = ["<-10%", "-10%~-5%", "-5%~0%", ">=0%"]

STOCK_DRAWDOWN_20D_EDGES: list[tuple[float, float, str]] = [
    (float("-inf"), -0.20, "<-20%"),
    (-0.20, -0.10, "-20%~-10%"),
    (-0.10, -0.05, "-10%~-5%"),
    (-0.05, 0.0, "-5%~0%"),
    (0.0, float("inf"), ">0%"),
]
STOCK_DRAWDOWN_20D_ORDER = ["<-20%", "-20%~-10%", "-10%~-5%", "-5%~0%", ">0%"]

MA20_DEVIATION_EDGES: list[tuple[float, float, str]] = [
    (float("-inf"), -0.10, "<=-10%"),
    (-0.10, -0.05, "-10%~-5%"),
    (-0.05, 0.0, "-5%~0%"),
    (0.0, float("inf"), ">0%"),
]
MA20_DEVIATION_ORDER = ["<=-10%", "-10%~-5%", "-5%~0%", ">0%"]

VOLUME_RATIO_20D_EDGES: list[tuple[float, float, str]] = [
    (float("-inf"), 0.8, "below_normal(<0.8x)"),
    (0.8, 1.2, "normal(0.8-1.2x)"),
    (1.2, 2.0, "increased(1.2-2.0x)"),
    (2.0, float("inf"), "surge(>=2.0x)"),
]
VOLUME_RATIO_20D_ORDER = [
    "below_normal(<0.8x)", "normal(0.8-1.2x)", "increased(1.2-2.0x)", "surge(>=2.0x)",
]

SIGNAL_COUNT_ORDER = ["0", "1", "2", "3+"]
LONG_SHORT_CONSENSUS_ORDER = ["LONG_ONLY", "LONG_MAJORITY", "TIE", "SHORT_MAJORITY"]


def bin_label(value: float, edges: list[tuple[float, float, str]]) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    for low, high, label in edges:
        if low <= value < high:
            return label
    return None


def _signal_count_bucket(other_count: int) -> str:
    return "3+" if other_count >= 3 else str(other_count)


# --- Generic per-bucket statistical battery -------------------------------------


@dataclass(frozen=True)
class ForwardReturnStats:
    window_days: int
    n: int
    mean_return: float | None
    median_return: float | None
    win_rate: float | None
    mean_mfe: float | None
    mean_mae: float | None


@dataclass(frozen=True)
class BucketResult:
    axis: str
    label: str
    n: int
    forward_return_stats: list[ForwardReturnStats]
    cost_metrics: dict[str, BacktestMetrics] | None
    bootstrap_expectancy: BootstrapResult | None
    block_bootstrap_expectancy: BlockBootstrapResult | None
    permutation: PermutationResult | None


@dataclass(frozen=True)
class DescriptiveStats:
    axis: str
    label: str
    n: int
    mean_return_5d: float | None
    median_return_5d: float | None
    win_rate_5d: float | None


def forward_return_stats(
    keys: pd.DataFrame, target_panel: pd.DataFrame, full_windows: bool = True
) -> list[ForwardReturnStats]:
    merged = keys[["ticker", "date"]].merge(target_panel, on=["ticker", "date"], how="left")
    windows = FORWARD_WINDOWS if full_windows else (PERMUTATION_FORWARD_WINDOW,)
    stats = []
    for w in windows:
        ret_col, mfe_col, mae_col = f"forward_return_{w}d", f"mfe_{w}d", f"mae_{w}d"
        values = merged[ret_col].dropna()
        stats.append(
            ForwardReturnStats(
                window_days=w,
                n=len(values),
                mean_return=float(values.mean()) if len(values) else None,
                median_return=float(values.median()) if len(values) else None,
                win_rate=float((values > 0).mean()) if len(values) else None,
                mean_mfe=float(merged[mfe_col].dropna().mean()) if mfe_col in merged else None,
                mean_mae=float(merged[mae_col].dropna().mean()) if mae_col in merged else None,
            )
        )
    return stats


def cost_metrics(trades: pd.DataFrame, tiers) -> dict[str, BacktestMetrics]:
    return {
        tier.name: compute_metrics(trades.assign(**{"return": apply_cost(trades["return"], tier)}))
        for tier in tiers
    }


def analyze_bucket(
    axis: str,
    label: str,
    keys: pd.DataFrame,
    target_panel: pd.DataFrame,
    trades_by_key: pd.DataFrame,
    population_forward_5d: np.ndarray,
    tiers,
    bootstrap_config,
    phase9_config,
    permutation_config,
) -> BucketResult:
    n = len(keys)
    fwd_stats = forward_return_stats(keys, target_panel)

    cost_metrics_result = None
    bootstrap_expectancy = None
    block_bootstrap_expectancy = None
    permutation = None

    key_set = set(zip(keys["ticker"], keys["date"]))
    trades = trades_by_key[trades_by_key["_key"].isin(key_set)]
    if not trades.empty:
        cost_metrics_result = cost_metrics(trades, tiers)
        base_tier = next(t for t in tiers if t.name == "base")
        base_returns = apply_cost(trades["return"], base_tier)
        bootstrap_expectancy = bootstrap_ci(base_returns.to_numpy(), "expectancy", bootstrap_config)
        block_bootstrap_expectancy = block_bootstrap(
            trades, "expectancy", phase9_config.block_bootstrap
        )

    merged = keys[["ticker", "date"]].merge(target_panel, on=["ticker", "date"], how="left")
    signal_returns = merged[_PERMUTATION_COL].dropna().to_numpy()
    if len(signal_returns) > 0:
        permutation = permutation_test(signal_returns, population_forward_5d, permutation_config)

    return BucketResult(
        axis=axis, label=label, n=n, forward_return_stats=fwd_stats,
        cost_metrics=cost_metrics_result, bootstrap_expectancy=bootstrap_expectancy,
        block_bootstrap_expectancy=block_bootstrap_expectancy, permutation=permutation,
    )


def descriptive_stats(
    axis: str, label: str, keys: pd.DataFrame, target_panel: pd.DataFrame
) -> DescriptiveStats:
    merged = keys[["ticker", "date"]].merge(target_panel, on=["ticker", "date"], how="left")
    values = merged[_PERMUTATION_COL].dropna()
    return DescriptiveStats(
        axis=axis, label=label, n=len(values),
        mean_return_5d=float(values.mean()) if len(values) else None,
        median_return_5d=float(values.median()) if len(values) else None,
        win_rate_5d=float((values > 0).mean()) if len(values) else None,
    )


# --- Score analysis (reuses scoring/validation.py unchanged) -------------------


@dataclass(frozen=True)
class ScoreBucketResult:
    bucket: str
    n: int
    mean_return_5d: float | None
    metrics: BacktestMetrics


@dataclass(frozen=True)
class ScoreAnalysis:
    buckets: list[ScoreBucketResult]
    monotonic: bool | None
    rank_correlation: float | None
    q5_q1_spread: float | None


def load_score_data(tickers: list[str], scores_dir: Path) -> pd.DataFrame:
    """ticker/date/total_score rows for TARGET_DIRECTION:TARGET_SIGNAL_NAME
    only, across every ticker - extracted from Phase 13's own axis 7 so
    Phase 14's Score Independence Test (spec section 24 Q9) can reuse the
    exact same loading logic against an arbitrary subset of keys (e.g.
    BEAR x TOPIX20d<=-10% only) instead of always the full target_records.
    """
    score_frames = []
    for ticker in tickers:
        try:
            df = load_score_records(ticker, scores_dir)
        except FileNotFoundError:
            continue
        sub = df[(df["direction"] == TARGET_DIRECTION) & (df["signal_name"] == TARGET_SIGNAL_NAME)]
        if not sub.empty:
            score_frames.append(sub[["ticker", "date", "total_score"]])
    return (
        pd.concat(score_frames, ignore_index=True)
        if score_frames
        else pd.DataFrame(columns=["ticker", "date", "total_score"])
    )


def compute_score_analysis(
    keys: pd.DataFrame, target_panel: pd.DataFrame, score_data: pd.DataFrame
) -> ScoreAnalysis:
    """Q1-Q5 quantile-bucketed mean Forward Return, monotonicity, rank
    correlation, and Q5-Q1 spread for whichever `keys` subset the caller
    passes in - `keys` need not be every long_oversold_rebound occurrence
    (Phase 13 passes target_records; Phase 14 passes a market-condition
    subset for its own within-condition Score Independence Test).
    """
    keys_with_score = keys.merge(score_data, on=["ticker", "date"], how="left").dropna(
        subset=["total_score"]
    )
    score_buckets_out: list[ScoreBucketResult] = []
    if not keys_with_score.empty:
        keys_with_score = keys_with_score.assign(
            score_bucket=assign_quantile_buckets(keys_with_score["total_score"])
        )
        merged_score = keys_with_score.merge(target_panel, on=["ticker", "date"], how="left")
        for label, group in merged_score.groupby("score_bucket", dropna=True, observed=True):
            values = group[_PERMUTATION_COL].dropna()
            renamed = group.rename(columns={_PERMUTATION_COL: "return"}).dropna(subset=["return"])
            score_buckets_out.append(
                ScoreBucketResult(
                    bucket=str(label), n=len(group),
                    mean_return_5d=float(values.mean()) if len(values) else None,
                    metrics=compute_metrics(renamed),
                )
            )
    q_order = [f"Q{i + 1}" for i in range(5)]
    bucket_by_label = {b.bucket: b for b in score_buckets_out}
    means = [bucket_by_label[q].mean_return_5d for q in q_order if q in bucket_by_label]
    monotonic = None
    rank_corr = None
    if len(means) >= 2:
        monotonic = all(
            a is not None and b is not None and a <= b for a, b in zip(means, means[1:])
        )
        valid = [(rank, m) for rank, m in enumerate(means) if m is not None]
        if len(valid) >= 2 and len({m for _, m in valid}) > 1:
            ranks_valid, means_valid = zip(*valid)
            rank_corr = float(np.corrcoef(np.array(ranks_valid), np.array(means_valid))[0, 1])
    q5_q1_spread = None
    if "Q5" in bucket_by_label and "Q1" in bucket_by_label:
        q5, q1 = bucket_by_label["Q5"].mean_return_5d, bucket_by_label["Q1"].mean_return_5d
        if q5 is not None and q1 is not None:
            q5_q1_spread = q5 - q1
    return ScoreAnalysis(
        buckets=score_buckets_out, monotonic=monotonic, rank_correlation=rank_corr,
        q5_q1_spread=q5_q1_spread,
    )


# --- Top-level report -------------------------------------------------------------


@dataclass(frozen=True)
class Phase13Report:
    config_check: ConfigCheckResult
    tickers: list[str]
    n_signal_total: int
    unique_tickers: int
    unique_signal_dates: int
    overall_forward_return_stats: list[ForwardReturnStats]
    regime_buckets: list[BucketResult]
    market_drawdown_20d_buckets: list[BucketResult]
    market_drawdown_secondary: list[DescriptiveStats]
    stock_drawdown_20d_buckets: list[BucketResult]
    stock_drawdown_secondary: list[DescriptiveStats]
    ma20_deviation_buckets: list[BucketResult]
    ma_deviation_secondary: list[DescriptiveStats]
    volume_buckets: list[BucketResult]
    volatility_buckets: list[BucketResult]
    score_analysis: ScoreAnalysis
    regime_x_score: list[BucketResult]
    drawdown_x_score: list[BucketResult]
    signal_count_buckets: list[BucketResult]
    long_short_consensus_buckets: list[BucketResult]
    event_case_a: BacktestMetrics
    event_case_b_excl_aug2024: BacktestMetrics
    event_case_c_excl_2024: BacktestMetrics
    event_case_d_bear_excl_aug2024: BacktestMetrics
    bear_episodes: list[RichEpisodeMetrics]
    fdr_tested_p_values: dict[str, float]


@dataclass(frozen=True)
class TargetData:
    """Everything computed ONCE from Full Universe data for
    long_oversold_rebound's own occurrences - reused by both Phase 13
    (this module) and Phase 14 (pipeline/run_phase14_validation.py) so
    neither has to duplicate the loading/backtest/regime-computation
    logic. A pure extraction of what run_phase13_conditional_analysis()
    already did inline - no behavior change (see
    tests/test_pipeline_run_phase13_conditional_analysis.py, which still
    passes unmodified after this refactor).
    """

    all_records: pd.DataFrame
    target_records: pd.DataFrame
    target_panel: pd.DataFrame
    population_forward_5d: np.ndarray
    target_trades: pd.DataFrame
    topix: pd.DataFrame
    regime_df: pd.DataFrame
    tiers: list[TransactionCostTierConfig]
    bootstrap_config: BootstrapConfig
    permutation_config: PermutationConfig
    phase9_config: Phase9Config


def load_target_data(config: AppConfig, tickers: list[str]) -> TargetData:
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)

    logger.info("loading all Signal Records for %d tickers", len(tickers))
    all_records = load_all_signal_records(tickers, signals_dir)
    target_records = all_records[
        (all_records["direction"] == TARGET_DIRECTION)
        & (all_records["signal_name"] == TARGET_SIGNAL_NAME)
    ][["ticker", "date"]].drop_duplicates()
    if target_records.empty:
        raise RuntimeError(f"{TARGET_DIRECTION}:{TARGET_SIGNAL_NAME} never triggered")

    logger.info(
        "building Forward Return + MFE/MAE target panel for %d tickers", len(tickers)
    )
    forward_panel = load_forward_return_panel(tickers, features_dir)
    mfe_mae_frames = []
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
        except FileNotFoundError:
            continue
        mfe_mae = compute_mfe_mae(panel, TARGET_DIRECTION)
        frame = panel[["date"]].join(mfe_mae)
        frame.insert(0, "ticker", ticker)
        mfe_mae_frames.append(frame)
    mfe_mae_panel = (
        pd.concat(mfe_mae_frames, ignore_index=True) if mfe_mae_frames else pd.DataFrame()
    )
    target_panel = forward_panel.merge(mfe_mae_panel, on=["ticker", "date"], how="left")
    population_forward_5d = forward_panel[_PERMUTATION_COL].dropna().to_numpy()

    logger.info("running Combined backtest over %d tickers", len(tickers))
    backtest_summary = run_backtest(config, tickers=tickers)
    trades = backtest_summary.trades
    target_trades = trades[
        (trades["direction"] == TARGET_DIRECTION) & (trades["signal_name"] == TARGET_SIGNAL_NAME)
    ].copy()
    target_trades["_key"] = list(zip(target_trades["ticker"], target_trades["signal_date"]))

    topix = load_ohlcv("TOPIX", config.data.processed_dir)
    regime_df = compute_market_regime(topix, config.validation.market_regime)
    tiers = config.validation.transaction_cost.tiers
    bootstrap_config = config.validation.bootstrap
    permutation_config = config.validation.permutation
    phase9_config = load_phase9_config()

    return TargetData(
        all_records=all_records,
        target_records=target_records,
        target_panel=target_panel,
        population_forward_5d=population_forward_5d,
        target_trades=target_trades,
        topix=topix,
        regime_df=regime_df,
        tiers=tiers,
        bootstrap_config=bootstrap_config,
        permutation_config=permutation_config,
        phase9_config=phase9_config,
    )


TOPIX_RETURN_WINDOWS = (1, 3, 5, 10, 20, 60)


def compute_topix_returns(topix: pd.DataFrame) -> pd.DataFrame:
    """date + topix_return_{w}d for w in TOPIX_RETURN_WINDOWS - reuses
    features/momentum.py::compute_return() unmodified on TOPIX's raw
    Close series (see module docstring for why TOPIX N-day return is
    computed here rather than read from a stored Feature column).
    Extracted so Phase 14's own market-condition bucketing (TOPIX 20d
    threshold grid, dose-response bins) can reuse the exact same
    computation instead of recomputing it.
    """
    topix_returns = pd.DataFrame({"date": topix["date"]})
    for w in TOPIX_RETURN_WINDOWS:
        topix_returns[f"topix_return_{w}d"] = compute_return(topix["close"], w)
    return topix_returns


def run_phase13_conditional_analysis(
    config: AppConfig,
    tickers: list[str],
    phase6_5_report_path: Path,
    phase7_report_path: Path,
) -> Phase13Report:
    config_check = verify_config_hash(phase6_5_report_path, phase7_report_path)
    if not config_check.matches:
        raise ConfigMismatchError(f"CONFIG_MISMATCH: {config_check}")

    features_dir = Path(config.data.features_dir)
    scores_dir = Path(config.data.scores_dir)

    data = load_target_data(config, tickers)
    all_records = data.all_records
    target_records = data.target_records
    target_panel = data.target_panel
    population_forward_5d = data.population_forward_5d
    target_trades = data.target_trades
    topix = data.topix
    regime_df = data.regime_df
    tiers = data.tiers
    bootstrap_config = data.bootstrap_config
    permutation_config = data.permutation_config
    phase9_config = data.phase9_config

    overall_stats = forward_return_stats(target_records, target_panel)

    # --- axis 1: Market Regime ---------------------------------------------------
    logger.info("Phase 13: axis 1 - Market Regime")
    keys_with_regime = target_records.merge(regime_df, on="date", how="left")
    regime_buckets = [
        analyze_bucket(
            "regime", str(regime), group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for regime, group in keys_with_regime.groupby("regime", dropna=True)
    ]

    # --- axis 2: Market Drawdown / Momentum (TOPIX N-day return) -----------------
    logger.info("Phase 13: axis 2 - Market Drawdown")
    topix_returns = compute_topix_returns(topix)
    keys_with_topix = target_records.merge(topix_returns, on="date", how="left")
    keys_with_topix["bucket_20d"] = keys_with_topix["topix_return_20d"].apply(
        lambda v: bin_label(v, MARKET_DRAWDOWN_20D_EDGES)
    )
    market_drawdown_20d_buckets = [
        analyze_bucket(
            "market_drawdown_20d", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in keys_with_topix.groupby("bucket_20d", dropna=True)
    ]
    market_drawdown_secondary = []
    for w in (1, 3, 5, 10, 60):
        col = f"topix_return_{w}d"
        edges = MARKET_DRAWDOWN_20D_EDGES  # same fixed edge shape reused across windows
        keys_with_topix[f"bucket_{w}d"] = keys_with_topix[col].apply(lambda v: bin_label(v, edges))
        for label, group in keys_with_topix.groupby(f"bucket_{w}d", dropna=True):
            market_drawdown_secondary.append(
                descriptive_stats(
                    f"market_drawdown_{w}d", str(label), group[["ticker", "date"]], target_panel
                )
            )

    # --- axis 3: Individual Stock Drawdown (return_20d Feature) ------------------
    logger.info("Phase 13: axis 3 - Individual Stock Drawdown")
    stock_return_frames = []
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
        except FileNotFoundError:
            continue
        cols = ["date"] + [
            f"return_{w}d" for w in (3, 5, 10, 20, 60) if f"return_{w}d" in panel.columns
        ]
        frame = panel[cols].copy()
        frame.insert(0, "ticker", ticker)
        stock_return_frames.append(frame)
    stock_returns = (
        pd.concat(stock_return_frames, ignore_index=True) if stock_return_frames else pd.DataFrame()
    )
    keys_with_stock_return = target_records.merge(stock_returns, on=["ticker", "date"], how="left")
    keys_with_stock_return["bucket_20d"] = keys_with_stock_return.get(
        "return_20d", pd.Series(dtype=float)
    ).apply(lambda v: bin_label(v, STOCK_DRAWDOWN_20D_EDGES))
    stock_drawdown_20d_buckets = [
        analyze_bucket(
            "stock_drawdown_20d", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in keys_with_stock_return.groupby("bucket_20d", dropna=True)
    ]
    stock_drawdown_secondary = []
    for w in (5, 10, 60):
        col = f"return_{w}d"
        if col not in keys_with_stock_return.columns:
            continue
        keys_with_stock_return[f"bucket_{w}d"] = keys_with_stock_return[col].apply(
            lambda v: bin_label(v, STOCK_DRAWDOWN_20D_EDGES)
        )
        for label, group in keys_with_stock_return.groupby(f"bucket_{w}d", dropna=True):
            stock_drawdown_secondary.append(
                descriptive_stats(
                    f"stock_drawdown_{w}d", str(label), group[["ticker", "date"]], target_panel
                )
            )

    # --- axis 4: MA deviation ------------------------------------------------------
    logger.info("Phase 13: axis 4 - MA deviation")
    ma_frames = []
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
        except FileNotFoundError:
            continue
        ma_candidates = ("close_to_sma_5", "close_to_sma_20", "close_to_sma_50")
        ma_cols = [c for c in ma_candidates if c in panel.columns]
        frame = panel[["date"] + ma_cols].copy()
        frame.insert(0, "ticker", ticker)
        ma_frames.append(frame)
    ma_data = pd.concat(ma_frames, ignore_index=True) if ma_frames else pd.DataFrame()
    keys_with_ma = target_records.merge(ma_data, on=["ticker", "date"], how="left")
    keys_with_ma["bucket_ma20"] = keys_with_ma.get("close_to_sma_20", pd.Series(dtype=float)).apply(
        lambda v: bin_label(v, MA20_DEVIATION_EDGES)
    )
    ma20_deviation_buckets = [
        analyze_bucket(
            "ma20_deviation", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in keys_with_ma.groupby("bucket_ma20", dropna=True)
    ]
    ma_deviation_secondary = []
    for col, tag in (("close_to_sma_5", "ma5"), ("close_to_sma_50", "ma50_proxy_for_ma60")):
        if col not in keys_with_ma.columns:
            continue
        keys_with_ma[f"bucket_{tag}"] = keys_with_ma[col].apply(
            lambda v: bin_label(v, MA20_DEVIATION_EDGES)
        )
        for label, group in keys_with_ma.groupby(f"bucket_{tag}", dropna=True):
            ma_deviation_secondary.append(
                descriptive_stats(tag, str(label), group[["ticker", "date"]], target_panel)
            )

    # --- axis 5: Volume --------------------------------------------------------------
    logger.info("Phase 13: axis 5 - Volume")
    volume_frames = []
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
        except FileNotFoundError:
            continue
        frame = panel[["date", "volume_ratio_20d"]].copy()
        frame.insert(0, "ticker", ticker)
        volume_frames.append(frame)
    volume_data = pd.concat(volume_frames, ignore_index=True) if volume_frames else pd.DataFrame()
    keys_with_volume = target_records.merge(volume_data, on=["ticker", "date"], how="left")
    keys_with_volume["bucket"] = keys_with_volume["volume_ratio_20d"].apply(
        lambda v: bin_label(v, VOLUME_RATIO_20D_EDGES)
    )
    volume_buckets = [
        analyze_bucket(
            "volume", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in keys_with_volume.groupby("bucket", dropna=True)
    ]

    # --- axis 6: Volatility (tercile quantile of the signal population) ----------
    logger.info("Phase 13: axis 6 - Volatility")
    vol_frames = []
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
        except FileNotFoundError:
            continue
        frame = panel[["date", "volatility_20d"]].copy()
        frame.insert(0, "ticker", ticker)
        vol_frames.append(frame)
    vol_data = pd.concat(vol_frames, ignore_index=True) if vol_frames else pd.DataFrame()
    keys_with_vol = target_records.merge(vol_data, on=["ticker", "date"], how="left").dropna(
        subset=["volatility_20d"]
    )
    keys_with_vol = keys_with_vol.assign(
        bucket=assign_quantile_buckets(keys_with_vol["volatility_20d"], n_buckets=3)
    )
    volatility_buckets = [
        analyze_bucket(
            "volatility", str(label), group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in keys_with_vol.groupby("bucket", dropna=True, observed=True)
    ]

    # --- axis 7: Score ---------------------------------------------------------------
    logger.info("Phase 13: axis 7 - Score")
    score_data = load_score_data(tickers, scores_dir)
    keys_with_score = target_records.merge(score_data, on=["ticker", "date"], how="left").dropna(
        subset=["total_score"]
    )
    if not keys_with_score.empty:
        keys_with_score = keys_with_score.assign(
            score_bucket=assign_quantile_buckets(keys_with_score["total_score"])
        )
    score_analysis = compute_score_analysis(target_records, target_panel, score_data)

    # --- axis 8: Regime x Score ------------------------------------------------------
    logger.info("Phase 13: axis 8 - Regime x Score")
    regime_x_score = []
    if not keys_with_score.empty:
        keys_rs = keys_with_score.merge(regime_df, on="date", how="left")
        for (regime, score_bucket), group in keys_rs.groupby(
            ["regime", "score_bucket"], dropna=True, observed=True
        ):
            if len(group) < config.validation.min_sample.min_oos_trades:
                continue
            regime_x_score.append(
                analyze_bucket(
                    "regime_x_score", f"{regime}:{score_bucket}", group[["ticker", "date"]],
                    target_panel, target_trades, population_forward_5d, tiers,
                    bootstrap_config, phase9_config, permutation_config,
                )
            )

    # --- axis 9: Market Drawdown x Score ---------------------------------------------
    logger.info("Phase 13: axis 9 - Market Drawdown x Score")
    drawdown_x_score = []
    if not keys_with_score.empty:
        keys_ds = keys_with_score.merge(
            keys_with_topix[["ticker", "date", "bucket_20d"]], on=["ticker", "date"], how="left"
        )
        for (dd_bucket, score_bucket), group in keys_ds.groupby(
            ["bucket_20d", "score_bucket"], dropna=True, observed=True
        ):
            if len(group) < config.validation.min_sample.min_oos_trades:
                continue
            drawdown_x_score.append(
                analyze_bucket(
                    "drawdown_x_score", f"{dd_bucket}:{score_bucket}", group[["ticker", "date"]],
                    target_panel, target_trades, population_forward_5d, tiers,
                    bootstrap_config, phase9_config, permutation_config,
                )
            )

    # --- axis 10: Signal Count (reuses Phase 12's Ensemble aggregation) ----------------
    logger.info("Phase 13: axis 10 - Signal Count")
    signal_counts_df = aggregate_signal_counts(all_records)
    target_with_counts = target_records.merge(
        signal_counts_df[["ticker", "date", "long_count", "short_count"]],
        on=["ticker", "date"], how="left",
    )
    target_with_counts["other_long_count"] = (target_with_counts["long_count"] - 1).clip(lower=0)
    target_with_counts["count_bucket"] = target_with_counts["other_long_count"].apply(
        _signal_count_bucket
    )
    signal_count_buckets = [
        analyze_bucket(
            "signal_count", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in target_with_counts.groupby("count_bucket", dropna=True)
    ]

    # --- axis 11: LONG/SHORT consensus ------------------------------------------------
    logger.info("Phase 13: axis 11 - LONG/SHORT consensus")

    def _consensus_label(row: pd.Series) -> str:
        long_n, short_n = row["long_count"], row["short_count"]
        if short_n == 0:
            return "LONG_ONLY"
        if long_n > short_n:
            return "LONG_MAJORITY"
        if long_n == short_n:
            return "TIE"
        return "SHORT_MAJORITY"

    target_with_counts["consensus"] = target_with_counts.apply(_consensus_label, axis=1)
    long_short_consensus_buckets = [
        analyze_bucket(
            "long_short_consensus", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in target_with_counts.groupby("consensus", dropna=True)
    ]

    # --- axis: Event Exclusion + Episode Analysis -------------------------------------
    logger.info("Phase 13: Event Exclusion + Episode Analysis")
    base_tier = next(t for t in tiers if t.name == "base")
    case_a = target_trades
    aug_mask = (target_trades["signal_date"] >= AUG_2024_EVENT_START) & (
        target_trades["signal_date"] <= AUG_2024_EVENT_END
    )
    case_b = target_trades[~aug_mask]
    case_c = target_trades[target_trades["signal_date"].apply(lambda d: d.year) != 2024]

    merged_regime = target_trades.merge(
        regime_df, left_on="signal_date", right_on="date", how="left"
    )
    bear_trades = merged_regime[merged_regime["regime"] == "BEAR"]
    bear_aug_mask = (bear_trades["signal_date"] >= AUG_2024_EVENT_START) & (
        bear_trades["signal_date"] <= AUG_2024_EVENT_END
    )
    case_d = bear_trades[~bear_aug_mask]

    def _cost(df: pd.DataFrame) -> BacktestMetrics:
        return compute_metrics(df.assign(**{"return": apply_cost(df["return"], base_tier)}))

    event_case_a, event_case_b, event_case_c, event_case_d = (
        _cost(case_a), _cost(case_b), _cost(case_c), _cost(case_d)
    )

    bear_episodes_all = identify_regime_episodes(regime_df, "BEAR")
    bear_episode_metrics = compute_rich_episode_metrics(target_trades, bear_episodes_all)

    # --- Multiple Testing bookkeeping --------------------------------------------------
    fdr_p_values: dict[str, float] = {}
    bucket_groups: list[tuple[str, list[BucketResult]]] = [
        ("regime", regime_buckets),
        ("market_drawdown_20d", market_drawdown_20d_buckets),
        ("stock_drawdown_20d", stock_drawdown_20d_buckets),
        ("ma20_deviation", ma20_deviation_buckets),
        ("volume", volume_buckets),
        ("volatility", volatility_buckets),
        ("signal_count", signal_count_buckets),
        ("long_short_consensus", long_short_consensus_buckets),
        ("regime_x_score", regime_x_score),
        ("drawdown_x_score", drawdown_x_score),
    ]
    for group_name, results in bucket_groups:
        for r in results:
            if r.permutation is not None:
                fdr_p_values[f"{group_name}:{r.label}"] = r.permutation.p_value

    return Phase13Report(
        config_check=config_check,
        tickers=tickers,
        n_signal_total=len(target_records),
        unique_tickers=target_records["ticker"].nunique(),
        unique_signal_dates=target_records["date"].nunique(),
        overall_forward_return_stats=overall_stats,
        regime_buckets=regime_buckets,
        market_drawdown_20d_buckets=market_drawdown_20d_buckets,
        market_drawdown_secondary=market_drawdown_secondary,
        stock_drawdown_20d_buckets=stock_drawdown_20d_buckets,
        stock_drawdown_secondary=stock_drawdown_secondary,
        ma20_deviation_buckets=ma20_deviation_buckets,
        ma_deviation_secondary=ma_deviation_secondary,
        volume_buckets=volume_buckets,
        volatility_buckets=volatility_buckets,
        score_analysis=score_analysis,
        regime_x_score=regime_x_score,
        drawdown_x_score=drawdown_x_score,
        signal_count_buckets=signal_count_buckets,
        long_short_consensus_buckets=long_short_consensus_buckets,
        event_case_a=event_case_a,
        event_case_b_excl_aug2024=event_case_b,
        event_case_c_excl_2024=event_case_c,
        event_case_d_bear_excl_aug2024=event_case_d,
        bear_episodes=bear_episode_metrics,
        fdr_tested_p_values=fdr_p_values,
    )
