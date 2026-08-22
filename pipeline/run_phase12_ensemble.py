"""Phase 12: Signal Ensemble Validation.

Analyzes co-occurrence of the 12 EXISTING frozen Signals - never a new
Signal, never a change to any existing Signal/Feature/Score/Backtest
code (spec section 1). This module only reads already-computed Signal
Records / Score Records / Backtest trades and Forward Targets
(targets/forward_returns.py, itself extended with 15d/20d in this same
Phase - see that module) and aggregates them.

Reuse (spec section 1's "既存とみなせる分析コードの追加は許可する", and
section 36's "同じデータに対して計算を繰り返さない"):
- run_backtest() is called EXACTLY ONCE for the full trade set (all 12
  Signals, both directions) - every downstream cost/PF/expectancy/
  bootstrap/permutation/regime/event slice re-filters this ONE
  DataFrame, never recomputes trades.
- Forward Returns (all 7 windows) and MFE/MAE are computed ONCE per
  ticker and reused across every Signal Count / NET / combination
  bucket.
- Bootstrap/Day-Cluster-Bootstrap/Block-Bootstrap/Permutation/FDR are
  the SAME primitives Phase 6-9 already use, completely unmodified.
- Phase 9's day_cluster_bootstrap/block_bootstrap configs (10,000
  resamples) are reused via config.loader.load_phase9_config() rather
  than inventing new Phase-12-specific seeds/resample counts.

Analysis depth is intentionally TIERED to keep Full-Universe runtime
bounded (spec section 36 forbids unbounded combinatorial exploration
anyway - section 12): the 17 primary Signal Count buckets (LONG 1-4+,
SHORT 1-4+, NET across 9 buckets) get the FULL battery (frequency,
Forward Return, MFE/MAE, cost-tier PF/expectancy, trade-level Bootstrap,
Day Cluster Bootstrap, Block Bootstrap, Permutation Test, Regime cross,
2024-08 event exclusion, Decision). Naturally-occurring Signal
combinations (spec section 12) get a lighter but still statistically
valid battery (frequency, Forward Return, cost-tier PF/expectancy,
trade-level Bootstrap, Permutation Test, Decision) - Day Cluster/Block
Bootstrap and the Event/Regime sub-splits are skipped per-combination
specifically because there can be dozens of qualifying combinations,
and those two checks are the most expensive; this tiering is fixed here
in this module's design, before any real run, not chosen after seeing
which combinations looked promising.

Dependency direction (mirrors Phase 9/11): this module imports FROM
signals/backtest/targets, and must never be imported BY them - see
tests/test_ensemble_no_lookahead.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.block_bootstrap import BlockBootstrapResult, block_bootstrap
from backtest.bootstrap import BootstrapResult, bootstrap_ci
from backtest.costs import apply_cost
from backtest.day_cluster_bootstrap import DayClusterBootstrapResult, day_cluster_bootstrap
from backtest.market_regime import compute_market_regime
from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import PermutationResult, permutation_test
from config.loader import AppConfig, load_phase9_config
from ensemble.combinations import (
    CombinationCount,
    PairwiseCooccurrence,
    aggregate_combinations,
    compute_pairwise_cooccurrence,
)
from ensemble.decision import EnsembleDecision, EnsembleDecisionInputs, classify_ensemble
from ensemble.frequency import FrequencyMetrics, compute_frequency_metrics
from ensemble.portfolio_sim import (
    TOP_N,
    EquityCurveMetrics,
    compute_equity_curve_metrics,
    dedupe_trades_by_ticker_date_direction,
    select_top_n_candidates,
)
from ensemble.signal_count import (
    LONG_COUNT_BUCKET_ORDER,
    NET_SIGNAL_COUNT_BUCKET_ORDER,
    SHORT_COUNT_BUCKET_ORDER,
    aggregate_signal_counts,
    net_signal_count_bucket,
    signal_count_bucket,
)
from forward_test.manifest import verify_strategy_hashes_unchanged
from pipeline.run_backtest import run_backtest
from pipeline.run_phase8_analysis import ConfigCheckResult, ConfigMismatchError, verify_config_hash
from pipeline.run_phase9_analysis import AUG_2024_EVENT_END, AUG_2024_EVENT_START
from signals.registry import all_signal_meta
from storage.parquet_store import (
    load_feature_panel,
    load_ohlcv,
    load_score_records,
    load_signal_records,
)
from targets.forward_returns import FORWARD_WINDOWS, compute_forward_returns

logger = logging.getLogger(__name__)

# Fixed before running any Phase 12 analysis (never tuned to results).
MIN_REGIME_SAMPLE = 10
MIN_EVENT_EXCLUSION_SAMPLE = 10
PERMUTATION_FORWARD_WINDOW = 5  # matches config.validation.permutation.forward_window default
_PERMUTATION_COL = f"forward_return_{PERMUTATION_FORWARD_WINDOW}d"


@dataclass(frozen=True)
class ForwardReturnStats:
    window_days: int
    n: int
    mean_return: float | None
    median_return: float | None
    win_rate: float | None


@dataclass(frozen=True)
class BucketAnalysis:
    label: str
    direction: str | None
    n_sample: int
    frequency: FrequencyMetrics
    forward_return_stats: list[ForwardReturnStats]
    cost_metrics: dict[str, BacktestMetrics] | None
    bootstrap_expectancy: BootstrapResult | None
    day_cluster_bootstrap: dict[str, DayClusterBootstrapResult] | None
    block_bootstrap: dict[str, BlockBootstrapResult] | None
    permutation: PermutationResult | None
    regime_metrics: dict[str, BacktestMetrics] | None
    case_a_metrics: BacktestMetrics | None
    case_b_metrics: BacktestMetrics | None
    decision: EnsembleDecision


@dataclass(frozen=True)
class CombinationAnalysis:
    combo: CombinationCount
    n_sample: int
    forward_return_stats: list[ForwardReturnStats]
    cost_metrics: dict[str, BacktestMetrics] | None
    bootstrap_expectancy: BootstrapResult | None
    permutation: PermutationResult | None
    decision: EnsembleDecision


@dataclass(frozen=True)
class ScoreCrossCell:
    signal_count_bucket: str
    score_bucket: str
    n: int
    metrics: BacktestMetrics


@dataclass(frozen=True)
class Phase12Report:
    config_check: ConfigCheckResult
    tickers: list[str]
    integrity_hash_matches_strategy_v1: bool
    integrity_hash_mismatches: list[str]
    total_trading_days: int
    long_count_buckets: list[BucketAnalysis]
    short_count_buckets: list[BucketAnalysis]
    net_count_buckets: list[BucketAnalysis]
    long_combinations: list[CombinationAnalysis]
    short_combinations: list[CombinationAnalysis]
    pairwise_cooccurrence_long: list[PairwiseCooccurrence]
    pairwise_cooccurrence_short: list[PairwiseCooccurrence]
    score_cross: list[ScoreCrossCell]
    fdr_results: dict[str, FDRResult]
    top5_simulation: EquityCurveMetrics | None


# --- data assembly (each computed ONCE, reused everywhere below) ---------------


def load_all_signal_records(tickers: list[str], signals_dir: Path) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            df = load_signal_records(ticker, signals_dir)
        except FileNotFoundError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "signal_name", "direction"])
    return pd.concat(frames, ignore_index=True)


def load_forward_return_panel(tickers: list[str], features_dir: Path) -> pd.DataFrame:
    """ticker/date/forward_return_{n}d for EVERY row of every ticker's
    Feature panel (not just triggered rows) - the permutation test's
    population AND the source Forward Return joined onto Signal Count
    bucket rows below.
    """
    frames = []
    for ticker in tickers:
        try:
            panel = load_feature_panel(ticker, features_dir)
        except FileNotFoundError:
            continue
        forward = compute_forward_returns(panel)
        frame = panel[["date"]].join(forward)
        frame.insert(0, "ticker", ticker)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_score_by_key(tickers: list[str], scores_dir: Path) -> dict[tuple[str, date_type], float]:
    result: dict[tuple[str, date_type], float] = {}
    for ticker in tickers:
        try:
            df = load_score_records(ticker, scores_dir)
        except FileNotFoundError:
            continue
        for row in df.itertuples(index=False):
            key = (row.ticker, row.date)
            result[key] = max(result.get(key, -np.inf), row.total_score)
    return result


# --- descriptive stats -----------------------------------------------------------


def _forward_return_stats_for_bucket(
    bucket_keys: pd.DataFrame, forward_panel: pd.DataFrame
) -> list[ForwardReturnStats]:
    merged = bucket_keys[["ticker", "date"]].merge(forward_panel, on=["ticker", "date"], how="left")
    stats = []
    for w in FORWARD_WINDOWS:
        col = f"forward_return_{w}d"
        values = merged[col].dropna()
        stats.append(
            ForwardReturnStats(
                window_days=w,
                n=len(values),
                mean_return=float(values.mean()) if len(values) else None,
                median_return=float(values.median()) if len(values) else None,
                win_rate=float((values > 0).mean()) if len(values) else None,
            )
        )
    return stats


# --- trade-based stats -------------------------------------------------------------


def _cost_metrics(trades: pd.DataFrame, tiers) -> dict[str, BacktestMetrics]:
    return {
        tier.name: compute_metrics(trades.assign(**{"return": apply_cost(trades["return"], tier)}))
        for tier in tiers
    }


def _regime_metrics(trades: pd.DataFrame, regime_df: pd.DataFrame) -> dict[str, BacktestMetrics]:
    if trades.empty:
        return {}
    merged = trades.merge(regime_df, left_on="signal_date", right_on="date", how="left")
    return {
        str(name): compute_metrics(group)
        for name, group in merged.groupby("regime", dropna=True)
    }


def _event_split(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (trades["signal_date"] >= AUG_2024_EVENT_START) & (
        trades["signal_date"] <= AUG_2024_EVENT_END
    )
    case_a = trades
    case_b = trades[~mask]
    return case_a, case_b


def _positive_flag(metrics: BacktestMetrics, min_sample: int) -> bool | None:
    if metrics.n_trades < min_sample:
        return None
    return (
        metrics.expectancy is not None and metrics.expectancy > 0
        and metrics.profit_factor is not None and metrics.profit_factor > 1.0
    )


# --- core per-bucket analysis --------------------------------------------------


def _analyze_bucket(
    label: str,
    direction: str | None,
    bucket_keys: pd.DataFrame,
    forward_panel: pd.DataFrame,
    all_trading_dates: pd.Series,
    deduped_trades: pd.DataFrame,
    population_forward_5d: np.ndarray,
    regime_df: pd.DataFrame,
    tiers,
    phase9_config,
    permutation_config,
    bootstrap_config,
) -> BucketAnalysis:
    n_sample = len(bucket_keys)
    frequency = compute_frequency_metrics(
        label, bucket_keys["date"], bucket_keys["ticker"], all_trading_dates
    )
    fwd_stats = _forward_return_stats_for_bucket(bucket_keys, forward_panel)

    cost_metrics = None
    bootstrap_expectancy = None
    day_cluster = None
    block = None
    permutation = None
    regime_metrics = None
    case_a_metrics = None
    case_b_metrics = None
    positive_excl_event: bool | None = None
    positive_bull: bool | None = None
    positive_neutral: bool | None = None
    positive_bear: bool | None = None

    if direction is not None:
        keys = set(zip(bucket_keys["ticker"], bucket_keys["date"]))
        trades = deduped_trades[
            (deduped_trades["direction"] == direction) & deduped_trades["_key"].isin(keys)
        ]
        cost_metrics = _cost_metrics(trades, tiers)
        base_tier = next(t for t in tiers if t.name == "base")
        base_returns = apply_cost(trades["return"], base_tier)
        bootstrap_expectancy = bootstrap_ci(
            base_returns.to_numpy(), "expectancy", bootstrap_config
        )
        day_cluster = {
            m: day_cluster_bootstrap(trades, m, phase9_config.day_cluster_bootstrap)
            for m in ("mean_return", "expectancy", "profit_factor")
        }
        block = {
            m: block_bootstrap(trades, m, phase9_config.block_bootstrap)
            for m in ("mean_return", "expectancy", "profit_factor")
        }

        merged = bucket_keys[["ticker", "date"]].merge(
            forward_panel, on=["ticker", "date"], how="left"
        )
        signal_returns = merged[_PERMUTATION_COL].dropna().to_numpy()
        if len(signal_returns) > 0:
            permutation = permutation_test(
                signal_returns, population_forward_5d, permutation_config
            )

        regime_metrics = _regime_metrics(trades, regime_df)
        case_a, case_b = _event_split(trades)
        case_a_metrics = compute_metrics(case_a)
        case_b_metrics = compute_metrics(case_b)

        positive_excl_event = _positive_flag(case_b_metrics, MIN_EVENT_EXCLUSION_SAMPLE)
        positive_bull = _positive_flag(
            regime_metrics.get("BULL", compute_metrics(trades.iloc[0:0])), MIN_REGIME_SAMPLE
        )
        positive_neutral = _positive_flag(
            regime_metrics.get("NEUTRAL", compute_metrics(trades.iloc[0:0])), MIN_REGIME_SAMPLE
        )
        positive_bear = _positive_flag(
            regime_metrics.get("BEAR", compute_metrics(trades.iloc[0:0])), MIN_REGIME_SAMPLE
        )

    decision_inputs = EnsembleDecisionInputs(
        n_sample=n_sample,
        pct_trading_days_with_occurrence=frequency.pct_trading_days_with_occurrence,
        expectancy_ci_low=bootstrap_expectancy.ci_low if bootstrap_expectancy else None,
        profit_factor_high_cost=(
            cost_metrics["high"].profit_factor if cost_metrics and "high" in cost_metrics else None
        ),
        permutation_p_value=permutation.p_value if permutation else None,
        positive_excluding_aug2024=positive_excl_event,
        positive_in_bull=positive_bull,
        positive_in_neutral=positive_neutral,
        positive_in_bear=positive_bear,
    )
    decision = classify_ensemble(decision_inputs)

    return BucketAnalysis(
        label=label, direction=direction, n_sample=n_sample, frequency=frequency,
        forward_return_stats=fwd_stats, cost_metrics=cost_metrics,
        bootstrap_expectancy=bootstrap_expectancy, day_cluster_bootstrap=day_cluster,
        block_bootstrap=block, permutation=permutation, regime_metrics=regime_metrics,
        case_a_metrics=case_a_metrics, case_b_metrics=case_b_metrics, decision=decision,
    )


def _analyze_combination(
    combo: CombinationCount,
    signal_counts_df: pd.DataFrame,
    forward_panel: pd.DataFrame,
    all_trading_dates: pd.Series,
    deduped_trades: pd.DataFrame,
    population_forward_5d: np.ndarray,
    tiers,
    permutation_config,
    bootstrap_config,
) -> CombinationAnalysis:
    col = "long_signals" if combo.direction == "LONG" else "short_signals"
    bucket_keys = signal_counts_df[signal_counts_df[col] == combo.signals][["ticker", "date"]]
    n_sample = len(bucket_keys)
    fwd_stats = _forward_return_stats_for_bucket(bucket_keys, forward_panel)

    cost_metrics = None
    bootstrap_expectancy = None
    permutation = None

    if combo.sufficient_sample:
        keys = set(zip(bucket_keys["ticker"], bucket_keys["date"]))
        trades = deduped_trades[
            (deduped_trades["direction"] == combo.direction) & deduped_trades["_key"].isin(keys)
        ]
        cost_metrics = _cost_metrics(trades, tiers)
        base_tier = next(t for t in tiers if t.name == "base")
        base_returns = apply_cost(trades["return"], base_tier)
        bootstrap_expectancy = bootstrap_ci(
            base_returns.to_numpy(), "expectancy", bootstrap_config
        )
        merged = bucket_keys.merge(forward_panel, on=["ticker", "date"], how="left")
        signal_returns = merged[_PERMUTATION_COL].dropna().to_numpy()
        if len(signal_returns) > 0:
            permutation = permutation_test(
                signal_returns, population_forward_5d, permutation_config
            )

    decision_inputs = EnsembleDecisionInputs(
        n_sample=n_sample,
        pct_trading_days_with_occurrence=n_sample / len(set(all_trading_dates))
        if len(all_trading_dates) else 0.0,
        expectancy_ci_low=bootstrap_expectancy.ci_low if bootstrap_expectancy else None,
        profit_factor_high_cost=(
            cost_metrics["high"].profit_factor if cost_metrics and "high" in cost_metrics else None
        ),
        permutation_p_value=permutation.p_value if permutation else None,
        positive_excluding_aug2024=None,
        positive_in_bull=None,
        positive_in_neutral=None,
        positive_in_bear=None,
    )
    decision = classify_ensemble(decision_inputs)

    return CombinationAnalysis(
        combo=combo, n_sample=n_sample, forward_return_stats=fwd_stats,
        cost_metrics=cost_metrics, bootstrap_expectancy=bootstrap_expectancy,
        permutation=permutation, decision=decision,
    )


def run_phase12_ensemble(
    config: AppConfig,
    tickers: list[str],
    phase6_5_report_path: Path,
    phase7_report_path: Path,
    strategy_manifest: dict,
) -> Phase12Report:
    config_check = verify_config_hash(phase6_5_report_path, phase7_report_path)
    if not config_check.matches:
        raise ConfigMismatchError(f"CONFIG_MISMATCH: {config_check}")

    hash_unchanged, hash_mismatches = verify_strategy_hashes_unchanged(strategy_manifest)

    # Sanity: exactly the 12 registered Signals are ever analyzed (spec
    # section 2 - no new Signal, none skipped).
    meta = all_signal_meta(config.signals)
    assert len(meta) == 12, f"expected 12 registered Signals, found {len(meta)}"

    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)
    scores_dir = Path(config.data.scores_dir)

    logger.info("Phase 12: loading Signal Records for %d tickers", len(tickers))
    all_records = load_all_signal_records(tickers, signals_dir)
    signal_counts_df = aggregate_signal_counts(all_records)

    logger.info("Phase 12: building Forward Return population for %d tickers", len(tickers))
    forward_panel = load_forward_return_panel(tickers, features_dir)
    population_forward_5d = forward_panel[_PERMUTATION_COL].dropna().to_numpy()
    all_trading_dates = (
        forward_panel["date"] if not forward_panel.empty else pd.Series([], dtype=object)
    )

    logger.info("Phase 12: running Combined backtest over %d tickers", len(tickers))
    backtest_summary = run_backtest(config, tickers=tickers)
    deduped_trades = dedupe_trades_by_ticker_date_direction(backtest_summary.trades)
    # Precomputed ONCE and reused by every bucket/combination filter below
    # (vectorized .isin() against a set, not a per-row Python apply() -
    # this DataFrame can be ~1M rows at Full Universe scale).
    deduped_trades = deduped_trades.assign(
        _key=list(zip(deduped_trades["ticker"], deduped_trades["signal_date"])),
        _key3=list(
            zip(
                deduped_trades["ticker"], deduped_trades["signal_date"], deduped_trades["direction"]
            )
        ),
    )

    topix = load_ohlcv("TOPIX", config.data.processed_dir)
    regime_df = compute_market_regime(topix, config.validation.market_regime)

    tiers = config.validation.transaction_cost.tiers
    permutation_config = config.validation.permutation
    bootstrap_config = config.validation.bootstrap
    phase9_config = load_phase9_config()

    # --- Signal Count buckets (LONG / SHORT / NET) - full battery -----------

    long_buckets = []
    for label in LONG_COUNT_BUCKET_ORDER:
        keys = signal_counts_df[
            signal_counts_df["long_count"].apply(signal_count_bucket) == label
        ]
        logger.info("Phase 12: analyzing LONG_COUNT=%s (n=%d)", label, len(keys))
        long_buckets.append(
            _analyze_bucket(
                f"LONG_COUNT={label}", "LONG", keys, forward_panel, all_trading_dates,
                deduped_trades, population_forward_5d, regime_df, tiers,
                phase9_config, permutation_config, bootstrap_config,
            )
        )

    short_buckets = []
    for label in SHORT_COUNT_BUCKET_ORDER:
        keys = signal_counts_df[
            signal_counts_df["short_count"].apply(signal_count_bucket) == label
        ]
        logger.info("Phase 12: analyzing SHORT_COUNT=%s (n=%d)", label, len(keys))
        short_buckets.append(
            _analyze_bucket(
                f"SHORT_COUNT={label}", "SHORT", keys, forward_panel, all_trading_dates,
                deduped_trades, population_forward_5d, regime_df, tiers,
                phase9_config, permutation_config, bootstrap_config,
            )
        )

    negative_net_labels = {"<=-4", "-3", "-2", "-1"}
    positive_net_labels = {"+1", "+2", "+3", ">=+4"}
    net_buckets = []
    for label in NET_SIGNAL_COUNT_BUCKET_ORDER:
        keys = signal_counts_df[
            signal_counts_df["net_signal_count"].apply(net_signal_count_bucket) == label
        ]
        if label in negative_net_labels:
            net_direction = "SHORT"
        elif label in positive_net_labels:
            net_direction = "LONG"
        else:
            net_direction = None  # "0": perfectly balanced, no clear position to take
        logger.info("Phase 12: analyzing NET=%s (n=%d)", label, len(keys))
        net_buckets.append(
            _analyze_bucket(
                f"NET={label}", net_direction, keys, forward_panel, all_trading_dates,
                deduped_trades, population_forward_5d, regime_df, tiers,
                phase9_config, permutation_config, bootstrap_config,
            )
        )

    # --- Combinations (lighter battery) ----------------------------------------

    long_combos_all = aggregate_combinations(signal_counts_df, "LONG")
    short_combos_all = aggregate_combinations(signal_counts_df, "SHORT")

    logger.info(
        "Phase 12: analyzing %d LONG combinations, %d SHORT combinations",
        len(long_combos_all), len(short_combos_all),
    )
    long_combinations = [
        _analyze_combination(
            c, signal_counts_df, forward_panel, all_trading_dates, deduped_trades,
            population_forward_5d, tiers, permutation_config, bootstrap_config,
        )
        for c in long_combos_all
    ]
    short_combinations = [
        _analyze_combination(
            c, signal_counts_df, forward_panel, all_trading_dates, deduped_trades,
            population_forward_5d, tiers, permutation_config, bootstrap_config,
        )
        for c in short_combos_all
    ]

    pairwise_long = compute_pairwise_cooccurrence(signal_counts_df, "LONG")
    pairwise_short = compute_pairwise_cooccurrence(signal_counts_df, "SHORT")

    # --- Signal Count x Score cross table (descriptive only) -------------------

    logger.info("Phase 12: building Signal Count x Score cross table")
    score_by_key = _load_score_by_key(tickers, scores_dir)
    signal_counts_df = signal_counts_df.assign(
        score=[
            score_by_key.get((t, d))
            for t, d in zip(signal_counts_df["ticker"], signal_counts_df["date"])
        ]
    )
    from scoring.validation import assign_quantile_buckets

    score_cross: list[ScoreCrossCell] = []
    with_score = signal_counts_df.dropna(subset=["score"])
    if not with_score.empty:
        with_score = with_score.assign(
            score_bucket=assign_quantile_buckets(with_score["score"]),
            long_bucket=with_score["long_count"].apply(
                lambda n: signal_count_bucket(n) if n >= 1 else None
            ),
        )
        merged_fwd = with_score.merge(forward_panel, on=["ticker", "date"], how="left")
        for (sc_bucket, score_bucket), group in merged_fwd.groupby(
            ["long_bucket", "score_bucket"], dropna=True, observed=True
        ):
            renamed = group.rename(columns={"forward_return_5d": "return"})
            score_cross.append(
                ScoreCrossCell(
                    signal_count_bucket=str(sc_bucket), score_bucket=str(score_bucket),
                    n=len(group), metrics=compute_metrics(renamed),
                )
            )

    # --- Multiple Testing (FDR) across every tested unit's permutation p ------

    p_values: dict[str, float] = {}
    for b in long_buckets + short_buckets + net_buckets:
        if b.permutation is not None:
            p_values[b.label] = b.permutation.p_value
    for c in long_combinations + short_combinations:
        if c.permutation is not None:
            p_values[f"{c.combo.direction}:{'+'.join(c.combo.signals)}"] = c.permutation.p_value
    fdr_results = benjamini_hochberg_correction(p_values)

    # --- Optional Top-5 simulation (spec section 25) ----------------------------

    top5_sim: EquityCurveMetrics | None = None
    if not signal_counts_df.empty:
        selected = select_top_n_candidates(signal_counts_df, score_by_key, top_n=TOP_N)
        if not selected.empty:
            selected_keys = set(
                zip(selected["ticker"], selected["date"], selected["dominant_direction"])
            )
            selected_trades = deduped_trades[deduped_trades["_key3"].isin(selected_keys)]
            top5_sim = compute_equity_curve_metrics(selected_trades)

    return Phase12Report(
        config_check=config_check,
        tickers=tickers,
        integrity_hash_matches_strategy_v1=hash_unchanged,
        integrity_hash_mismatches=hash_mismatches,
        total_trading_days=len(set(all_trading_dates)),
        long_count_buckets=long_buckets,
        short_count_buckets=short_buckets,
        net_count_buckets=net_buckets,
        long_combinations=long_combinations,
        short_combinations=short_combinations,
        pairwise_cooccurrence_long=pairwise_long,
        pairwise_cooccurrence_short=pairwise_short,
        score_cross=score_cross,
        fdr_results=fdr_results,
        top5_simulation=top5_sim,
    )
