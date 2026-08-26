"""Phase V2-2 orchestrator (spec section 33 STEP 6-19): builds the V2-1
ranked panel ONCE (spec section 28's efficiency requirement - never
recomputed per Holding Period/analysis), then runs the full validation
battery.

Depth is TIERED across the 7 Forward Windows (targets.forward_returns.
FORWARD_WINDOWS, reused unmodified) - mirroring Phase 12/13's own
established "tiered analysis depth" precedent (documented there as a
deliberate, pre-run design decision, not a runtime shortcut): the
PRIMARY window (5 trading days - matching every other Phase's own
"primary" window convention, e.g. Phase 6+'s PermutationConfig.
forward_window default and Phase 13/14's PERMUTATION_FORWARD_WINDOW)
gets the full battery (Bootstrap x3 methods, Permutation, Event
Exclusion, Year-by-Year, Concentration, Regime, Cost, Top-N). Every
OTHER window gets the lighter battery (Q1-Q5 bucket stats,
Monotonicity, daily IC, Top-N) needed for the Holding Period comparison
(spec section 11) without repeating the expensive bootstrap/permutation
work 7 times over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtest.bootstrap import BootstrapDiffResult, bootstrap_diff_ci
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import permutation_test
from config.loader import (
    BlockBootstrapConfig,
    BootstrapConfig,
    DayClusterBootstrapConfig,
    MarketRegimeConfig,
    PermutationConfig,
)
from scoring.validation import assign_quantile_buckets
from storage.parquet_store import load_ohlcv
from targets.forward_returns import FORWARD_WINDOWS
from v2.config.loader import V2Config
from v2.pipeline import run_v2_ranking
from v2.stats import (
    QuantileBucketStats,
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    exclude_implausible_returns,
)
from v2.validation.concentration import ConcentrationResult, compute_concentration
from v2.validation.cost import CostSensitivityResult, compute_cost_sensitivity
from v2.validation.event_year_analysis import (
    SliceResult,
    analyze_event_exclusion,
    analyze_year_by_year,
)
from v2.validation.ic import ICSummary, compute_daily_spearman_ic, summarize_ic
from v2.validation.monotonicity import MonotonicityResult, compute_monotonicity
from v2.validation.permutation_analysis import (
    BucketPermutationResult,
    run_bucket_permutation_tests,
)
from v2.validation.regime import RegimeSliceResult, analyze_by_regime, build_regime_series
from v2.validation.spread_bootstrap import (
    SpreadBootstrapResult,
    block_spread_bootstrap,
    day_cluster_spread_bootstrap,
)
from v2.validation.topn import TOP_N_VALUES, TopNResult, compute_topn_daily_returns, summarize_topn

logger = logging.getLogger(__name__)

PRIMARY_WINDOW_DAYS = 5

# Fixed, pre-registered statistical parameters (spec section 16/17's own
# "block lengthを結果に合わせて変更してはいけない" discipline, matching
# every V1 Phase 9+ diagnostic config) - never tuned against a result.
# Kept as plain module constants rather than a new YAML+manifest layer
# (spec section 2/28 emphasize reuse and avoiding unnecessary structure);
# these are V2-2's OWN new analysis parameters, distinct from - and
# never perturbing - Phase V2-1's frozen config_hash/code_hash.
TRADE_LEVEL_BOOTSTRAP_CONFIG = BootstrapConfig(n_resamples=10_000, seed=142, confidence_level=0.95)
DAY_CLUSTER_BOOTSTRAP_CONFIG = DayClusterBootstrapConfig(
    n_resamples=10_000, seed=144, confidence_level=0.95
)
BLOCK_BOOTSTRAP_CONFIG = BlockBootstrapConfig(
    block_length_days=5, n_resamples=10_000, seed=145, confidence_level=0.95
)
# backtest.permutation.permutation_test()'s argpartition-based null draw is
# O(n_population) per permutation - V1's own Signal permutation tests keep
# n_signal a small fraction of n_population (a rare Signal trigger vs the
# full population), so 10,000 permutations stays tractable there. A Score
# QUANTILE bucket is a full 20% of the population (Q5 alone is ~600K rows
# at Full Universe scale) - measured directly (see research/phase_v2_2_report.md's
# own timing note) at ~53ms/permutation for a 3M-row population, so 10,000
# permutations would take ~9 MINUTES per single call, and this module
# calls permutation_test() ~22 times (2 primary + ~20 FDR-family sweep) -
# tens of GB of pure compute time if left at V1's default. Reduced here
# BEFORE any real run completed (a computational-tractability fix
# discovered during implementation testing, not a result-driven loosening
# of significance criteria - the primary Q1/Q5 test still gets a fully
# adequate p-value resolution of 0.001; the FDR sweep's lighter 0.005
# resolution is appropriate for its own tiered-depth role, matching this
# module's own "tiered analysis depth" precedent for bootstrap/permutation
# across Holding Periods).
PERMUTATION_CONFIG = PermutationConfig(n_permutations=1_000, seed=143, forward_window=5)
FDR_SWEEP_PERMUTATION_CONFIG = PermutationConfig(n_permutations=300, seed=146, forward_window=5)
MARKET_REGIME_CONFIG = MarketRegimeConfig()  # V1's own defaults, reused as-is (spec section 12)


def _return_col(window_days: int) -> str:
    return f"forward_return_{window_days}d"


@dataclass(frozen=True)
class LightWindowResult:
    window_days: int
    n_resolved: int
    n_excluded_as_implausible: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None
    monotonicity: MonotonicityResult
    ic_summary: ICSummary
    topn_results: list[TopNResult]


@dataclass(frozen=True)
class PrimaryWindowResult:
    window_days: int
    light: LightWindowResult
    trade_level_bootstrap: BootstrapDiffResult
    day_cluster_bootstrap: SpreadBootstrapResult
    block_bootstrap: SpreadBootstrapResult
    permutation: list[BucketPermutationResult]
    cost_sensitivity: list[CostSensitivityResult]
    event_exclusion: dict[str, SliceResult]
    year_by_year: dict[int, SliceResult]
    concentration: ConcentrationResult
    regime: list[RegimeSliceResult]


@dataclass(frozen=True)
class V2_2Report:
    n_tickers: int
    n_rows_total: int
    n_rows_scored: int
    date_range_start: object
    date_range_end: object
    primary: PrimaryWindowResult
    holding_period_comparison: dict[int, LightWindowResult]
    fdr_results: dict[str, FDRResult] = field(default_factory=dict)


def _light_window_analysis(scored: pd.DataFrame, window_days: int) -> LightWindowResult:
    return_col = _return_col(window_days)
    raw = scored.dropna(subset=[return_col])
    clean = exclude_implausible_returns(raw, return_col)
    n_excluded = len(raw) - len(clean)

    bucket_stats = compute_quantile_bucket_stats(clean, "score_bucket", return_col, window_days)
    bucket_means = {b.bucket: b.stats.mean_return for b in bucket_stats}

    daily_ic = compute_daily_spearman_ic(clean, "total_score", return_col)
    topn_results = [
        summarize_topn(
            compute_topn_daily_returns(clean, n, return_col), n, window_days
        )
        for n in TOP_N_VALUES
    ]

    return LightWindowResult(
        window_days=window_days,
        n_resolved=len(clean),
        n_excluded_as_implausible=n_excluded,
        bucket_stats=bucket_stats,
        q5_q1_spread=compute_q5_q1_spread(bucket_stats),
        monotonicity=compute_monotonicity(bucket_means, window_days),
        ic_summary=summarize_ic(daily_ic, window_days),
        topn_results=topn_results,
    )


def _primary_window_analysis(
    scored: pd.DataFrame, window_days: int, topix_ohlcv: pd.DataFrame
) -> PrimaryWindowResult:
    return_col = _return_col(window_days)
    raw = scored.dropna(subset=[return_col])
    clean = exclude_implausible_returns(raw, return_col)

    light = _light_window_analysis(scored, window_days)

    q5 = clean[clean["score_bucket"] == "Q5"]
    q1 = clean[clean["score_bucket"] == "Q1"]

    trade_level = bootstrap_diff_ci(
        q5[return_col].to_numpy(), q1[return_col].to_numpy(), TRADE_LEVEL_BOOTSTRAP_CONFIG
    )
    q5_for_bootstrap = q5.rename(columns={return_col: "return"})
    q1_for_bootstrap = q1.rename(columns={return_col: "return"})
    day_cluster = day_cluster_spread_bootstrap(
        q5_for_bootstrap, q1_for_bootstrap, DAY_CLUSTER_BOOTSTRAP_CONFIG
    )
    block = block_spread_bootstrap(q5_for_bootstrap, q1_for_bootstrap, BLOCK_BOOTSTRAP_CONFIG)

    permutation = run_bucket_permutation_tests(clean, return_col, window_days, PERMUTATION_CONFIG)

    cost_sensitivity = compute_cost_sensitivity(q5[return_col])
    event_exclusion = analyze_event_exclusion(clean, return_col, window_days)
    year_by_year = analyze_year_by_year(clean, return_col, window_days)
    concentration = compute_concentration(clean, return_col, window_days)
    regime_df = build_regime_series(topix_ohlcv, MARKET_REGIME_CONFIG)
    regime = analyze_by_regime(clean, regime_df, return_col, window_days)

    return PrimaryWindowResult(
        window_days=window_days,
        light=light,
        trade_level_bootstrap=trade_level,
        day_cluster_bootstrap=day_cluster,
        block_bootstrap=block,
        permutation=permutation,
        cost_sensitivity=cost_sensitivity,
        event_exclusion=event_exclusion,
        year_by_year=year_by_year,
        concentration=concentration,
        regime=regime,
    )


def run_v2_2_validation(config: V2Config, tickers: list[str] | None = None) -> V2_2Report:
    ranked = run_v2_ranking(config, tickers=tickers)
    n_tickers = ranked["ticker"].nunique()
    n_rows_total = len(ranked)

    scored = ranked.dropna(subset=["total_score"]).copy()
    scored["score_bucket"] = assign_quantile_buckets(scored["total_score"])
    n_rows_scored = len(scored)

    topix_ohlcv = load_ohlcv("TOPIX", config.source_processed_dir)

    logger.info("V2-2: primary window (%dd) full battery", PRIMARY_WINDOW_DAYS)
    primary = _primary_window_analysis(scored, PRIMARY_WINDOW_DAYS, topix_ohlcv)

    holding_period_comparison: dict[int, LightWindowResult] = {}
    for window in FORWARD_WINDOWS:
        if window == PRIMARY_WINDOW_DAYS:
            holding_period_comparison[window] = primary.light
            continue
        logger.info("V2-2: light analysis for %dd window", window)
        holding_period_comparison[window] = _light_window_analysis(scored, window)

    # --- Multiple Testing / FDR (spec section 18) -----------------------
    # Family = every permutation test this Phase runs: Holding Period
    # (Q1/Q5 at each window), Regime (Q5 within each regime), Candidate
    # size (Top-5/10/20 vs population) - matching spec section 18's own
    # named family list, fixed here rather than added to after seeing
    # which axis looked significant.
    return_col_5d = _return_col(PRIMARY_WINDOW_DAYS)
    population = scored[return_col_5d].dropna().to_numpy()
    p_values: dict[str, float] = {}

    # The primary window's Q1/Q5 pair was already computed above at full
    # PERMUTATION_CONFIG depth (primary.permutation) - reused here rather
    # than recomputed at the lighter FDR_SWEEP_PERMUTATION_CONFIG depth,
    # both to avoid wasteful duplicate work and to avoid mixing two
    # different resample counts for the same (window, bucket) cell.
    for bp in primary.permutation:
        p_values[f"holding_period:{PRIMARY_WINDOW_DAYS}d:{bp.bucket_label}"] = bp.result.p_value

    for window in FORWARD_WINDOWS:
        if window == PRIMARY_WINDOW_DAYS:
            continue
        window_return_col = _return_col(window)
        window_clean = exclude_implausible_returns(
            scored.dropna(subset=[window_return_col]), window_return_col
        )
        window_population = window_clean[window_return_col].dropna().to_numpy()
        for bucket in ("Q1", "Q5"):
            bucket_returns = window_clean.loc[
                window_clean["score_bucket"] == bucket, window_return_col
            ].dropna().to_numpy()
            p = permutation_test(
                bucket_returns, window_population, FDR_SWEEP_PERMUTATION_CONFIG
            ).p_value
            if p == p:
                p_values[f"holding_period:{window}d:{bucket}"] = p

    q5_clean_5d = exclude_implausible_returns(
        scored.dropna(subset=[return_col_5d]), return_col_5d
    )
    regime_series = build_regime_series(topix_ohlcv, MARKET_REGIME_CONFIG)
    q5_with_regime = q5_clean_5d[q5_clean_5d["score_bucket"] == "Q5"].merge(
        regime_series, on="date", how="left"
    )
    for rs in primary.regime:
        regime_q5_returns = q5_with_regime.loc[
            q5_with_regime["regime"] == rs.regime, return_col_5d
        ].dropna().to_numpy()
        p = permutation_test(regime_q5_returns, population, FDR_SWEEP_PERMUTATION_CONFIG).p_value
        if p == p:
            p_values[f"regime:{rs.regime}:Q5"] = p

    for topn_result in primary.light.topn_results:
        topn_returns = pd.Series(
            [d.equal_weight_return for d in topn_result.daily_returns], dtype=float
        ).dropna().to_numpy()
        p = permutation_test(topn_returns, population, FDR_SWEEP_PERMUTATION_CONFIG).p_value
        if p == p:
            p_values[f"candidate_size:top{topn_result.n}"] = p

    fdr_results = benjamini_hochberg_correction(p_values)

    return V2_2Report(
        n_tickers=n_tickers,
        n_rows_total=n_rows_total,
        n_rows_scored=n_rows_scored,
        date_range_start=ranked["date"].min(),
        date_range_end=ranked["date"].max(),
        primary=primary,
        holding_period_comparison=holding_period_comparison,
        fdr_results=fdr_results,
    )
