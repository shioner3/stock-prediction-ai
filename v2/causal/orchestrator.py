"""Phase V2-3 orchestrator (spec section 39 STEP 6-21): builds the V2-1
ranked panel + per-feature percentiles ONCE, assigns Score Q1-Q5 buckets
(scoring.validation.assign_quantile_buckets, unmodified), then runs the
full causal-decomposition battery focused on the PRIMARY 5d window (the
same "primary window" convention every prior Phase - V1 Phase 6+, Phase
V2-2 - already uses), with a lighter Holding-Period sweep across all 7
windows (spec sections 8/17/23's own explicit multi-window requirements).

Every statistical parameter below is fixed BEFORE the real run (spec
section 37's "結果を得るために新しいthreshold/除外ルールが必要" stop
condition exists precisely to prevent tuning these after seeing results).
The permutation n_permutations values directly reuse Phase V2-2's own
already-documented tractability fix (research/phase_v2_2_report.md
section 26's Limitations note): a Score-derived quantile bucket is a full
20% of the ~3M-row population, so 10,000 permutations per call is not
tractable at this scale - reduced to 1,000 for the true primary test and
300 for the FDR sweep family, exactly as Phase V2-2 established.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from backtest.block_bootstrap import BlockBootstrapResult, block_bootstrap
from backtest.bootstrap import BootstrapResult, bootstrap_ci
from backtest.day_cluster_bootstrap import DayClusterBootstrapResult, day_cluster_bootstrap
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import PermutationResult, permutation_test
from config.loader import (
    BlockBootstrapConfig,
    BootstrapConfig,
    DayClusterBootstrapConfig,
    MarketRegimeConfig,
    PermutationConfig,
)
from pipeline.run_phase9_analysis import YEARS
from scoring.validation import assign_quantile_buckets
from storage.parquet_store import load_ohlcv
from targets.forward_returns import FORWARD_WINDOWS
from v2.causal.feature_stats import (
    FEATURE_LIST,
    CategoryContribution,
    FeatureBucketProfile,
    compute_category_contribution,
    compute_category_correlation_matrix,
    compute_feature_bucket_profile,
    compute_feature_percentiles,
)
from v2.causal.heterogeneity import HeterogeneityResult, analyze_q1_heterogeneity
from v2.causal.interaction import (
    CATEGORY_REPRESENTATIVE_FEATURE,
    PairwiseInteractionResult,
    ScoreFeatureCrosstab,
    pairwise_feature_interaction,
    score_feature_crosstab,
)
from v2.causal.placebo import PlaceboLagResult, run_timing_placebo
from v2.causal.random_control import RandomControlResult, run_random_control
from v2.causal.segment import (
    GroupReturnStats,
    LiquidityProfile,
    attach_liquidity_columns,
    attach_segment_columns,
    compute_group_stats,
    compute_liquidity_profile,
    load_ticker_segment_map,
)
from v2.causal.single_feature import (
    FeatureDirection,
    SingleFeatureWindowResult,
    analyze_all_features,
    assign_feature_buckets,
    classify_feature_direction,
)
from v2.causal.stability import DailyStabilityResult, compute_daily_stability
from v2.config.loader import V2Config
from v2.pipeline import run_v2_ranking
from v2.stats import (
    QuantileBucketStats,
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    exclude_implausible_returns,
)
from v2.validation.concentration import ConcentrationResult, compute_concentration
from v2.validation.event_year_analysis import (
    AUG_2024_EVENT_END,
    AUG_2024_EVENT_START,
    SliceResult,
    analyze_event_exclusion,
    analyze_year_by_year,
)
from v2.validation.regime import REGIMES, RegimeSliceResult, analyze_by_regime, build_regime_series
from v2.validation.spread_bootstrap import (
    SpreadBootstrapResult,
    block_spread_bootstrap,
    day_cluster_spread_bootstrap,
)

logger = logging.getLogger(__name__)

PRIMARY_WINDOW_DAYS = 5

TRADE_LEVEL_BOOTSTRAP_CONFIG = BootstrapConfig(n_resamples=10_000, seed=242, confidence_level=0.95)
DAY_CLUSTER_BOOTSTRAP_CONFIG = DayClusterBootstrapConfig(
    n_resamples=10_000, seed=244, confidence_level=0.95
)
BLOCK_BOOTSTRAP_CONFIG = BlockBootstrapConfig(
    block_length_days=5, n_resamples=10_000, seed=245, confidence_level=0.95
)
# See module docstring - reuses Phase V2-2's own tractability finding.
PERMUTATION_CONFIG = PermutationConfig(n_permutations=1_000, seed=243, forward_window=5)
FDR_SWEEP_PERMUTATION_CONFIG = PermutationConfig(n_permutations=300, seed=246, forward_window=5)
MARKET_REGIME_CONFIG = MarketRegimeConfig()


def _return_col(window_days: int) -> str:
    return f"forward_return_{window_days}d"


def _clean_window(scored: pd.DataFrame, window_days: int) -> pd.DataFrame:
    return_col = _return_col(window_days)
    return exclude_implausible_returns(scored.dropna(subset=[return_col]), return_col)


@dataclass(frozen=True)
class V2_3Report:
    n_tickers: int
    n_rows_total: int
    n_rows_scored: int
    date_range_start: object
    date_range_end: object

    # sections 6/7
    feature_profile_q1: list[FeatureBucketProfile]
    feature_profile_q5: list[FeatureBucketProfile]
    category_contribution_q1: list[CategoryContribution]
    category_correlation_full: pd.DataFrame
    category_correlation_q1: pd.DataFrame

    # section 8/9 - keyed by window_days
    single_feature_by_window: dict[int, list[SingleFeatureWindowResult]]
    feature_directions: dict[str, FeatureDirection]

    # section 11
    heterogeneity: HeterogeneityResult

    # section 12/13
    score_feature_crosstabs: dict[str, ScoreFeatureCrosstab]
    pairwise_interactions: list[PairwiseInteractionResult]

    # section 14/15
    regime_results: list[RegimeSliceResult]

    # section 16
    year_results: dict[int, SliceResult]

    # section 17 - keyed by window_days -> (bucket_stats, q5_q1_spread)
    holding_period_results: dict[int, tuple[list[QuantileBucketStats], float | None]]

    # section 18/19
    segment_stats: list[GroupReturnStats]
    sector_stats: list[GroupReturnStats]
    scale_stats: list[GroupReturnStats]
    liquidity_q1: LiquidityProfile
    liquidity_q5: LiquidityProfile

    # section 20
    concentration_q1: ConcentrationResult

    # section 21
    event_exclusion: dict[str, SliceResult]

    # section 22
    q1_trade_level_bootstrap: BootstrapResult
    q1_day_cluster_bootstrap: DayClusterBootstrapResult
    q1_block_bootstrap: BlockBootstrapResult
    spread_day_cluster_bootstrap: SpreadBootstrapResult
    spread_block_bootstrap: SpreadBootstrapResult

    # section 23 - keyed by window_days
    permutation_by_window: dict[int, PermutationResult]

    # section 27
    stability: DailyStabilityResult

    # section 28/29
    random_control: RandomControlResult

    # section 24
    fdr_results: dict[str, FDRResult] = field(default_factory=dict)

    # section 26
    placebo_results: list[PlaceboLagResult] = field(default_factory=list)


def build_scored_panel(config: V2Config, tickers: list[str] | None = None) -> pd.DataFrame:
    """STEP 6 preamble: V2-1's ranked panel (frozen) + per-feature
    percentiles (v2/causal/feature_stats.py) + Score Q1-Q5 buckets - built
    ONCE, reused by every section below and by STEP 3's reproducibility
    check (spec section 35's "同じFeatureについて何度も再計算しない").
    """
    ranked = run_v2_ranking(config, tickers=tickers)
    ranked = compute_feature_percentiles(ranked)
    scored = ranked.dropna(subset=["total_score"]).copy()
    scored["score_bucket"] = assign_quantile_buckets(scored["total_score"])
    return ranked, scored


def run_v2_3_causal_analysis_on_panel(
    config: V2Config, ranked: pd.DataFrame, scored: pd.DataFrame
) -> V2_3Report:
    """Runs the full battery against an ALREADY-BUILT (ranked, scored)
    panel pair - the CLI script builds this panel once (build_scored_panel()),
    runs STEP 3's V2-2 reproducibility check against it, and only then
    calls this function, so the panel is never built twice (spec section
    35's "同じFeatureについて何度も再計算しない").
    """
    n_tickers = int(ranked["ticker"].nunique())
    n_rows_total = len(ranked)
    n_rows_scored = len(scored)

    return_col_5d = _return_col(PRIMARY_WINDOW_DAYS)
    clean5 = _clean_window(scored, PRIMARY_WINDOW_DAYS)

    logger.info("V2-3: feature decomposition / contribution")
    feature_profile_q1 = compute_feature_bucket_profile(clean5, bucket_label="Q1")
    feature_profile_q5 = compute_feature_bucket_profile(clean5, bucket_label="Q5")
    category_contribution_q1 = compute_category_contribution(clean5, bucket_label="Q1")
    category_correlation_full = compute_category_correlation_matrix(clean5)
    category_correlation_q1 = compute_category_correlation_matrix(
        clean5, bucket_col="score_bucket", bucket_label="Q1"
    )

    logger.info("V2-3: single-feature analysis across %d windows", len(FORWARD_WINDOWS))
    single_feature_by_window = {}
    for window in FORWARD_WINDOWS:
        col = _return_col(window)
        window_clean = _clean_window(scored, window)
        single_feature_by_window[window] = analyze_all_features(window_clean, col, window)
    primary_single_feature = single_feature_by_window[PRIMARY_WINDOW_DAYS]
    feature_directions = {
        r.feature: classify_feature_direction(r) for r in primary_single_feature
    }

    logger.info("V2-3: Q1 internal heterogeneity")
    heterogeneity = analyze_q1_heterogeneity(clean5, return_col_5d, PRIMARY_WINDOW_DAYS)

    logger.info("V2-3: Score x Feature / pairwise Feature interaction")
    score_feature_crosstabs: dict[str, ScoreFeatureCrosstab] = {}
    representative_features = set(CATEGORY_REPRESENTATIVE_FEATURE.values())
    for _category, feature, _higher_is_better in FEATURE_LIST:
        buckets = ("Q1", "Q2", "Q3", "Q4", "Q5") if feature in representative_features else ("Q1",)
        score_feature_crosstabs[feature] = score_feature_crosstab(
            clean5, feature, return_col_5d, PRIMARY_WINDOW_DAYS, score_buckets=buckets
        )
    pairwise_interactions = [
        pairwise_feature_interaction(clean5, fa, fb, return_col_5d, PRIMARY_WINDOW_DAYS)
        for fa, fb in combinations(CATEGORY_REPRESENTATIVE_FEATURE.values(), 2)
    ]

    logger.info("V2-3: Regime / Year / Holding Period")
    topix_ohlcv = load_ohlcv("TOPIX", config.source_processed_dir)
    regime_df = build_regime_series(topix_ohlcv, MARKET_REGIME_CONFIG)
    regime_results = analyze_by_regime(clean5, regime_df, return_col_5d, PRIMARY_WINDOW_DAYS)
    year_results = analyze_year_by_year(clean5, return_col_5d, PRIMARY_WINDOW_DAYS)

    holding_period_results: dict[int, tuple[list[QuantileBucketStats], float | None]] = {}
    for window in FORWARD_WINDOWS:
        col = _return_col(window)
        window_clean = clean5 if window == PRIMARY_WINDOW_DAYS else _clean_window(scored, window)
        bucket_stats = compute_quantile_bucket_stats(window_clean, "score_bucket", col, window)
        holding_period_results[window] = (bucket_stats, compute_q5_q1_spread(bucket_stats))

    logger.info("V2-3: Sector / Segment / Size / Liquidity")
    segment_map = load_ticker_segment_map()
    clean5_seg = attach_segment_columns(clean5, segment_map)
    segment_stats = compute_group_stats(clean5_seg, "market_segment", return_col_5d)
    sector_stats = compute_group_stats(clean5_seg, "sector33", return_col_5d)
    scale_stats = compute_group_stats(clean5_seg, "scale", return_col_5d)
    clean5_liq = attach_liquidity_columns(clean5)
    liquidity_q1 = compute_liquidity_profile(clean5_liq, bucket_label="Q1")
    liquidity_q5 = compute_liquidity_profile(clean5_liq, bucket_label="Q5")

    logger.info("V2-3: Concentration / Event Exclusion")
    concentration_q1 = compute_concentration(
        clean5, return_col_5d, PRIMARY_WINDOW_DAYS, bucket="Q1"
    )
    event_exclusion = analyze_event_exclusion(clean5, return_col_5d, PRIMARY_WINDOW_DAYS)

    logger.info("V2-3: Bootstrap (Q1-alone + Q5-Q1 spread)")
    q1_rows = clean5[clean5["score_bucket"] == "Q1"]
    q5_rows = clean5[clean5["score_bucket"] == "Q5"]
    q1_trade_level_bootstrap = bootstrap_ci(
        q1_rows[return_col_5d].to_numpy(), "mean_return", TRADE_LEVEL_BOOTSTRAP_CONFIG
    )
    q1_for_cluster = q1_rows[["date", return_col_5d]].rename(columns={return_col_5d: "return"})
    q1_day_cluster_bootstrap = day_cluster_bootstrap(
        q1_for_cluster, "mean_return", DAY_CLUSTER_BOOTSTRAP_CONFIG, date_col="date"
    )
    q1_block_bootstrap = block_bootstrap(
        q1_for_cluster, "mean_return", BLOCK_BOOTSTRAP_CONFIG, date_col="date"
    )
    q5_for_spread = q5_rows.rename(columns={return_col_5d: "return"})
    q1_for_spread = q1_rows.rename(columns={return_col_5d: "return"})
    spread_day_cluster_bootstrap = day_cluster_spread_bootstrap(
        q5_for_spread, q1_for_spread, DAY_CLUSTER_BOOTSTRAP_CONFIG
    )
    spread_block_bootstrap = block_spread_bootstrap(
        q5_for_spread, q1_for_spread, BLOCK_BOOTSTRAP_CONFIG
    )

    logger.info("V2-3: Permutation (Q1 vs population, %d windows)", len(FORWARD_WINDOWS))
    permutation_by_window: dict[int, PermutationResult] = {}
    for window in FORWARD_WINDOWS:
        col = _return_col(window)
        window_clean = clean5 if window == PRIMARY_WINDOW_DAYS else _clean_window(scored, window)
        population = window_clean[col].dropna().to_numpy()
        q1_values = window_clean.loc[window_clean["score_bucket"] == "Q1", col].dropna().to_numpy()
        cfg = PERMUTATION_CONFIG if window == PRIMARY_WINDOW_DAYS else FDR_SWEEP_PERMUTATION_CONFIG
        permutation_by_window[window] = permutation_test(q1_values, population, cfg)

    logger.info("V2-3: FDR across Holding Period / Feature / Regime / Year families")
    p_values: dict[str, float] = {}
    for window, result in permutation_by_window.items():
        p_values[f"holding_period:{window}d:Q1"] = result.p_value

    population_5d = clean5[return_col_5d].dropna().to_numpy()
    for _category, feature, _higher_is_better in FEATURE_LIST:
        feature_buckets = assign_feature_buckets(clean5, feature)
        feature_q1_values = clean5.loc[feature_buckets == "Q1", return_col_5d].dropna().to_numpy()
        p = permutation_test(feature_q1_values, population_5d, FDR_SWEEP_PERMUTATION_CONFIG).p_value
        if p == p:  # not NaN
            p_values[f"feature:{feature}:Q1"] = p

    q1_with_regime = q1_rows.merge(regime_df, on="date", how="left")
    for regime_name in REGIMES:
        regime_values = q1_with_regime.loc[
            q1_with_regime["regime"] == regime_name, return_col_5d
        ].dropna().to_numpy()
        p = permutation_test(regime_values, population_5d, FDR_SWEEP_PERMUTATION_CONFIG).p_value
        if p == p:
            p_values[f"regime:{regime_name}:Q1"] = p

    for year in YEARS:
        year_df = clean5[clean5["date"].apply(lambda d: d.year) == year]
        if year_df.empty:
            continue
        year_population = year_df[return_col_5d].dropna().to_numpy()
        year_q1_mask = year_df["score_bucket"] == "Q1"
        year_q1_values = year_df.loc[year_q1_mask, return_col_5d].dropna().to_numpy()
        p = permutation_test(year_q1_values, year_population, FDR_SWEEP_PERMUTATION_CONFIG).p_value
        if p == p:
            p_values[f"year:{year}:Q1"] = p

    fdr_results = benjamini_hochberg_correction(p_values)

    logger.info("V2-3: Timing Placebo / Cross-sectional Stability / Random Control")
    placebo_results = run_timing_placebo(clean5, return_col_5d)
    stability = compute_daily_stability(clean5, return_col_5d, PRIMARY_WINDOW_DAYS)
    random_control = run_random_control(clean5, return_col_5d, PRIMARY_WINDOW_DAYS)

    return V2_3Report(
        n_tickers=n_tickers,
        n_rows_total=n_rows_total,
        n_rows_scored=n_rows_scored,
        date_range_start=ranked["date"].min(),
        date_range_end=ranked["date"].max(),
        feature_profile_q1=feature_profile_q1,
        feature_profile_q5=feature_profile_q5,
        category_contribution_q1=category_contribution_q1,
        category_correlation_full=category_correlation_full,
        category_correlation_q1=category_correlation_q1,
        single_feature_by_window=single_feature_by_window,
        feature_directions=feature_directions,
        heterogeneity=heterogeneity,
        score_feature_crosstabs=score_feature_crosstabs,
        pairwise_interactions=pairwise_interactions,
        regime_results=regime_results,
        year_results=year_results,
        holding_period_results=holding_period_results,
        segment_stats=segment_stats,
        sector_stats=sector_stats,
        scale_stats=scale_stats,
        liquidity_q1=liquidity_q1,
        liquidity_q5=liquidity_q5,
        concentration_q1=concentration_q1,
        event_exclusion=event_exclusion,
        q1_trade_level_bootstrap=q1_trade_level_bootstrap,
        q1_day_cluster_bootstrap=q1_day_cluster_bootstrap,
        q1_block_bootstrap=q1_block_bootstrap,
        spread_day_cluster_bootstrap=spread_day_cluster_bootstrap,
        spread_block_bootstrap=spread_block_bootstrap,
        permutation_by_window=permutation_by_window,
        fdr_results=fdr_results,
        placebo_results=placebo_results,
        stability=stability,
        random_control=random_control,
    )


def run_v2_3_causal_analysis(config: V2Config, tickers: list[str] | None = None) -> V2_3Report:
    """Convenience wrapper for tests/ad-hoc use: builds the panel AND runs
    the battery in one call. The CLI script does NOT use this - it calls
    build_scored_panel() once, runs STEP 3's reproducibility check, THEN
    run_v2_3_causal_analysis_on_panel() against the same panel.
    """
    ranked, scored = build_scored_panel(config, tickers=tickers)
    return run_v2_3_causal_analysis_on_panel(config, ranked, scored)


__all__ = [
    "AUG_2024_EVENT_START",
    "AUG_2024_EVENT_END",
    "PRIMARY_WINDOW_DAYS",
    "V2_3Report",
    "build_scored_panel",
    "run_v2_3_causal_analysis",
    "run_v2_3_causal_analysis_on_panel",
]
