"""Phase V3-4 orchestrator: ties together every decomposition/robustness
module in this package into one report, built entirely on top of Phase
V3-3's frozen pipeline (see `reproduce.py`'s hash verification). Runs
against the Primary combination (Model A, `target_raw_5d`) only, per
spec section 26 ("V3を良くすることではない...結果を見て仕様変更が必要に
なった場合はSTOP") - Model B/C and the 3 secondary Target Variants are
NOT re-decomposed this Phase (only the 3 secondary HORIZON targets, for
section 15's Holding Period comparison, which V3-3 already trained).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.market_regime import compute_market_regime
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import PermutationConfig, permutation_test
from config.loader import TransactionCostTierConfig
from scoring.validation import assign_quantile_buckets
from v3.robustness.aux_panel import attach_sector_and_scale, build_price_volume_panel
from v3.robustness.beta import compute_rolling_beta
from v3.robustness.cost_sensitivity import CostTierResult, run_cost_sensitivity
from v3.robustness.cross_sectional_decomp import (
    StructuralInvarianceCheck,
    check_structural_invariance,
    daily_market_component_correlation,
    run_cross_sectional_decomposition,
)
from v3.robustness.decision_v3_4 import (
    EdgeClassificationInputs,
    EdgeClassificationResult,
    classify_edge_source,
)
from v3.robustness.economic_significance import EconomicSignificance, compute_economic_significance
from v3.robustness.leave_one_out import (
    DayConcentrationRobustness,
    RegimeRobustness,
    StockConcentrationRobustness,
    YearRobustness,
    run_day_concentration_robustness,
    run_regime_robustness,
    run_stock_concentration_robustness,
    run_year_robustness,
)
from v3.robustness.market_decomposition import (
    VARIANT_BETA_ADJUSTED,
    VARIANT_MARKET_NEUTRALIZED,
    VARIANT_TOPIX_RELATIVE,
    ReturnVariantResult,
    build_return_variant_columns,
    run_market_decomposition,
)
from v3.robustness.matched_control import (
    MatchedControlResult,
    build_full_day_panel,
    run_matched_control_analysis,
)
from v3.robustness.sector_concentration import SectorConcentrationResult, run_sector_concentration
from v3.robustness.v3_3_reference import (
    MOMENTUM_BASELINE_Q5_Q1_SPREAD,
    PRIMARY_FDR_SIGNIFICANT,
    RANDOM_BASELINE_Q5_Q1_SPREAD,
    V2_SCORE_BASELINE_Q5_Q1_SPREAD,
)
from v3.validation.decision import V3_3Decision, V3_3DecisionInputs, classify_v3_3_decision
from v3.validation.ranking_metrics import RankingResult, evaluate_ranking
from v3.validation.regime_year_event import analyze_event_exclusion
from v3.validation.robustness import run_bucket_permutation_tests
from v3.validation.topn_portfolio import compute_topn_portfolio_metrics
from v3.validation.wfo_config import PERMUTATION_CONFIG, PRIMARY_TARGET_COL

HORIZON_BY_TARGET = {
    "target_raw_5d": 5, "target_raw_10d": 10, "target_raw_15d": 15, "target_raw_20d": 20,
}
FDR_SWEEP_CONFIG = PermutationConfig(n_permutations=300, seed=347, forward_window=5)


@dataclass(frozen=True)
class V3_4Report:
    primary_ranking: RankingResult  # reproducibility check vs V3-3's reported spread
    market_decomposition: dict[str, ReturnVariantResult]
    cross_sectional_decomposition: dict
    structural_invariance: StructuralInvarianceCheck
    market_component_correlation: float | None
    regime_robustness: RegimeRobustness
    day_concentration: DayConcentrationRobustness
    year_robustness: YearRobustness
    stock_concentration: StockConcentrationRobustness
    sector_concentration: SectorConcentrationResult
    matched_control: MatchedControlResult
    holding_period: dict[int, RankingResult]
    cost_sensitivity: dict[str, CostTierResult]
    economic_significance: dict[int, EconomicSignificance]
    variant_fdr_results: dict[str, FDRResult]
    v3_3_decision: V3_3Decision
    v3_3_decision_reasons: list[str]
    edge_classification: EdgeClassificationResult


def run_v3_4_analysis(
    dataset: pd.DataFrame,
    reproduced_predictions: dict[str, pd.DataFrame],
    tickers: list[str],
    topix_ohlcv: pd.DataFrame,
    market_regime_config,
    cost_tiers: list[TransactionCostTierConfig],
    v3_config,
) -> V3_4Report:
    primary = reproduced_predictions[PRIMARY_TARGET_COL]
    horizon = HORIZON_BY_TARGET[PRIMARY_TARGET_COL]

    primary_ranking = evaluate_ranking(primary, horizon)

    beta_panel = compute_rolling_beta(dataset)
    price_volume_panel = build_price_volume_panel(tickers, v3_config)
    sector_map_predictions = attach_sector_and_scale(primary[["date", "ticker"]].drop_duplicates())
    # sector33/market_segment/scale keyed by ticker only (JPX's classification
    # is not date-varying in the local cache) - reused as a ticker -> segment
    # lookup by every module below.
    sector_cols = ["ticker", "market_segment", "sector33", "scale"]
    sector_map = sector_map_predictions[sector_cols].drop_duplicates(subset=["ticker"])

    primary_with_variants = build_return_variant_columns(primary, dataset, beta_panel, sector_map)
    market_decomp = run_market_decomposition(primary_with_variants, horizon)

    cross_sectional = run_cross_sectional_decomposition(primary, horizon)
    structural_invariance = check_structural_invariance(cross_sectional)
    market_component_corr = daily_market_component_correlation(primary)

    regime_df = compute_market_regime(topix_ohlcv, market_regime_config)
    regime_robust = run_regime_robustness(primary, regime_df, horizon)
    day_concentration = run_day_concentration_robustness(primary, horizon)
    year_robust = run_year_robustness(primary, horizon)
    stock_concentration = run_stock_concentration_robustness(primary, horizon)

    primary_with_sector = primary.merge(sector_map, on="ticker", how="left")
    sector_concentration = run_sector_concentration(primary_with_sector, horizon)

    full_day_panel = build_full_day_panel(dataset, price_volume_panel, sector_map)
    valid_primary = primary.dropna(subset=["prediction", "actual"]).copy()
    valid_primary["_bucket"] = assign_quantile_buckets(valid_primary["prediction"])
    q5_primary = valid_primary[valid_primary["_bucket"] == "Q5"]
    matched_control = run_matched_control_analysis(q5_primary, full_day_panel)

    holding_period = {
        HORIZON_BY_TARGET[target_col]: evaluate_ranking(df, HORIZON_BY_TARGET[target_col])
        for target_col, df in reproduced_predictions.items()
    }

    cost_result = run_cost_sensitivity(primary, cost_tiers, horizon)

    topn_metrics = {
        n: compute_topn_portfolio_metrics(primary, n, "actual", horizon) for n in (5, 10, 20)
    }
    economic_significance = {n: compute_economic_significance(m) for n, m in topn_metrics.items()}

    # Section 13: Permutation (+ FDR) on the market-timing-decomposition
    # variants and on Top-N - a NEW, explicitly-recorded test family,
    # distinct from V3-3's own 16-test family (v3_3_reference.py).
    p_values: dict[str, float] = {}
    for variant in (VARIANT_BETA_ADJUSTED, VARIANT_TOPIX_RELATIVE, VARIANT_MARKET_NEUTRALIZED):
        bp = run_bucket_permutation_tests(
            primary_with_variants, actual_col=f"actual_{variant}", config=FDR_SWEEP_CONFIG
        )
        for r in bp:
            if r.bucket_label == "Q5":
                p_values[f"variant:{variant}:Q5"] = r.result.p_value

    full_population = primary["actual"].dropna().to_numpy()
    for n, metrics in topn_metrics.items():
        topn_returns = pd.Series(
            [d.equal_weight_return for d in metrics.base.daily_returns], dtype=float
        ).dropna().to_numpy()
        if len(topn_returns) > 0:
            p = permutation_test(topn_returns, full_population, FDR_SWEEP_CONFIG).p_value
            if p == p:  # not NaN
                p_values[f"topn:{n}"] = p
    variant_fdr_results = benjamini_hochberg_correction(p_values) if p_values else {}

    decision_inputs = _build_v3_4_decision_inputs(primary, primary_ranking)
    v3_3_decision_result = classify_v3_3_decision(decision_inputs)

    edge_inputs = EdgeClassificationInputs(
        orig_q5_q1_spread=primary_ranking.q5_q1_spread,
        beta_adjusted_q5_q1_spread=market_decomp[VARIANT_BETA_ADJUSTED].q5_q1_spread,
        topix_relative_q5_q1_spread=market_decomp[VARIANT_TOPIX_RELATIVE].q5_q1_spread,
        bear_excluded_q5_q1_spread=regime_robust.leave_one_out["excl_BEAR"].ranking.q5_q1_spread,
        day_top20_excluded_q5_q1_spread=day_concentration.top_k_exclusion["top20"].ranking.q5_q1_spread,
    )
    edge_classification = classify_edge_source(edge_inputs)

    return V3_4Report(
        primary_ranking=primary_ranking, market_decomposition=market_decomp,
        cross_sectional_decomposition=cross_sectional, structural_invariance=structural_invariance,
        market_component_correlation=market_component_corr, regime_robustness=regime_robust,
        day_concentration=day_concentration, year_robustness=year_robust,
        stock_concentration=stock_concentration, sector_concentration=sector_concentration,
        matched_control=matched_control, holding_period=holding_period,
        cost_sensitivity=cost_result, economic_significance=economic_significance,
        variant_fdr_results=variant_fdr_results, v3_3_decision=v3_3_decision_result.decision,
        v3_3_decision_reasons=v3_3_decision_result.reasons, edge_classification=edge_classification,
    )


def _build_v3_4_decision_inputs(
    primary: pd.DataFrame, primary_ranking: RankingResult,
) -> V3_3DecisionInputs:
    """Rebuilds V3-3's own decision inputs from the REPRODUCED Primary
    predictions (spec section 18: reapply the SAME framework, no new
    thresholds). Bootstrap CIs / Q5 permutation / event-exclusion are
    RECOMPUTED here (cheap, deterministic given the hash-verified-
    identical data). Baseline spreads and the 16-test FDR significance
    flag are REUSED from V3-3's own published report (`v3_3_reference.py`)
    rather than rebuilt, since doing so would require re-running Model
    B/C and the 6 secondary Target combinations this Phase does not touch.
    """
    from v3.validation.robustness import bootstrap_q5_q1_spread

    per_window_spreads = []
    for _idx, group in primary.groupby("window_index"):
        r = evaluate_ranking(group, 5)
        if r.q5_q1_spread is not None:
            per_window_spreads.append(r.q5_q1_spread)
    window_direction_agreement = (
        sum(1 for s in per_window_spreads if s > 0) / len(per_window_spreads)
        if per_window_spreads else None
    )

    spread_bootstrap = bootstrap_q5_q1_spread(primary)
    q5_perm = next(
        (r.result.p_value for r in run_bucket_permutation_tests(primary, config=PERMUTATION_CONFIG)
         if r.bucket_label == "Q5"), None,
    )
    event = analyze_event_exclusion(primary, 5)
    event_spreads = [
        s.q5_q1_spread for label, s in event.items()
        if label != "full_period" and s.q5_q1_spread is not None
    ]
    survives_event = all(s > 0 for s in event_spreads) if event_spreads else None

    top5 = compute_topn_portfolio_metrics(primary, 5, "actual", 5)
    top10 = compute_topn_portfolio_metrics(primary, 10, "actual", 5)
    top20 = compute_topn_portfolio_metrics(primary, 20, "actual", 5)

    return V3_3DecisionInputs(
        n_windows=len(per_window_spreads),
        primary_q5_q1_spread=primary_ranking.q5_q1_spread,
        rank_ic_mean=primary_ranking.ic_summary.mean_ic,
        window_direction_agreement=window_direction_agreement,
        day_cluster_ci_low=spread_bootstrap.day_cluster.ci_low,
        block_ci_low=spread_bootstrap.block.ci_low,
        q5_permutation_p=q5_perm,
        fdr_significant=PRIMARY_FDR_SIGNIFICANT,
        survives_event_exclusion=survives_event,
        top5_mean_return=top5.base.stats.mean_return,
        top10_mean_return=top10.base.stats.mean_return,
        top20_mean_return=top20.base.stats.mean_return,
        random_baseline_spread=RANDOM_BASELINE_Q5_Q1_SPREAD,
        momentum_baseline_spread=MOMENTUM_BASELINE_Q5_Q1_SPREAD,
        v2_score_baseline_spread=V2_SCORE_BASELINE_Q5_Q1_SPREAD,
    )
