"""Phase V3-5 orchestrator: for each of 4 Target definitions (A=Raw,
B=TOPIX-relative, C=Beta-adjusted Residual, D=Sector-relative) x 4
Horizons (5/10/15/20d) = 16 combinations, evaluates Q1-Q5/Rank IC/Pearson
IC/Top-N (spec section 33: ALL 16 reported, never narrowed to "the best
one"). The 4 PRIMARY combinations (each definition at 5d, spec's own
implicit Horizon choice matching V1/V2/V3-3's convention) get the FULL
battery (Regime/Event/Year/Concentration/Bootstrap/Permutation/Matched
Control/Cost/Economic-significance) - the other 12 get the light
battery only (Q1-Q5/Rank IC/Pearson IC/Top-N), mirroring the tiered-depth
precedent V2-2/V3-3/V3-4 already established.

Every statistics primitive below is a plain reuse of V1/V2/V3-3/V3-4
code - only the Market Neutralization comparison table, residual_
strength ratios, and the FDR family/Edge Classification wiring are new
to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.market_regime import compute_market_regime
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import PermutationConfig
from config.loader import TransactionCostTierConfig
from scoring.validation import assign_quantile_buckets
from v3.residual.decision_v3_5 import (
    EdgeClassificationInputs,
    EdgeClassificationResult,
    classify_edge_source,
)
from v3.residual.ic_pearson import summarize_pearson_ic
from v3.residual.reproduce import TARGET_A_RAW, TARGET_DEFINITIONS
from v3.residual.residual_strength import residual_strength
from v3.robustness.aux_panel import attach_sector_and_scale, build_price_volume_panel
from v3.robustness.cost_sensitivity import CostTierResult, run_cost_sensitivity
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
from v3.robustness.matched_control import (
    MatchedControlResult,
    build_full_day_panel,
    run_matched_control_analysis,
)
from v3.robustness.sector_concentration import SectorConcentrationResult, run_sector_concentration
from v3.validation.ranking_metrics import RankingResult, evaluate_ranking
from v3.validation.robustness import (
    SpreadBootstrapBattery,
    bootstrap_q5_q1_spread,
    run_bucket_permutation_tests,
)
from v3.validation.topn_portfolio import TopNPortfolioMetrics, compute_topn_portfolio_metrics
from v3.validation.wfo_config import PERMUTATION_CONFIG, SIGNIFICANCE_ALPHA

HORIZONS: tuple[int, ...] = (5, 10, 15, 20)
PRIMARY_HORIZON = 5
FDR_SWEEP_CONFIG = PermutationConfig(n_permutations=300, seed=353, forward_window=5)


@dataclass(frozen=True)
class LightResult:
    definition: str
    horizon: int
    ranking: RankingResult
    pearson_ic_mean: float | None
    topn: dict[int, TopNPortfolioMetrics]


@dataclass(frozen=True)
class PrimaryResult:
    definition: str
    light: LightResult
    regime: RegimeRobustness
    day_concentration: DayConcentrationRobustness
    year: YearRobustness
    stock_concentration: StockConcentrationRobustness
    sector_concentration: SectorConcentrationResult
    matched_control: MatchedControlResult
    spread_bootstrap: SpreadBootstrapBattery
    permutation_q5_p: float | None
    cost_sensitivity: dict[str, CostTierResult]
    economic_significance: dict[int, EconomicSignificance]


@dataclass(frozen=True)
class MarketNeutralizationRow:
    definition: str
    q5_q1_spread: float | None
    rank_ic: float | None
    top5_expectancy: float | None
    top5_pf: float | None
    day_cluster_ci_low: float | None
    block_ci_low: float | None
    permutation_p: float | None
    fdr_significant: bool | None
    survives: bool  # spread>0 AND both bootstrap CIs>0 AND permutation+FDR significant


@dataclass(frozen=True)
class V3_5Report:
    light_results: dict[tuple[str, int], LightResult]
    primary_results: dict[str, PrimaryResult]
    market_neutralization_table: list[MarketNeutralizationRow]
    residual_strength_by_horizon: dict[tuple[str, int], float | None]
    fdr_results: dict[str, FDRResult]
    edge_classification: EdgeClassificationResult


def _evaluate_light(
    definition: str, horizon: int, predictions: pd.DataFrame,
) -> LightResult:
    ranking = evaluate_ranking(predictions, horizon)
    pearson = summarize_pearson_ic(predictions, horizon)
    topn = {
        n: compute_topn_portfolio_metrics(predictions, n, "actual", horizon) for n in (5, 10, 20)
    }
    return LightResult(
        definition=definition, horizon=horizon, ranking=ranking,
        pearson_ic_mean=pearson.mean_ic, topn=topn,
    )


def run_v3_5_analysis(
    augmented_dataset: pd.DataFrame,
    predictions_by_combo: dict[tuple[str, int], pd.DataFrame],
    tickers: list[str],
    topix_ohlcv: pd.DataFrame,
    market_regime_config,
    cost_tiers: list[TransactionCostTierConfig],
    v3_config,
) -> V3_5Report:
    price_volume_panel = build_price_volume_panel(tickers, v3_config)
    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_cols = ["ticker", "sector33", "scale"]
    sector_map = (
        attach_sector_and_scale(ticker_frame)[sector_cols].drop_duplicates(subset=["ticker"])
    )

    # Both depend only on augmented_dataset/topix_ohlcv, not on `definition`
    # - built ONCE and reused across all 4 Primary combinations below.
    full_day_panel = build_full_day_panel(augmented_dataset, price_volume_panel, sector_map)
    regime_df = compute_market_regime(topix_ohlcv, market_regime_config)

    light_results: dict[tuple[str, int], LightResult] = {}
    for definition in TARGET_DEFINITIONS:
        for horizon in HORIZONS:
            predictions = predictions_by_combo[(definition, horizon)]
            light_results[(definition, horizon)] = _evaluate_light(definition, horizon, predictions)

    primary_results: dict[str, PrimaryResult] = {}
    for definition in TARGET_DEFINITIONS:
        light = light_results[(definition, PRIMARY_HORIZON)]
        predictions = predictions_by_combo[(definition, PRIMARY_HORIZON)]
        predictions_with_sector = predictions.merge(
            sector_map[["ticker", "sector33"]], on="ticker", how="left"
        )
        sector_concentration = run_sector_concentration(predictions_with_sector, PRIMARY_HORIZON)

        regime = run_regime_robustness(predictions, regime_df, PRIMARY_HORIZON)
        day_concentration = run_day_concentration_robustness(predictions, PRIMARY_HORIZON)
        year = run_year_robustness(predictions, PRIMARY_HORIZON)
        stock_concentration = run_stock_concentration_robustness(predictions, PRIMARY_HORIZON)

        valid = predictions.dropna(subset=["prediction", "actual"]).copy()
        valid["_bucket"] = assign_quantile_buckets(valid["prediction"])
        q5 = valid[valid["_bucket"] == "Q5"]
        matched_control = run_matched_control_analysis(q5, full_day_panel)

        spread_bootstrap = bootstrap_q5_q1_spread(predictions)
        permutation_results = run_bucket_permutation_tests(predictions, config=PERMUTATION_CONFIG)
        q5_p = next((r.result.p_value for r in permutation_results if r.bucket_label == "Q5"), None)

        cost_result = run_cost_sensitivity(predictions, cost_tiers, PRIMARY_HORIZON)
        econ_sig = {n: compute_economic_significance(m) for n, m in light.topn.items()}

        primary_results[definition] = PrimaryResult(
            definition=definition, light=light, regime=regime, day_concentration=day_concentration,
            year=year, stock_concentration=stock_concentration,
            sector_concentration=sector_concentration, matched_control=matched_control,
            spread_bootstrap=spread_bootstrap, permutation_q5_p=q5_p, cost_sensitivity=cost_result,
            economic_significance=econ_sig,
        )

    # Section 27: Permutation + FDR on B/C/D's Q5 across all 4 Horizons
    # (12 NEW tests, explicitly recorded per spec's own instruction -
    # distinct from V3-3's 16-test and V3-4's 6-test families).
    p_values: dict[str, float] = {}
    for definition in TARGET_DEFINITIONS:
        if definition == TARGET_A_RAW:
            continue
        for horizon in HORIZONS:
            predictions = predictions_by_combo[(definition, horizon)]
            results = run_bucket_permutation_tests(predictions, config=FDR_SWEEP_CONFIG)
            q5_result = next((r for r in results if r.bucket_label == "Q5"), None)
            if q5_result is not None:
                p_values[f"{definition}:{horizon}d:Q5"] = q5_result.result.p_value
    fdr_results = benjamini_hochberg_correction(p_values) if p_values else {}

    market_neutralization_table = _build_market_neutralization_table(
        primary_results, fdr_results
    )
    residual_strength_by_horizon = _compute_residual_strength_by_horizon(light_results)

    edge_inputs = _build_edge_classification_inputs(
        primary_results, market_neutralization_table, fdr_results
    )
    edge_classification = classify_edge_source(edge_inputs)

    return V3_5Report(
        light_results=light_results, primary_results=primary_results,
        market_neutralization_table=market_neutralization_table,
        residual_strength_by_horizon=residual_strength_by_horizon, fdr_results=fdr_results,
        edge_classification=edge_classification,
    )


def _build_market_neutralization_table(
    primary_results: dict[str, PrimaryResult], fdr_results: dict[str, FDRResult],
) -> list[MarketNeutralizationRow]:
    rows = []
    for definition in TARGET_DEFINITIONS:
        p = primary_results[definition]
        spread = p.light.ranking.q5_q1_spread
        rank_ic = p.light.ranking.ic_summary.mean_ic
        top5_stats = p.light.topn[5].base.stats
        day_cluster_low = p.spread_bootstrap.day_cluster.ci_low
        block_low = p.spread_bootstrap.block.ci_low
        fdr_key = f"{definition}:5d:Q5"
        # Target A (Raw) is never included in this Phase's own FDR sweep
        # (spec section 27 - it's already covered by V3-3's own 16-test
        # family) - its fdr_significant is always None here, by design.
        fdr_significant = fdr_results[fdr_key].significant if fdr_key in fdr_results else None
        survives = (
            spread is not None and spread > 0
            and day_cluster_low is not None and day_cluster_low > 0
            and block_low is not None and block_low > 0
            and p.permutation_q5_p is not None and p.permutation_q5_p < SIGNIFICANCE_ALPHA
            and bool(fdr_significant)
        )
        rows.append(
            MarketNeutralizationRow(
                definition=definition, q5_q1_spread=spread, rank_ic=rank_ic,
                top5_expectancy=top5_stats.mean_return, top5_pf=top5_stats.profit_factor,
                day_cluster_ci_low=day_cluster_low, block_ci_low=block_low,
                permutation_p=p.permutation_q5_p, fdr_significant=fdr_significant,
                survives=survives,
            )
        )
    return rows


def _compute_residual_strength_by_horizon(
    light_results: dict[tuple[str, int], LightResult],
) -> dict[tuple[str, int], float | None]:
    out: dict[tuple[str, int], float | None] = {}
    for horizon in HORIZONS:
        original = light_results[(TARGET_A_RAW, horizon)].ranking.q5_q1_spread
        for definition in TARGET_DEFINITIONS:
            if definition == TARGET_A_RAW:
                continue
            residual = light_results[(definition, horizon)].ranking.q5_q1_spread
            out[(definition, horizon)] = residual_strength(residual, original)
    return out


def _build_edge_classification_inputs(
    primary_results: dict[str, PrimaryResult], table: list[MarketNeutralizationRow],
    fdr_results: dict[str, FDRResult],
) -> EdgeClassificationInputs:
    by_definition = {row.definition: row for row in table}
    c = primary_results["beta_residual"]
    c_row = by_definition["beta_residual"]
    bear_excl_spread = c.regime.leave_one_out["excl_BEAR"].ranking.q5_q1_spread
    top5 = c.economic_significance[5].expected_return_per_trade
    top10 = c.economic_significance[10].expected_return_per_trade
    top20 = c.economic_significance[20].expected_return_per_trade
    fdr_key = "beta_residual:5d:Q5"
    fdr_significant = fdr_results[fdr_key].significant if fdr_key in fdr_results else False

    return EdgeClassificationInputs(
        raw_q5_q1=by_definition[TARGET_A_RAW].q5_q1_spread,
        topix_relative_q5_q1=by_definition["topix_relative"].q5_q1_spread,
        beta_residual_q5_q1=c_row.q5_q1_spread,
        beta_residual_bear_excluded_q5_q1=bear_excl_spread,
        beta_residual_top5_expectancy=top5,
        beta_residual_top10_expectancy=top10,
        beta_residual_top20_expectancy=top20,
        beta_residual_day_cluster_ci_low=c.spread_bootstrap.day_cluster.ci_low,
        beta_residual_block_ci_low=c.spread_bootstrap.block.ci_low,
        beta_residual_permutation_p=c.permutation_q5_p,
        beta_residual_fdr_significant=fdr_significant,
    )
