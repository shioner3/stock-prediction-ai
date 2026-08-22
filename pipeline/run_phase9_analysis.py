"""Phase 9: robustness/sensitivity checks for long_oversold_rebound -
actively searching for conditions under which its Phase 8 edge breaks
down, rather than trying to further confirm it. Every Signal/Score/
Backtest/WFO/Cost/Decision component remains completely unchanged; this
module only re-slices and re-samples the SAME already-computed trades in
additional ways: Day Cluster / Block Bootstrap, Leave-One-BEAR-Episode-Out
and Leave-One-Year-Out (with bootstrap CIs), day-level event
concentration, a fixed timing-offset placebo sweep, cross-sectional
sector/liquidity breakdowns, cost stress by scope, forward-holding-period
sensitivity, and 6 pre-fixed stress scenarios (A-F).

Deliberately does NOT call pipeline.run_walk_forward.run_walk_forward()
again - Phase 8's saved report already has the Combined-OOS Decision/
permutation/FDR result, and recomputing it is a ~1-hour operation at this
data scale (dominated by the 12-signal Combined-period Permutation Test),
unnecessary for Phase 9's purpose. Phase 9 instead calls
pipeline.run_backtest.run_backtest() directly (fast, ~45s) for the raw
trades it actually needs, and reuses Phase 8's config_hash check
unchanged (pipeline.run_phase8_analysis.verify_config_hash).

Dependency direction (spec section 19): this module IMPORTS FROM
signals/backtest/targets, and is never imported BY them - see
tests/test_phase9_no_lookahead.py's AST-based check, mirroring
tests/test_target_leakage.py's existing pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import pandas as pd

from backtest.block_bootstrap import BlockBootstrapResult, block_bootstrap
from backtest.bootstrap import BootstrapResult, bootstrap_ci
from backtest.costs import apply_cost
from backtest.day_cluster_bootstrap import DayClusterBootstrapResult, day_cluster_bootstrap
from backtest.engine import run_backtest_for_ticker
from backtest.episode_analysis import (
    RegimeEpisode,
    RichEpisodeMetrics,
    compute_rich_episode_metrics,
    identify_regime_episodes,
)
from backtest.event_concentration import DayConcentrationMetrics, compute_day_concentration
from backtest.market_regime import compute_market_regime
from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.timing_shift import shift_signal_records
from config.loader import AppConfig, Phase9Config
from pipeline.run_backtest import run_backtest
from pipeline.run_phase8_analysis import ConfigCheckResult, ConfigMismatchError, verify_config_hash
from storage.parquet_store import load_feature_panel, load_ohlcv, load_signal_records
from targets.forward_returns import FORWARD_WINDOWS, compute_forward_returns
from universe.jpx_master import parse_jpx_master

logger = logging.getLogger(__name__)

BEAR_REGIME = "BEAR"
AUG_2024_EVENT_START = date_type(2024, 8, 2)
AUG_2024_EVENT_END = date_type(2024, 8, 15)
YEARS = (2022, 2023, 2024, 2025, 2026)


# --- Leave-One-Period-Out with bootstrap CI (year and episode share this) ----


@dataclass
class LOPOWithCI:
    period_label: str
    n_trades_removed: int
    full_sample_metrics: BacktestMetrics
    remaining_metrics: BacktestMetrics
    remaining_bootstrap_expectancy: BootstrapResult


def lopo_with_bootstrap(
    trades_with_group: pd.DataFrame, group_col: str, bootstrap_config
) -> list[LOPOWithCI]:
    valid = trades_with_group.dropna(subset=[group_col])
    if valid.empty:
        return []
    full_metrics = compute_metrics(valid)
    labels = sorted(valid[group_col].unique(), key=str)

    results = []
    for label in labels:
        mask = valid[group_col] == label
        remaining = valid[~mask]
        remaining_metrics = compute_metrics(remaining)
        ci = bootstrap_ci(remaining["return"].to_numpy(dtype=float), "expectancy", bootstrap_config)
        results.append(
            LOPOWithCI(
                period_label=str(label),
                n_trades_removed=int(mask.sum()),
                full_sample_metrics=full_metrics,
                remaining_metrics=remaining_metrics,
                remaining_bootstrap_expectancy=ci,
            )
        )
    return results


# --- Timing Placebo sweep -----------------------------------------------------


@dataclass
class TimingOffsetResult:
    offset_days: int
    n_trades_total: int
    n_trades_bear: int
    metrics_bear: BacktestMetrics


def build_signal_ticker_cache(
    config: AppConfig, tickers: list[str], direction: str, signal_name: str
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Loads each triggered ticker's (signal_records, feature panel) ONCE,
    reused across every timing offset - the per-ticker Parquet I/O is the
    dominant cost of the sweep, not the (cheap) backtest re-run itself.
    """
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for ticker in tickers:
        signal_records = load_signal_records(ticker, signals_dir)
        sig = signal_records[
            (signal_records["direction"] == direction)
            & (signal_records["signal_name"] == signal_name)
        ]
        if sig.empty:
            continue
        panel = load_feature_panel(ticker, features_dir)
        cache[ticker] = (sig, panel)
    return cache


def _run_timing_offset_sweep(
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    offsets: list[int],
    regime_df: pd.DataFrame,
    backtest_config,
) -> list[TimingOffsetResult]:
    results = []
    for offset in offsets:
        trade_frames = []
        for sig, panel in cache.values():
            shifted = shift_signal_records(sig, panel, offset)
            if shifted.empty:
                continue
            result = run_backtest_for_ticker(shifted, panel, backtest_config)
            if not result.trades.empty:
                trade_frames.append(result.trades)

        combined = (
            pd.concat(trade_frames, ignore_index=True)
            if trade_frames
            else pd.DataFrame(columns=["ticker", "signal_date", "return"])
        )
        merged = combined.merge(regime_df, left_on="signal_date", right_on="date", how="left")
        bear = merged[merged["regime"] == BEAR_REGIME]

        results.append(
            TimingOffsetResult(
                offset_days=offset,
                n_trades_total=len(combined),
                n_trades_bear=len(bear),
                metrics_bear=compute_metrics(bear),
            )
        )
    return results


# --- Cross-sectional: sector / liquidity ---------------------------------------


def _load_sector_map(jpx_master_path: Path) -> dict[str, tuple[str | None, str | None]]:
    if not jpx_master_path.exists():
        return {}
    master = parse_jpx_master(jpx_master_path)
    return {
        str(row.code): (row.sector33, row.sector17) for row in master.itertuples(index=False)
    }


def _sector_breakdown(
    trades: pd.DataFrame, sector_map: dict[str, tuple[str | None, str | None]]
) -> dict[str, BacktestMetrics]:
    if not sector_map:
        return {}
    sector33 = trades["ticker"].map(lambda t: sector_map.get(t, (None, None))[0])
    tagged = trades.assign(sector33=sector33).dropna(subset=["sector33"])
    if tagged.empty:
        return {}
    return {str(s): compute_metrics(g) for s, g in tagged.groupby("sector33")}


def _ticker_liquidity_proxy(
    tickers: list[str], raw_dir: Path, lookback_days: int = 20
) -> dict[str, float]:
    """Mean volume over each ticker's EARLIEST `lookback_days` rows - the
    same basis universe/filters.py::check_price_and_liquidity() already
    uses for the Universe's own liquidity decision (Phase 6.5's leakage
    fix), reused here unchanged rather than inventing a new liquidity
    definition.
    """
    result = {}
    for ticker in tickers:
        try:
            df = load_ohlcv(ticker, raw_dir)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        earliest = df.sort_values("date").head(lookback_days)
        result[ticker] = float(earliest["volume"].mean())
    return result


def _liquidity_bucket_breakdown(
    trades: pd.DataFrame, liquidity_map: dict[str, float], n_buckets: int = 4
) -> dict[str, BacktestMetrics]:
    liquidity = trades["ticker"].map(liquidity_map)
    tagged = trades.assign(liquidity=liquidity).dropna(subset=["liquidity"])
    if tagged.empty:
        return {}
    labels = [f"Q{i + 1}" for i in range(n_buckets)]
    try:
        tagged = tagged.assign(
            bucket=pd.qcut(tagged["liquidity"], n_buckets, labels=labels, duplicates="drop")
        )
    except ValueError:
        return {}
    return {str(b): compute_metrics(g) for b, g in tagged.groupby("bucket", observed=True)}


# --- Cost stress by scope -------------------------------------------------------


def _cost_stress(trades: pd.DataFrame, tiers) -> dict[str, BacktestMetrics]:
    return {
        tier.name: compute_metrics(trades.assign(**{"return": apply_cost(trades["return"], tier)}))
        for tier in tiers
    }


# --- Forward Holding Period sensitivity -----------------------------------------


@dataclass
class ForwardHorizonResult:
    horizon_days: int
    n: int
    mean_return: float | None
    median_return: float | None
    win_rate: float | None
    bootstrap: BootstrapResult


def _forward_horizon_profile(
    config: AppConfig, tickers: list[str], direction: str, signal_name: str, bootstrap_config
) -> list[ForwardHorizonResult]:
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)
    values_by_horizon: dict[int, list[float]] = {h: [] for h in FORWARD_WINDOWS}

    for ticker in tickers:
        signal_records = load_signal_records(ticker, signals_dir)
        sig = signal_records[
            (signal_records["direction"] == direction)
            & (signal_records["signal_name"] == signal_name)
        ]
        if sig.empty:
            continue
        panel = load_feature_panel(ticker, features_dir)
        forward = compute_forward_returns(panel)
        forward_with_date = panel[["date"]].join(forward)
        merged = sig.merge(forward_with_date, on="date", how="left")
        for h in FORWARD_WINDOWS:
            values_by_horizon[h].extend(merged[f"forward_return_{h}d"].dropna().tolist())

    results = []
    for h in FORWARD_WINDOWS:
        arr = pd.Series(values_by_horizon[h], dtype=float)
        ci = bootstrap_ci(arr.to_numpy(), "mean_return", bootstrap_config)
        results.append(
            ForwardHorizonResult(
                horizon_days=h,
                n=len(arr),
                mean_return=float(arr.mean()) if len(arr) else None,
                median_return=float(arr.median()) if len(arr) else None,
                win_rate=float((arr > 0).mean()) if len(arr) else None,
                bootstrap=ci,
            )
        )
    return results


# --- Predefined stress scenarios A-F --------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    description: str
    metrics: BacktestMetrics


def _winsorize(trades: pd.DataFrame, lower_q: float, upper_q: float) -> pd.DataFrame:
    lower = trades["return"].quantile(lower_q)
    upper = trades["return"].quantile(upper_q)
    return trades.assign(**{"return": trades["return"].clip(lower=lower, upper=upper)})


def _exclude_date_range(trades: pd.DataFrame, start: date_type, end: date_type) -> pd.DataFrame:
    mask = (trades["signal_date"] >= start) & (trades["signal_date"] <= end)
    return trades[~mask]


def _exclude_year(trades: pd.DataFrame, year: int) -> pd.DataFrame:
    return trades[trades["signal_date"].apply(lambda d: d.year) != year]


def _run_scenarios(
    signal_trades: pd.DataFrame, phase9_config: Phase9Config
) -> list[ScenarioResult]:
    lower_q = phase9_config.winsorization.lower_percentile
    upper_q = phase9_config.winsorization.upper_percentile

    scenario_a = signal_trades
    scenario_b = _exclude_date_range(signal_trades, AUG_2024_EVENT_START, AUG_2024_EVENT_END)
    scenario_c = _exclude_year(signal_trades, 2024)
    scenario_e = _winsorize(signal_trades, lower_q, upper_q)
    scenario_f = _winsorize(scenario_b, lower_q, upper_q)

    return [
        ScenarioResult("A", "Full Combined OOS", compute_metrics(scenario_a)),
        ScenarioResult(
            "B",
            f"exclude {AUG_2024_EVENT_START}..{AUG_2024_EVENT_END}",
            compute_metrics(scenario_b),
        ),
        ScenarioResult("C", "exclude year 2024 entirely", compute_metrics(scenario_c)),
        # Scenario D (each BEAR episode leave-one-out) is reported via
        # Phase9Report.lopo_by_episode - not duplicated here.
        ScenarioResult(
            "E", f"winsorize returns at [{lower_q}, {upper_q}] percentile",
            compute_metrics(scenario_e),
        ),
        ScenarioResult("F", "Scenario E + Scenario B combined", compute_metrics(scenario_f)),
    ]


# --- Top-level report ------------------------------------------------------------


@dataclass
class Phase9Report:
    config_check: ConfigCheckResult
    tickers: list[str]
    n_signal_trades_total: int
    bear_episodes: list[RegimeEpisode]
    rich_episode_metrics: list[RichEpisodeMetrics]
    lopo_by_episode: list[LOPOWithCI]
    lopo_by_year: list[LOPOWithCI]
    day_concentration_bear: DayConcentrationMetrics
    day_cluster_bootstrap_bear: dict[str, DayClusterBootstrapResult]
    day_cluster_bootstrap_combined: dict[str, DayClusterBootstrapResult]
    block_bootstrap_bear: dict[str, BlockBootstrapResult]
    block_bootstrap_combined: dict[str, BlockBootstrapResult]
    timing_offset_sweep: list[TimingOffsetResult]
    sector_breakdown_bear: dict[str, BacktestMetrics]
    liquidity_breakdown_bear: dict[str, BacktestMetrics]
    cost_stress_combined: dict[str, BacktestMetrics]
    cost_stress_bear: dict[str, BacktestMetrics]
    cost_stress_aug2024_episode: dict[str, BacktestMetrics]
    cost_stress_bear_excl_aug2024: dict[str, BacktestMetrics]
    forward_horizon_profile: list[ForwardHorizonResult]
    scenarios: list[ScenarioResult]


def run_phase9_analysis(
    config: AppConfig,
    phase9_config: Phase9Config,
    tickers: list[str],
    phase6_5_report_path: Path,
    phase7_report_path: Path,
    jpx_master_path: Path,
    target_direction: str = "LONG",
    target_signal_name: str = "long_oversold_rebound",
) -> Phase9Report:
    config_check = verify_config_hash(phase6_5_report_path, phase7_report_path)
    if not config_check.matches:
        raise ConfigMismatchError(f"CONFIG_MISMATCH: {config_check}")

    logger.info("Phase 9: running Combined backtest over %d tickers", len(tickers))
    backtest_summary = run_backtest(config, tickers=tickers)
    trades = backtest_summary.trades
    signal_trades = trades[
        (trades["direction"] == target_direction) & (trades["signal_name"] == target_signal_name)
    ]
    if signal_trades.empty:
        raise RuntimeError(f"target signal {target_direction}:{target_signal_name} never triggered")

    topix = load_ohlcv("TOPIX", config.data.processed_dir)
    regime_df = compute_market_regime(topix, config.validation.market_regime)

    merged = signal_trades.merge(regime_df, left_on="signal_date", right_on="date", how="left")
    bear_trades = merged[merged["regime"] == BEAR_REGIME].copy()

    bear_episodes = identify_regime_episodes(regime_df, BEAR_REGIME)
    rich_episode_metrics = compute_rich_episode_metrics(signal_trades, bear_episodes)

    bear_trades_with_episode = bear_trades.assign(
        episode_index=_assign_episode_index(bear_trades["signal_date"], bear_episodes)
    )
    lopo_by_episode = lopo_with_bootstrap(
        bear_trades_with_episode, "episode_index", config.validation.bootstrap
    )

    signal_trades_with_year = signal_trades.assign(
        year=signal_trades["signal_date"].apply(lambda d: d.year)
    )
    lopo_by_year = lopo_with_bootstrap(
        signal_trades_with_year, "year", config.validation.bootstrap
    )

    day_concentration_bear = compute_day_concentration(bear_trades)

    day_cluster_bear = {
        m: day_cluster_bootstrap(bear_trades, m, phase9_config.day_cluster_bootstrap)
        for m in ("mean_return", "expectancy", "profit_factor")
    }
    day_cluster_combined = {
        m: day_cluster_bootstrap(signal_trades, m, phase9_config.day_cluster_bootstrap)
        for m in ("mean_return", "expectancy", "profit_factor")
    }
    block_bear = {
        m: block_bootstrap(bear_trades, m, phase9_config.block_bootstrap)
        for m in ("mean_return", "expectancy", "profit_factor")
    }
    block_combined = {
        m: block_bootstrap(signal_trades, m, phase9_config.block_bootstrap)
        for m in ("mean_return", "expectancy", "profit_factor")
    }

    logger.info(
        "Phase 9: timing placebo sweep over %d offsets, %d candidate tickers",
        len(phase9_config.timing_placebo.offsets), len(tickers),
    )
    signal_cache = build_signal_ticker_cache(config, tickers, target_direction, target_signal_name)
    timing_sweep = _run_timing_offset_sweep(
        signal_cache, phase9_config.timing_placebo.offsets, regime_df, config.backtest
    )

    sector_map = _load_sector_map(jpx_master_path)
    sector_breakdown = _sector_breakdown(bear_trades, sector_map)

    liquidity_map = _ticker_liquidity_proxy(tickers, Path(config.data.raw_dir))
    liquidity_breakdown = _liquidity_bucket_breakdown(bear_trades, liquidity_map)

    tiers = config.validation.transaction_cost.tiers
    aug2024_episode_trades = signal_trades[
        (signal_trades["signal_date"] >= AUG_2024_EVENT_START)
        & (signal_trades["signal_date"] <= AUG_2024_EVENT_END)
    ]
    bear_excl_aug2024 = _exclude_date_range(bear_trades, AUG_2024_EVENT_START, AUG_2024_EVENT_END)

    cost_stress_combined = _cost_stress(signal_trades, tiers)
    cost_stress_bear = _cost_stress(bear_trades, tiers)
    cost_stress_aug2024 = _cost_stress(aug2024_episode_trades, tiers)
    cost_stress_bear_excl_aug2024 = _cost_stress(bear_excl_aug2024, tiers)

    logger.info("Phase 9: forward holding period profile over %d tickers", len(tickers))
    forward_horizon_profile = _forward_horizon_profile(
        config, tickers, target_direction, target_signal_name, config.validation.bootstrap
    )

    scenarios = _run_scenarios(signal_trades, phase9_config)

    return Phase9Report(
        config_check=config_check,
        tickers=tickers,
        n_signal_trades_total=len(signal_trades),
        bear_episodes=bear_episodes,
        rich_episode_metrics=rich_episode_metrics,
        lopo_by_episode=lopo_by_episode,
        lopo_by_year=lopo_by_year,
        day_concentration_bear=day_concentration_bear,
        day_cluster_bootstrap_bear=day_cluster_bear,
        day_cluster_bootstrap_combined=day_cluster_combined,
        block_bootstrap_bear=block_bear,
        block_bootstrap_combined=block_combined,
        timing_offset_sweep=timing_sweep,
        sector_breakdown_bear=sector_breakdown,
        liquidity_breakdown_bear=liquidity_breakdown,
        cost_stress_combined=cost_stress_combined,
        cost_stress_bear=cost_stress_bear,
        cost_stress_aug2024_episode=cost_stress_aug2024,
        cost_stress_bear_excl_aug2024=cost_stress_bear_excl_aug2024,
        forward_horizon_profile=forward_horizon_profile,
        scenarios=scenarios,
    )


def _assign_episode_index(dates: pd.Series, episodes: list[RegimeEpisode]) -> pd.Series:
    result: pd.Series = pd.Series([None] * len(dates), index=dates.index, dtype=object)
    for ep in episodes:
        mask = (dates >= ep.start_date) & (dates <= ep.end_date)
        result = result.mask(mask, ep.index)
    return result
