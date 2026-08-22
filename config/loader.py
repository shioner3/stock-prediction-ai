"""Typed access to config/settings.yaml and config/universe_filters.yaml.

All tunable parameters (retry behaviour, universe filters, storage paths,
and - in later phases - score weights and backtest windows) live in these
YAML files, never hardcoded in application code.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel


class FetchConfig(BaseModel):
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    timeout_seconds: float = 15.0


class MarketIndexConfig(BaseModel):
    provider: str = "yfinance"  # doubles as MarketIndexMeta.source
    ticker: str = "1306.T"
    # name/type feed providers.base.MarketIndexMeta so every consumer
    # (logs, FeatureMeta descriptions) can say "TOPIX Proxy (1306.T)"
    # instead of "TOPIX" - see providers/market_index.py.
    name: str = "TOPIX Proxy"
    type: str = "ETF_PROXY"


class DataConfig(BaseModel):
    provider: str = "yfinance"
    start_date: date
    end_date: date | None = None
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    features_dir: Path = Path("data/features")
    signals_dir: Path = Path("data/signals")
    scores_dir: Path = Path("data/scores")
    fetch: FetchConfig = FetchConfig()
    market_index: MarketIndexConfig = MarketIndexConfig()


class UniverseConfig(BaseModel):
    master_list_path: Path
    segments: list[str] = ["Prime", "Standard", "Growth"]
    exclude_etf: bool = True
    exclude_reit: bool = True


class PullbackFeatureConfig(BaseModel):
    # Lookback window for the Pullback category's "recent high" reference
    # (features/pullback.py's distance_from_recent_high / pullback_depth).
    recent_high_window: int = 20


class FeatureConfig(BaseModel):
    pullback: PullbackFeatureConfig = PullbackFeatureConfig()


# --- Signal thresholds (Phase 4) --------------------------------------------
#
# Every value below is an initial hypothesis, not a tuned/optimized
# parameter - Phase 4 explicitly forbids grid search, Bayesian
# optimization, or adjusting these after looking at backtest results.
# See research/signal_notes/*.md for the hypothesis each threshold serves.


class LongBreakoutSignalConfig(BaseModel):
    lookback: int = 20  # must be one of features/breakout.py's HIGH_LOW_WINDOWS
    volume_multiple: float = 1.5  # triggers on volume_ratio_20d > this


class LongPullbackSignalConfig(BaseModel):
    sma_fast: int = 20
    sma_slow: int = 50
    min_depth: float = 0.03  # pullback_depth must be at least this deep...
    max_depth: float = 0.15  # ...and no deeper than this (avoid a falling knife)


class LongMaReboundSignalConfig(BaseModel):
    sma_fast: int = 20
    sma_slow: int = 50


class LongMomentumContinuationSignalConfig(BaseModel):
    return_5d_min: float = 0.03
    return_20d_min: float = 0.0
    sma_period: int = 20  # Close must be above this SMA


class LongVolumeBreakoutSignalConfig(BaseModel):
    return_1d_min: float = 0.03
    volume_ratio_min: float = 2.0


class LongOversoldReboundSignalConfig(BaseModel):
    rsi_period: int = 14  # must be one of features/indicators.py's RSI_PERIODS
    rsi_max: float = 30.0


class LongSignalsConfig(BaseModel):
    breakout: LongBreakoutSignalConfig = LongBreakoutSignalConfig()
    pullback: LongPullbackSignalConfig = LongPullbackSignalConfig()
    ma_rebound: LongMaReboundSignalConfig = LongMaReboundSignalConfig()
    momentum_continuation: LongMomentumContinuationSignalConfig = (
        LongMomentumContinuationSignalConfig()
    )
    volume_breakout: LongVolumeBreakoutSignalConfig = LongVolumeBreakoutSignalConfig()
    oversold_rebound: LongOversoldReboundSignalConfig = LongOversoldReboundSignalConfig()


class ShortBreakdownSignalConfig(BaseModel):
    lookback: int = 20
    volume_multiple: float = 1.5


class ShortPullbackSignalConfig(BaseModel):
    sma_fast: int = 20
    sma_slow: int = 50
    min_depth: float = 0.03  # bounce_depth must be at least this deep...
    max_depth: float = 0.15  # ...and no deeper than this


class ShortMaRejectionSignalConfig(BaseModel):
    sma_fast: int = 20
    sma_slow: int = 50


class ShortMomentumContinuationSignalConfig(BaseModel):
    return_5d_max: float = -0.03
    return_20d_max: float = 0.0
    sma_period: int = 20  # Close must be below this SMA


class ShortVolumeBreakdownSignalConfig(BaseModel):
    return_1d_max: float = -0.03
    volume_ratio_min: float = 2.0


class ShortOverboughtReversalSignalConfig(BaseModel):
    rsi_period: int = 14
    rsi_min: float = 70.0


class ShortSignalsConfig(BaseModel):
    breakdown: ShortBreakdownSignalConfig = ShortBreakdownSignalConfig()
    pullback: ShortPullbackSignalConfig = ShortPullbackSignalConfig()
    ma_rejection: ShortMaRejectionSignalConfig = ShortMaRejectionSignalConfig()
    momentum_continuation: ShortMomentumContinuationSignalConfig = (
        ShortMomentumContinuationSignalConfig()
    )
    volume_breakdown: ShortVolumeBreakdownSignalConfig = ShortVolumeBreakdownSignalConfig()
    overbought_reversal: ShortOverboughtReversalSignalConfig = (
        ShortOverboughtReversalSignalConfig()
    )


class SignalsConfig(BaseModel):
    long: LongSignalsConfig = LongSignalsConfig()
    short: ShortSignalsConfig = ShortSignalsConfig()


class BacktestConfig(BaseModel):
    hold_days: int = 5
    # Section 14: if a same-(ticker, signal_name, direction) trade is
    # still open (within hold_days of an earlier trigger), skip opening
    # a second one on top of it.
    suppress_overlapping_signals: bool = True


# --- Score thresholds (Phase 5) ---------------------------------------------
#
# Every value below is an initial hypothesis, not a tuned/optimized
# parameter - Phase 5 explicitly forbids grid search or adjusting these
# after looking at bucket-analysis results. See README's "Score Mapping"
# for the full point allocation this config drives.


class ScoreWeightsConfig(BaseModel):
    """Max points per category - these ARE 20/20/15/15/20/10 (they sum to
    100), not multipliers applied to some other 0-1 subscore.
    """

    trend: int = 20
    momentum: int = 20
    volume: int = 15
    relative: int = 15
    setup: int = 20
    risk: int = 10


class ScoreTrendConfig(BaseModel):
    sma_fast: int = 20
    sma_slow: int = 50
    # A slope must exceed this (in absolute value, signed by direction)
    # to count as "trending" - guards against float noise at exactly 0.
    slope_epsilon: float = 0.0


class ScoreMomentumConfig(BaseModel):
    return_5d_period: int = 5  # must be one of features/momentum.py's RETURN_WINDOWS
    return_10d_period: int = 10
    rsi_period: int = 14  # must be one of features/indicators.py's RSI_PERIODS
    rsi_midline: float = 50.0


class ScoreVolumeConfig(BaseModel):
    ratio_confirm_threshold: float = 1.2


class ScoreRelativeConfig(BaseModel):
    # 5 strictly-descending breakpoints -> 5/4/3/2/1 points; below all of
    # them -> 0. Applied directly for LONG (higher rs = better) and to
    # -rs for SHORT (lower/more negative rs = better) - see
    # scoring/scorer.py::_map_rs_to_points.
    thresholds: list[float] = [0.05, 0.02, 0.0, -0.02, -0.05]


class ScoreSetupConfig(BaseModel):
    # Deliberately stricter than any single Signal's own trigger
    # threshold - Setup measures "how far past the bar", not "did it
    # clear the bar" (which the Signal itself already established).
    return_5d_strong: float = 0.05
    volume_ratio_strong: float = 2.0
    ma_stack_fast: int = 20
    ma_stack_mid: int = 50
    ma_stack_slow: int = 75
    atr_move_ratio_threshold: float = 1.0


class ScoreRiskConfig(BaseModel):
    # Risk is direction-agnostic: instability is bad whether LONG or SHORT.
    atr_pct_max: float = 0.05
    volatility_max: float = 0.03


class ScoringConfig(BaseModel):
    weights: ScoreWeightsConfig = ScoreWeightsConfig()
    trend: ScoreTrendConfig = ScoreTrendConfig()
    momentum: ScoreMomentumConfig = ScoreMomentumConfig()
    volume: ScoreVolumeConfig = ScoreVolumeConfig()
    relative: ScoreRelativeConfig = ScoreRelativeConfig()
    setup: ScoreSetupConfig = ScoreSetupConfig()
    risk: ScoreRiskConfig = ScoreRiskConfig()


# --- Walk Forward / Statistical Validation (Phase 6) ------------------------
#
# Every value below is fixed BEFORE looking at any OOS result (see
# README's "OOS期間の固定" section for the actual dates this produces
# against the data on disk as of Phase 6). None of these may be changed
# after seeing OOS/bootstrap/permutation output - that would invalidate
# the "OOS" label entirely (Phase 6 spec, "最重要ルール").


class WalkForwardConfig(BaseModel):
    train_months: int = 12
    validation_months: int = 3
    oos_months: int = 3
    step_months: int = 3
    # A trailing window shorter than this fraction of oos_months (data
    # ran out mid-window) is dropped rather than analyzed as a partial,
    # non-comparable OOS period.
    min_oos_completeness: float = 0.5


class TransactionCostTierConfig(BaseModel):
    name: str
    round_trip_bps: float  # basis points of round-trip cost, deducted from each trade's return


class TransactionCostConfig(BaseModel):
    tiers: list[TransactionCostTierConfig] = [
        TransactionCostTierConfig(name="zero", round_trip_bps=0.0),
        TransactionCostTierConfig(name="low", round_trip_bps=10.0),
        TransactionCostTierConfig(name="base", round_trip_bps=30.0),
        TransactionCostTierConfig(name="high", round_trip_bps=80.0),
    ]


class BootstrapConfig(BaseModel):
    n_resamples: int = 10_000
    seed: int = 42
    confidence_level: float = 0.95


class PermutationConfig(BaseModel):
    n_permutations: int = 10_000
    seed: int = 43
    # forward_return_{n}d window used as the permutation test's outcome
    # variable - must be one of targets/forward_returns.py's FORWARD_WINDOWS.
    forward_window: int = 5


class MarketRegimeConfig(BaseModel):
    # Regime label is based purely on the market benchmark's (TOPIX
    # Proxy) own trailing return - no information beyond what Phase 1-3
    # already fetch. Uses features/momentum.py's return_60d column on
    # the benchmark's Feature panel (must be one of RETURN_WINDOWS).
    lookback_days: int = 60
    bull_threshold: float = 0.05
    bear_threshold: float = -0.05


class MinSampleConfig(BaseModel):
    min_oos_trades: int = 30


class WalkForwardValidationConfig(BaseModel):
    """Root of every Phase 6 setting - accessed as config.validation.* to
    read naturally (config.validation.bootstrap.n_resamples, etc).
    """

    walk_forward: WalkForwardConfig = WalkForwardConfig()
    transaction_cost: TransactionCostConfig = TransactionCostConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()
    permutation: PermutationConfig = PermutationConfig()
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    min_sample: MinSampleConfig = MinSampleConfig()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: Path = Path("logs")


class AppConfig(BaseModel):
    data: DataConfig
    universe: UniverseConfig
    features: FeatureConfig = FeatureConfig()
    signals: SignalsConfig = SignalsConfig()
    backtest: BacktestConfig = BacktestConfig()
    scoring: ScoringConfig = ScoringConfig()
    validation: WalkForwardValidationConfig = WalkForwardValidationConfig()
    logging: LoggingConfig = LoggingConfig()


class PriceFilterConfig(BaseModel):
    min_close_price: float | None = None
    max_close_price: float | None = None


class LiquidityFilterConfig(BaseModel):
    min_avg_volume_20d: float | None = None
    min_avg_dollar_volume_20d: float | None = None


class ActivityFilterConfig(BaseModel):
    max_zero_volume_days_in_20: int | None = None


class UniverseFilterConfig(BaseModel):
    price: PriceFilterConfig = PriceFilterConfig()
    liquidity: LiquidityFilterConfig = LiquidityFilterConfig()
    activity: ActivityFilterConfig = ActivityFilterConfig()


def load_app_config(path: Path = Path("config/settings.yaml")) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def load_universe_filters(
    path: Path = Path("config/universe_filters.yaml"),
) -> UniverseFilterConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return UniverseFilterConfig.model_validate(raw or {})


# --- Phase 9 (robustness/sensitivity analysis) -------------------------------
#
# Deliberately kept in a SEPARATE file/model from AppConfig: these values
# have nothing to do with Signal/Score/Backtest/WFO behavior (they only
# parameterize Phase 9's diagnostic re-analyses of already-computed
# trades), and pipeline/run_walk_forward.py's config_hash is intentionally
# computed from ONLY config/settings.yaml + config/universe_filters.yaml -
# adding fields there would make config_hash stop proving "Signal/Score
# unchanged since Phase 6.5/7", which every phase since 6.5 has relied on.
# Every value here is fixed BEFORE running any Phase 9 analysis (spec
# section 10/11: block_length and offsets must not be tuned to the result).


class DayClusterBootstrapConfig(BaseModel):
    n_resamples: int = 10_000
    seed: int = 44
    confidence_level: float = 0.95


class TickerClusterBootstrapConfig(BaseModel):
    """Phase 14 section 15: resamples by TICKER instead of by trading day
    or individual trade, to avoid pseudo-replication bias when the same
    ticker fires long_oversold_rebound repeatedly within one episode -
    Day Cluster Bootstrap alone does not protect against that, since a
    single ticker's multiple trades can still land on different days.
    """

    n_resamples: int = 10_000
    seed: int = 46
    confidence_level: float = 0.95


class BlockBootstrapConfig(BaseModel):
    block_length_days: int = 5
    n_resamples: int = 10_000
    seed: int = 45
    confidence_level: float = 0.95


class TimingPlaceboConfig(BaseModel):
    offsets: list[int] = [-15, -10, -5, -3, -1, 5, 10]


class WinsorizationConfig(BaseModel):
    lower_percentile: float = 0.01
    upper_percentile: float = 0.99


class Phase9Config(BaseModel):
    day_cluster_bootstrap: DayClusterBootstrapConfig = DayClusterBootstrapConfig()
    block_bootstrap: BlockBootstrapConfig = BlockBootstrapConfig()
    timing_placebo: TimingPlaceboConfig = TimingPlaceboConfig()
    winsorization: WinsorizationConfig = WinsorizationConfig()


class Phase14TimingPlaceboConfig(BaseModel):
    # Phase 14 spec section 12's own explicit offset list - deliberately
    # a SEPARATE config from Phase9Config.timing_placebo (which used a
    # coarser [-15,-10,-5,-3,-1,5,10] sweep for Phase 9's own diagnostic
    # purpose): Phase 14 pre-registers this exact, denser list before any
    # real run and must never revise it after seeing results.
    offsets: list[int] = [-15, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10]


class Phase14Config(BaseModel):
    """Phase 14 section 28: every value here is pre-registered - fixed
    and saved to data/walk_forward/phase14_preregistration.json BEFORE
    any real analysis runs, and never edited after seeing results (spec
    section 2/3/29's central prohibition). Kept OUT of AppConfig/
    config_hash, same isolation pattern as Phase9Config
    (config/phase9_settings.yaml) - a Phase 14 diagnostic-analysis
    parameter must never perturb the Strategy Hash integrity anchor.
    """

    # Cumulative/nested TOPIX 20d return thresholds (spec section 2's own
    # grid, adopted verbatim): each day belongs to every threshold whose
    # bound it clears, not to a single mutually-exclusive bucket - see
    # pipeline/run_phase14_validation.py's TOPIX_20D_THRESHOLD_GRID for
    # the paired human-readable labels.
    topix_20d_threshold_grid: list[float] = [-0.05, -0.075, -0.10, -0.125, -0.15]
    # A BEAR episode with at least this many core-condition trades counts
    # as "major" for Leave-One-Episode-Out's individual-episode sweep
    # (spec section 13) - fixed here so which episodes get individually
    # excluded is never chosen after seeing which ones look favorable.
    major_episode_min_trades: int = 5
    ticker_cluster_bootstrap: TickerClusterBootstrapConfig = TickerClusterBootstrapConfig()
    timing_placebo: Phase14TimingPlaceboConfig = Phase14TimingPlaceboConfig()


def load_phase14_config(path: Path = Path("config/phase14_settings.yaml")) -> Phase14Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Phase14Config.model_validate(raw or {})


def load_phase9_config(path: Path = Path("config/phase9_settings.yaml")) -> Phase9Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Phase9Config.model_validate(raw or {})
