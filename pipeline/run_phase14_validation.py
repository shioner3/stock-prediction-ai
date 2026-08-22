"""Phase 14: long_oversold_rebound Conditional Edge Independent Validation.

Phase 13 found (exploratory, no p-hacking safeguards applied) that
long_oversold_rebound's Forward Return edge appears concentrated in
BEAR regime combined with large TOPIX drawdowns, largely independent of
Score. Phase 14 is a SEPARATE, INDEPENDENT, PRE-REGISTERED robustness
validation of that specific hypothesis - never a re-run of Phase 13,
never a Signal/Score/Backtest change, never a new Signal, never a
threshold decision (spec section 1/25).

PRE-REGISTRATION (spec section 2/3/28, the module's central discipline):
every threshold, bucket edge, and offset list below is a FIXED,
module-level constant, frozen here before any real run against Full
Universe data and reviewable via `git log`/`git diff` exactly like every
other "事前固定" convention already used by Phase 6.5-13
(pipeline/run_phase13_conditional_analysis.py's own module docstring is
the direct precedent). save_preregistration() additionally SNAPSHOTS
these frozen values to a JSON file as the very first action
run_phase14_validation() takes - before touching any Signal/Score/
Backtest/Forward Return data - so the saved file's own mtime/git history
proves the values were committed before results existed, not chosen
after. THE THRESHOLD GRID AND EVERY OTHER PRE-REGISTERED VALUE MUST
NEVER BE EDITED AFTER SEEING RESULTS - not even to "improve" a
borderline finding (spec section 2's explicit prohibition, called out as
非常に重要 / very important).

CORE CONDITION: BEAR regime AND TOPIX 20d return <= CORE_TOPIX_20D_THRESHOLD
(-10%) - the specific bucket Phase 13's report concentrated its finding
in (spec section 11's own anchor for the Score Independence Test), and
the bucket backtest/conditional_edge_decision.py's Decision Framework
classifies. CONTROL CONDITION: non-BEAR AND TOPIX 20d return >
CONTROL_TOPIX_20D_THRESHOLD (-5%, the mildest pre-registered grid
threshold's own complement) - the "does the effect show up even without
the regime/drawdown condition" negative comparison.

Reuse (spec section 27's efficiency requirement - compute once, filter/
aggregate per condition, never recompute):
- pipeline.run_phase13_conditional_analysis.load_target_data() for the
  Signal Record / Forward Return+MFE-MAE panel / Combined Backtest trades
  / TOPIX OHLCV / Regime classification bundle (ONE run_backtest() call,
  reused across every condition below, exactly as Phase 13 already does).
- .compute_topix_returns(), .analyze_bucket(), .forward_return_stats(),
  .cost_metrics(), .descriptive_stats(), .load_score_data(),
  .compute_score_analysis(), .bin_label() - Phase 13's own per-bucket
  statistical battery, unmodified.
- pipeline.run_phase9_analysis.lopo_with_bootstrap() for Leave-One-
  Episode-Out and Leave-One-Year-Out (same primitive, two different
  group_col values) and .build_signal_ticker_cache() for the Timing
  Placebo sweep's per-ticker (signal_records, feature panel) cache.
- backtest.day_cluster_bootstrap/.block_bootstrap/.ticker_cluster_bootstrap
  (backtest.ticker_cluster_bootstrap is NEW - see its own module
  docstring for why a third clustering axis was needed) plus
  backtest.bootstrap.bootstrap_ci - all four resampling methods applied
  to the SAME core-condition trade set (spec section 15/16's "Ticker
  Cluster Bootstrapを新規追加する場合は理由を明記する" requirement, met
  by ticker_cluster_bootstrap.py's own module docstring).
- backtest.permutation.permutation_test() + backtest.multiple_testing.
  benjamini_hochberg_correction() - Phase 6-9/12/13's exact, unmodified
  significance-testing pair (Phase 13 caught its own FDR-application gap
  self-review; this module calls benjamini_hochberg_correction()
  directly on fdr_tested_p_values before returning, not as a follow-up
  script this time).
- backtest.episode_analysis.identify_regime_episodes()/
  compute_rich_episode_metrics(), backtest.timing_shift.shift_signal_records(),
  backtest.market_regime.compute_market_regime() - unmodified.
- backtest.conditional_edge_decision.classify_conditional_edge() - the
  new seven-category Decision Framework (spec section 23), itself
  pre-registered (every threshold fixed in that module before this run).

Dependency direction: this module imports FROM signals/backtest/targets/
ensemble/pipeline (Phase 9/13), and must never be imported BY them - see
tests/test_phase9_no_lookahead.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from backtest.block_bootstrap import BlockBootstrapResult, block_bootstrap
from backtest.bootstrap import BootstrapResult, bootstrap_ci
from backtest.conditional_edge_decision import (
    ConditionalEdgeDecision,
    ConditionalEdgeDecisionInputs,
    ConditionalEdgeDecisionResult,
    classify_conditional_edge,
)
from backtest.costs import apply_cost
from backtest.day_cluster_bootstrap import DayClusterBootstrapResult, day_cluster_bootstrap
from backtest.episode_analysis import (
    RegimeEpisode,
    RichEpisodeMetrics,
    compute_rich_episode_metrics,
    identify_regime_episodes,
)
from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.multiple_testing import FDRResult, benjamini_hochberg_correction
from backtest.permutation import PermutationResult, permutation_test
from backtest.ticker_cluster_bootstrap import TickerClusterBootstrapResult, ticker_cluster_bootstrap
from backtest.timing_shift import shift_signal_records
from common.hashing import hash_files
from config.loader import AppConfig, Phase14Config, load_phase14_config
from forward_test.manifest import verify_strategy_hashes_unchanged
from pipeline.run_backtest import run_backtest_for_ticker
from pipeline.run_phase8_analysis import ConfigCheckResult, ConfigMismatchError, verify_config_hash
from pipeline.run_phase9_analysis import (
    AUG_2024_EVENT_END,
    AUG_2024_EVENT_START,
    BEAR_REGIME,
    YEARS,
    LOPOWithCI,
    build_signal_ticker_cache,
    lopo_with_bootstrap,
)
from pipeline.run_phase13_conditional_analysis import (
    TARGET_DIRECTION,
    TARGET_SIGNAL_NAME,
    BucketResult,
    ForwardReturnStats,
    ScoreAnalysis,
    analyze_bucket,
    bin_label,
    compute_score_analysis,
    compute_topix_returns,
    cost_metrics,
    forward_return_stats,
    load_score_data,
    load_target_data,
)
from pipeline.run_walk_forward import CONFIG_FILES
from targets.forward_returns import FORWARD_WINDOWS

logger = logging.getLogger(__name__)

PREREGISTRATION_PATH = Path("data/walk_forward/phase14_preregistration.json")
PHASE14_CONFIG_PATH = Path("config/phase14_settings.yaml")

# --- Pre-registered fixed constants (spec section 2/9/11 - never revisited) ------

CORE_TOPIX_20D_THRESHOLD = -0.10
CONTROL_TOPIX_20D_THRESHOLD = -0.05

# Mutually-exclusive dose-response bins (spec section 9's own example,
# adopted verbatim) - distinct from the CUMULATIVE/nested threshold grid
# below (spec section 2), which reuses the exact same boundary VALUES but
# evaluates each one as "<=threshold" independently, not as a partition.
MARKET_DRAWDOWN_DOSE_RESPONSE_EDGES: list[tuple[float, float, str]] = [
    (float("-inf"), -0.15, "<-15%"),
    (-0.15, -0.125, "-15%~-12.5%"),
    (-0.125, -0.10, "-12.5%~-10%"),
    (-0.10, -0.075, "-10%~-7.5%"),
    (-0.075, -0.05, "-7.5%~-5%"),
    (-0.05, float("inf"), ">=-5%"),
]
DOSE_RESPONSE_ORDER = ["<-15%", "-15%~-12.5%", "-12.5%~-10%", "-10%~-7.5%", "-7.5%~-5%", ">=-5%"]


def threshold_label(threshold: float) -> str:
    return f"<={threshold * 100:g}%"


# --- Pre-registration snapshot (spec section 28 - saved BEFORE any real run) -----


def _json_default(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, date_type):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_preregistration(
    phase14_config: Phase14Config,
    phase13_report_path: Path,
    output_path: Path = PREREGISTRATION_PATH,
) -> dict:
    """Writes the frozen threshold grid / dose-response bins / timing
    placebo offsets / major-episode threshold / core-and-control
    condition definitions, plus integrity hashes proving WHICH
    Signal/Score/Backtest/config/Phase 13 report this pre-registration
    was frozen against, to `output_path`. Called as
    run_phase14_validation()'s very first action (spec section 28's
    ordering requirement) - before any Signal/Score/Backtest/Forward
    Return data is touched.
    """
    snapshot = {
        "phase": "14",
        "purpose": (
            "long_oversold_rebound Conditional Edge Independent Validation - "
            "pre-registered BEFORE any real run, never edited after seeing results"
        ),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "core_condition": {
            "regime": BEAR_REGIME,
            "topix_20d_return_threshold": CORE_TOPIX_20D_THRESHOLD,
            "description": (
                f"regime == {BEAR_REGIME!r} AND topix_return_20d <= {CORE_TOPIX_20D_THRESHOLD}"
            ),
        },
        "control_condition": {
            "regime_excludes": BEAR_REGIME,
            "topix_20d_return_threshold": CONTROL_TOPIX_20D_THRESHOLD,
            "description": (
                f"regime != {BEAR_REGIME!r} AND topix_return_20d > {CONTROL_TOPIX_20D_THRESHOLD}"
            ),
        },
        "topix_20d_threshold_grid": [
            {"threshold": t, "label": threshold_label(t)}
            for t in phase14_config.topix_20d_threshold_grid
        ],
        "topix_20d_threshold_grid_control_label": f">{CONTROL_TOPIX_20D_THRESHOLD * 100:g}%",
        "market_drawdown_dose_response_bins": [
            {"low": low, "high": high, "label": label}
            for low, high, label in MARKET_DRAWDOWN_DOSE_RESPONSE_EDGES
        ],
        "timing_placebo_offsets": phase14_config.timing_placebo.offsets,
        "major_episode_min_trades": phase14_config.major_episode_min_trades,
        "forward_horizons": list(FORWARD_WINDOWS),
        "named_event_windows": {
            "aug_2024": {
                "start": AUG_2024_EVENT_START.isoformat(),
                "end": AUG_2024_EVENT_END.isoformat(),
            },
        },
        "years_swept_for_leave_one_year_out": list(YEARS),
        "ticker_cluster_bootstrap": phase14_config.ticker_cluster_bootstrap.model_dump(),
        "integrity_hashes": {
            "phase14_config_hash": hash_files([PHASE14_CONFIG_PATH]),
            "app_config_hash": hash_files(CONFIG_FILES),
            "phase13_report_hash": (
                hash_files([phase13_report_path]) if phase13_report_path.exists() else None
            ),
        },
        "decision_framework": {
            "module": "backtest.conditional_edge_decision",
            "categories": [d.value for d in ConditionalEdgeDecision],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    logger.info("Phase 14: pre-registration saved to %s", output_path)
    return snapshot


class StrategyHashMismatchError(RuntimeError):
    """Spec section 29 stop condition #1: Strategy Hash changed
    unexpectedly since Strategy Version 1's forward_test manifest was
    saved. Phase 14 must STOP AND REPORT, never proceed."""


# --- Core/control condition + Regime x TOPIX20d cross-tabs (spec section 1/8) ----


def enrich_with_regime_and_topix(
    target_records: pd.DataFrame, regime_df: pd.DataFrame, topix_returns: pd.DataFrame
) -> pd.DataFrame:
    """target_records (ticker, date) + regime + every topix_return_{w}d
    window - the base frame every condition below filters from, built
    ONCE and reused (spec section 27).
    """
    return target_records.merge(regime_df, on="date", how="left").merge(
        topix_returns, on="date", how="left"
    )


def core_condition_mask(enriched: pd.DataFrame) -> pd.Series:
    return (enriched["regime"] == BEAR_REGIME) & (
        enriched["topix_return_20d"] <= CORE_TOPIX_20D_THRESHOLD
    )


def control_condition_mask(enriched: pd.DataFrame) -> pd.Series:
    return (enriched["regime"] != BEAR_REGIME) & (
        enriched["topix_return_20d"] > CONTROL_TOPIX_20D_THRESHOLD
    )


def compute_regime_x_dose_response(
    enriched: pd.DataFrame,
    target_panel: pd.DataFrame,
    target_trades: pd.DataFrame,
    population_forward_5d,
    tiers,
    bootstrap_config,
    phase9_config,
    permutation_config,
) -> list[BucketResult]:
    """Spec section 8: Regime x TOPIX20d dose-response bucket cross-tab -
    every (regime, dose_response_bucket) cell with at least 1 row gets
    the full analyze_bucket() battery, labelled "{regime}:{bucket}".
    """
    df = enriched.assign(
        dose_bucket=enriched["topix_return_20d"].apply(
            lambda v: bin_label(v, MARKET_DRAWDOWN_DOSE_RESPONSE_EDGES)
        )
    )
    results = []
    for (regime, bucket), group in df.groupby(["regime", "dose_bucket"], dropna=True):
        results.append(
            analyze_bucket(
                "regime_x_dose_response", f"{regime}:{bucket}", group[["ticker", "date"]],
                target_panel, target_trades, population_forward_5d, tiers,
                bootstrap_config, phase9_config, permutation_config,
            )
        )
    return results


def compute_threshold_grid(
    enriched: pd.DataFrame,
    phase14_config: Phase14Config,
    target_panel: pd.DataFrame,
    target_trades: pd.DataFrame,
    population_forward_5d,
    tiers,
    bootstrap_config,
    phase9_config,
    permutation_config,
) -> list[BucketResult]:
    """Spec section 2: the pre-registered CUMULATIVE/nested TOPIX 20d
    threshold grid, evaluated AS-IS - each threshold's bucket is
    "topix_return_20d <= threshold" over ALL regimes (not mutually
    exclusive with the other thresholds; a day at -12% belongs to every
    bucket from <=-5% through <=-12.5%), plus the control bucket
    (topix_return_20d > CONTROL_TOPIX_20D_THRESHOLD). This must never be
    used to search for an "optimal" threshold - every entry is reported.
    """
    results = []
    for t in phase14_config.topix_20d_threshold_grid:
        mask = enriched["topix_return_20d"] <= t
        results.append(
            analyze_bucket(
                "topix_20d_threshold", threshold_label(t), enriched[mask][["ticker", "date"]],
                target_panel, target_trades, population_forward_5d, tiers,
                bootstrap_config, phase9_config, permutation_config,
            )
        )
    control_mask = enriched["topix_return_20d"] > CONTROL_TOPIX_20D_THRESHOLD
    results.append(
        analyze_bucket(
            "topix_20d_threshold", f">{CONTROL_TOPIX_20D_THRESHOLD * 100:g}%",
            enriched[control_mask][["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
    )
    return results


# --- Episode-level analysis, LOEO, LOYO, event exclusion (spec 10/13/14) ---------


@dataclass(frozen=True)
class EpisodeAnalysisResult:
    rich_metrics: RichEpisodeMetrics
    is_major: bool
    mean_mfe_5d: float | None
    mean_mae_5d: float | None


def compute_episode_analysis(
    bear_episodes: list[RegimeEpisode],
    core_trades: pd.DataFrame,
    target_records: pd.DataFrame,
    target_panel: pd.DataFrame,
    major_episode_min_trades: int,
) -> list[EpisodeAnalysisResult]:
    """Spec section 10: per-BEAR-episode win rate/PF/expectancy/
    cumulative return/trade count/avg return (all via
    compute_rich_episode_metrics(), reused unmodified) plus MFE/MAE
    (added here - RichEpisodeMetrics does not carry it). "Major" episodes
    (>= major_episode_min_trades CORE-CONDITION trades) are the ones
    Leave-One-Episode-Out individually excludes - fixed BEFORE running,
    never chosen after seeing which episodes look favorable.
    """
    rich = compute_rich_episode_metrics(core_trades, bear_episodes)
    results = []
    for ep, rich_metrics in zip(bear_episodes, rich):
        in_episode = target_records[
            (target_records["date"] >= ep.start_date) & (target_records["date"] <= ep.end_date)
        ]
        merged = in_episode.merge(target_panel, on=["ticker", "date"], how="left")
        mfe = merged["mfe_5d"].dropna() if "mfe_5d" in merged else pd.Series(dtype=float)
        mae = merged["mae_5d"].dropna() if "mae_5d" in merged else pd.Series(dtype=float)
        results.append(
            EpisodeAnalysisResult(
                rich_metrics=rich_metrics,
                is_major=rich_metrics.metrics.n_trades >= major_episode_min_trades,
                mean_mfe_5d=float(mfe.mean()) if len(mfe) else None,
                mean_mae_5d=float(mae.mean()) if len(mae) else None,
            )
        )
    return results


def find_bear_episode_in_month(
    bear_episodes: list[RegimeEpisode], year: int, month: int
) -> RegimeEpisode | None:
    """The BEAR episode (if any) whose [start_date, end_date] overlaps
    the given calendar month - used to identify the "2025-04" episode
    the spec names alongside 2024-08 without hardcoding an arbitrary
    date range: the boundary is derived from the SAME frozen Regime
    classification every other Phase 14 condition already uses, not
    invented after seeing which dates looked favorable.
    """
    for ep in bear_episodes:
        start_in_month = ep.start_date.year == year and ep.start_date.month == month
        end_in_month = ep.end_date.year == year and ep.end_date.month == month
        spans_month = ep.start_date < date_type(year, month, 1) and ep.end_date >= date_type(
            year, month, 1
        )
        if start_in_month or end_in_month or spans_month:
            return ep
    return None


def compute_loeo(
    core_trades: pd.DataFrame,
    episode_analysis: list[EpisodeAnalysisResult],
    bootstrap_config,
) -> list[LOPOWithCI]:
    """Leave-One-Episode-Out (spec section 13): only MAJOR episodes are
    individually excluded (lopo_with_bootstrap() drops rows with a null
    group_col, so non-major-episode trades are simply never grouped and
    never excluded as their own unit - they remain in every resample).
    """
    label_by_episode = {
        (ea.rich_metrics.episode.start_date, ea.rich_metrics.episode.end_date): (
            f"{ea.rich_metrics.episode.start_date}~{ea.rich_metrics.episode.end_date}"
        )
        for ea in episode_analysis
        if ea.is_major
    }
    if not label_by_episode:
        return []

    def _label_for_trade(signal_date: date_type) -> str | None:
        for (start, end), label in label_by_episode.items():
            if start <= signal_date <= end:
                return label
        return None

    trades_with_group = core_trades.assign(
        episode_label=core_trades["signal_date"].apply(_label_for_trade)
    )
    return lopo_with_bootstrap(trades_with_group, "episode_label", bootstrap_config)


def compute_loyo(core_trades: pd.DataFrame, bootstrap_config) -> list[LOPOWithCI]:
    """Leave-One-Year-Out (spec section 14): every year present among
    CORE-CONDITION trades is individually excludable - no majorness
    filter, since a year is not a discretionary choice the way an
    episode boundary can be.
    """
    trades_with_group = core_trades.assign(
        year=core_trades["signal_date"].apply(lambda d: d.year)
    )
    return lopo_with_bootstrap(trades_with_group, "year", bootstrap_config)


@dataclass(frozen=True)
class EventExclusionResult:
    label: str
    n_trades_before: int
    n_trades_after: int
    metrics_after: BacktestMetrics
    bootstrap_expectancy_after: BootstrapResult | None
    positive: bool | None  # None when n_trades_after == 0 (untestable)


def _event_exclusion(
    label: str, core_trades: pd.DataFrame, mask_to_exclude: pd.Series, bootstrap_config
) -> EventExclusionResult:
    remaining = core_trades[~mask_to_exclude]
    metrics_after = compute_metrics(remaining)
    ci = (
        bootstrap_ci(remaining["return"].to_numpy(dtype=float), "expectancy", bootstrap_config)
        if not remaining.empty
        else None
    )
    if metrics_after.n_trades == 0:
        positive = None
    else:
        positive = metrics_after.expectancy is not None and metrics_after.expectancy > 0
    return EventExclusionResult(
        label=label, n_trades_before=len(core_trades), n_trades_after=len(remaining),
        metrics_after=metrics_after, bootstrap_expectancy_after=ci, positive=positive,
    )


def compute_event_exclusions(
    core_trades: pd.DataFrame,
    apr_2025_episode: RegimeEpisode | None,
    bootstrap_config,
) -> tuple[EventExclusionResult, EventExclusionResult, EventExclusionResult]:
    """Spec section 25: exclude 2024-08, exclude 2025-04, exclude both -
    each with its own before/after expectancy + bootstrap CI. If no
    2025-04 BEAR episode was found in this run's data, that exclusion
    mask is simply empty (nothing removed, "before"=="after") rather
    than raising - the spec treats "such an episode did not occur" as
    informative, not an error.
    """
    dates = core_trades["signal_date"]
    aug_mask = (dates >= AUG_2024_EVENT_START) & (dates <= AUG_2024_EVENT_END)
    if apr_2025_episode is not None:
        apr_mask = (dates >= apr_2025_episode.start_date) & (dates <= apr_2025_episode.end_date)
    else:
        apr_mask = pd.Series(False, index=core_trades.index)

    exclude_aug2024 = _event_exclusion("exclude_2024_08", core_trades, aug_mask, bootstrap_config)
    exclude_apr2025 = _event_exclusion("exclude_2025_04", core_trades, apr_mask, bootstrap_config)
    exclude_both = _event_exclusion(
        "exclude_2024_08_and_2025_04", core_trades, aug_mask | apr_mask, bootstrap_config
    )
    return exclude_aug2024, exclude_apr2025, exclude_both


# --- Timing Placebo sweep (spec section 12) --------------------------------------


@dataclass(frozen=True)
class TimingPlaceboOffsetResult:
    offset_days: int
    n_core_trades: int
    metrics: BacktestMetrics
    core_positive: bool  # expectancy > 0 and profit_factor > 1 at this offset


def run_timing_placebo_sweep(
    signal_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    offsets: list[int],
    regime_df: pd.DataFrame,
    topix_returns: pd.DataFrame,
    backtest_config,
) -> list[TimingPlaceboOffsetResult]:
    """Shifts long_oversold_rebound's own trigger dates by each
    pre-registered offset (backtest.timing_shift.shift_signal_records(),
    unmodified) and re-backtests the FULL shifted signal set per ticker
    (backtest.engine.run_backtest_for_ticker(), unmodified) - NOT
    pre-filtered to core-condition dates, because the engine suppresses
    overlapping same-signal trades per (ticker, signal_name, direction)
    (backtest/engine.py), and removing non-core-condition occurrences
    before backtesting would strip suppression history the real Combined
    backtest actually applies, fabricating trades that would not exist
    in run_phase13_conditional_analysis.load_target_data()'s own
    core_trades population. THEN, exactly like the main pipeline and
    Phase 9's own established Timing Placebo precedent
    (pipeline.run_phase9_analysis._run_timing_offset_sweep), the CORE
    CONDITION filter is applied to the resulting TRADES afterward, using
    the SHIFTED signal_date's own regime/TOPIX value (not the original
    trigger date's) - this answers "if the signal had fired offset_days
    earlier/later, would BEAR/large-drawdown days it landed on still
    show an edge". The signal condition itself never changes; only which
    date gets evaluated shifts (spec's own framing).
    """
    condition = regime_df.merge(topix_returns, on="date", how="left")
    condition_by_date = {
        row.date: (row.regime == BEAR_REGIME and row.topix_return_20d is not None
                   and row.topix_return_20d <= CORE_TOPIX_20D_THRESHOLD)
        for row in condition.itertuples(index=False)
    }

    results = []
    for offset in offsets:
        trade_frames = []
        for sig, panel in signal_cache.values():
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
        core_mask = combined["signal_date"].apply(lambda d: condition_by_date.get(d, False))
        combined = combined[core_mask]
        metrics = compute_metrics(combined)
        core_positive = (
            metrics.n_trades > 0
            and metrics.expectancy is not None and metrics.expectancy > 0
            and metrics.profit_factor is not None and metrics.profit_factor > 1.0
        )
        results.append(
            TimingPlaceboOffsetResult(
                offset_days=offset, n_core_trades=len(combined), metrics=metrics,
                core_positive=core_positive,
            )
        )
    return results


def timing_placebo_positive_fraction(sweep: list[TimingPlaceboOffsetResult]) -> float | None:
    """Fraction of NON-ZERO offsets that clear core_positive - the
    Decision Framework's TIMING_DEPENDENT input (backtest.
    conditional_edge_decision's own docstring explains the direction:
    a HIGH fraction means shifted placebo dates work just as well as the
    real trigger, so the edge is not Signal-specific).
    """
    nonzero = [r for r in sweep if r.offset_days != 0]
    if not nonzero:
        return None
    return sum(1 for r in nonzero if r.core_positive) / len(nonzero)


# --- Market Drawdown dose-response (spec section 9) -------------------------------


def compute_dose_response(
    enriched: pd.DataFrame,
    target_panel: pd.DataFrame,
    target_trades: pd.DataFrame,
    population_forward_5d,
    tiers,
    bootstrap_config,
    phase9_config,
    permutation_config,
) -> list[BucketResult]:
    """Mutually-exclusive TOPIX 20d drawdown bins (ALL regimes, unlike
    the BEAR-only core condition) - full analyze_bucket() battery since
    this dose-response curve is one of Phase 14's own headline questions
    (spec section 9), not a secondary/exploratory axis.
    """
    df = enriched.assign(
        dose_bucket=enriched["topix_return_20d"].apply(
            lambda v: bin_label(v, MARKET_DRAWDOWN_DOSE_RESPONSE_EDGES)
        )
    )
    return [
        analyze_bucket(
            "dose_response", label, group[["ticker", "date"]], target_panel, target_trades,
            population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
        )
        for label, group in df.groupby("dose_bucket", dropna=True)
    ]


# --- Forward Horizon BEAR vs NON-BEAR (spec section 17) ---------------------------


@dataclass(frozen=True)
class ForwardHorizonComparison:
    window_days: int
    bear: ForwardReturnStats
    non_bear: ForwardReturnStats


def compute_forward_horizon_comparison(
    enriched: pd.DataFrame, target_panel: pd.DataFrame
) -> list[ForwardHorizonComparison]:
    bear_keys = enriched[enriched["regime"] == BEAR_REGIME][["ticker", "date"]]
    non_bear_keys = enriched[
        enriched["regime"].notna() & (enriched["regime"] != BEAR_REGIME)
    ][["ticker", "date"]]
    bear_stats = {s.window_days: s for s in forward_return_stats(bear_keys, target_panel)}
    non_bear_stats = {s.window_days: s for s in forward_return_stats(non_bear_keys, target_panel)}
    return [
        ForwardHorizonComparison(window_days=w, bear=bear_stats[w], non_bear=non_bear_stats[w])
        for w in FORWARD_WINDOWS
    ]


# --- Transaction Cost sensitivity (spec section 18) -------------------------------


@dataclass(frozen=True)
class CostSensitivityResult:
    label: str
    metrics_by_tier: dict[str, BacktestMetrics]


def compute_cost_sensitivity(
    core_trades: pd.DataFrame,
    apr_2025_episode: RegimeEpisode | None,
    episode_analysis: list[EpisodeAnalysisResult],
    tiers,
) -> list[CostSensitivityResult]:
    """Spec section 18: cost tier sensitivity for the core condition,
    each named-event exclusion, and each major episode's LOEO remainder
    - reuses pipeline.run_phase13_conditional_analysis.cost_metrics()
    (the same 4-tier apply_cost()+compute_metrics() battery every other
    Phase 13/14 bucket already uses) rather than a new cost calculation.
    """
    results = [CostSensitivityResult("core", cost_metrics(core_trades, tiers))]

    aug_mask = (
        (core_trades["signal_date"] >= AUG_2024_EVENT_START)
        & (core_trades["signal_date"] <= AUG_2024_EVENT_END)
    )
    results.append(
        CostSensitivityResult("excl_2024_08", cost_metrics(core_trades[~aug_mask], tiers))
    )
    if apr_2025_episode is not None:
        apr_mask = (
            (core_trades["signal_date"] >= apr_2025_episode.start_date)
            & (core_trades["signal_date"] <= apr_2025_episode.end_date)
        )
        results.append(
            CostSensitivityResult("excl_2025_04", cost_metrics(core_trades[~apr_mask], tiers))
        )

    for ea in episode_analysis:
        if not ea.is_major:
            continue
        ep = ea.rich_metrics.episode
        mask = (core_trades["signal_date"] >= ep.start_date) & (
            core_trades["signal_date"] <= ep.end_date
        )
        results.append(
            CostSensitivityResult(
                f"excl_episode_{ep.start_date}~{ep.end_date}",
                cost_metrics(core_trades[~mask], tiers),
            )
        )
    return results


# --- Bootstrap battery (spec section 15/16) ---------------------------------------


@dataclass(frozen=True)
class BootstrapBattery:
    trade_level: BootstrapResult
    day_cluster: DayClusterBootstrapResult
    block: BlockBootstrapResult
    ticker_cluster: TickerClusterBootstrapResult


def compute_bootstrap_battery(
    core_trades: pd.DataFrame, tiers, bootstrap_config, phase9_config, phase14_config
) -> BootstrapBattery:
    """All FOUR resampling methods on the SAME core-condition trade set,
    "expectancy" statistic throughout: trade-level bootstrap_ci() (base
    cost tier, matching analyze_bucket()'s own convention), day/block/
    ticker cluster bootstrap on raw (uncosted) returns (matching
    day_cluster_bootstrap.py/block_bootstrap.py's own established
    convention, unmodified).
    """
    base_tier = next(t for t in tiers if t.name == "base")
    base_returns = apply_cost(core_trades["return"], base_tier)
    return BootstrapBattery(
        trade_level=bootstrap_ci(base_returns.to_numpy(), "expectancy", bootstrap_config),
        day_cluster=day_cluster_bootstrap(
            core_trades, "expectancy", phase9_config.day_cluster_bootstrap
        ),
        block=block_bootstrap(core_trades, "expectancy", phase9_config.block_bootstrap),
        ticker_cluster=ticker_cluster_bootstrap(
            core_trades, "expectancy", phase14_config.ticker_cluster_bootstrap
        ),
    )


# --- Permutation + FDR (spec section 20/21) ---------------------------------------


def compute_permutation_battery(
    enriched: pd.DataFrame,
    target_panel: pd.DataFrame,
    population_forward_5d,
    phase14_config: Phase14Config,
    permutation_config,
) -> dict[str, PermutationResult]:
    """Permutation p-value for Overall / BEAR / NON-BEAR / TOPIX<=-10%
    (any regime) / each pre-registered threshold (spec section 20) - the
    dict key is later fed straight into benjamini_hochberg_correction()
    for the FDR-adjusted p-values every one of these tests shares (spec
    section 21).
    """

    def _p(keys: pd.DataFrame) -> PermutationResult:
        merged = keys[["ticker", "date"]].merge(target_panel, on=["ticker", "date"], how="left")
        signal_returns = merged["forward_return_5d"].dropna().to_numpy()
        return permutation_test(signal_returns, population_forward_5d, permutation_config)

    results = {
        "overall": _p(enriched[["ticker", "date"]]),
        "bear": _p(enriched[enriched["regime"] == BEAR_REGIME][["ticker", "date"]]),
        "non_bear": _p(
            enriched[enriched["regime"].notna() & (enriched["regime"] != BEAR_REGIME)][
                ["ticker", "date"]
            ]
        ),
        "core": _p(enriched[core_condition_mask(enriched)][["ticker", "date"]]),
    }
    for t in phase14_config.topix_20d_threshold_grid:
        results[f"threshold:{threshold_label(t)}"] = _p(
            enriched[enriched["topix_return_20d"] <= t][["ticker", "date"]]
        )
    return results


# --- Decision Framework wiring (spec section 23) -----------------------------------


def build_decision_inputs(
    core_condition_bucket: BucketResult,
    control_condition_bucket: BucketResult,
    exclude_aug2024: EventExclusionResult,
    exclude_apr2025: EventExclusionResult,
    loeo: list[LOPOWithCI],
    loyo: list[LOPOWithCI],
    timing_sweep: list[TimingPlaceboOffsetResult],
    score_independence_core: ScoreAnalysis,
    fdr_results: dict[str, FDRResult],
) -> ConditionalEdgeDecisionInputs:
    expectancy_ci_low = (
        core_condition_bucket.bootstrap_expectancy.ci_low
        if core_condition_bucket.bootstrap_expectancy is not None
        else None
    )
    profit_factor_high_cost = (
        core_condition_bucket.cost_metrics["high"].profit_factor
        if core_condition_bucket.cost_metrics is not None
        else None
    )
    core_fdr = fdr_results.get("core")
    permutation_p_fdr = core_fdr.adjusted_p_value if core_fdr is not None else None

    positive_excluding_each_major_episode = (
        None
        if not loeo
        else all(
            r.remaining_metrics.expectancy is not None and r.remaining_metrics.expectancy > 0
            for r in loeo
        )
    )
    positive_excluding_each_year = (
        None
        if not loyo
        else all(
            r.remaining_metrics.expectancy is not None and r.remaining_metrics.expectancy > 0
            for r in loyo
        )
    )

    control_positive = None
    if control_condition_bucket.cost_metrics is not None:
        base = control_condition_bucket.cost_metrics.get("base")
        if base is not None:
            control_positive = base.expectancy is not None and base.expectancy > 0

    score_no_discrimination = None
    if score_independence_core.monotonic is not None:
        has_positive_spread = (
            score_independence_core.q5_q1_spread is not None
            and score_independence_core.q5_q1_spread > 0
        )
        score_no_discrimination = not (
            score_independence_core.monotonic and has_positive_spread
        )

    return ConditionalEdgeDecisionInputs(
        n_sample=core_condition_bucket.n,
        expectancy_ci_low=expectancy_ci_low,
        profit_factor_high_cost=profit_factor_high_cost,
        permutation_p_value_fdr_adjusted=permutation_p_fdr,
        positive_excluding_aug2024=exclude_aug2024.positive,
        positive_excluding_apr2025=exclude_apr2025.positive,
        positive_excluding_each_major_episode=positive_excluding_each_major_episode,
        positive_excluding_each_year=positive_excluding_each_year,
        timing_placebo_positive_fraction=timing_placebo_positive_fraction(timing_sweep),
        control_bucket_also_positive=control_positive,
        score_adds_no_discriminating_power=score_no_discrimination,
    )


# --- Top-level report ----------------------------------------------------------------


@dataclass(frozen=True)
class Phase14Report:
    config_check: ConfigCheckResult
    strategy_hash_matches: bool
    strategy_hash_mismatches: list[str]
    preregistration: dict
    tickers: list[str]
    n_core_condition_trades: int
    core_condition_bucket: BucketResult
    control_condition_bucket: BucketResult
    regime_x_dose_response: list[BucketResult]
    threshold_grid: list[BucketResult]
    bear_episodes: list[EpisodeAnalysisResult]
    apr_2025_episode: RegimeEpisode | None
    loeo: list[LOPOWithCI]
    loyo: list[LOPOWithCI]
    exclude_aug2024: EventExclusionResult
    exclude_apr2025: EventExclusionResult
    exclude_both: EventExclusionResult
    timing_placebo: list[TimingPlaceboOffsetResult]
    dose_response: list[BucketResult]
    score_independence_core: ScoreAnalysis
    forward_horizon_comparison: list[ForwardHorizonComparison]
    cost_sensitivity: list[CostSensitivityResult]
    bootstrap_battery: BootstrapBattery
    permutation_battery: dict[str, PermutationResult]
    fdr_results: dict[str, FDRResult]
    decision: ConditionalEdgeDecisionResult


def run_phase14_validation(
    config: AppConfig,
    tickers: list[str],
    phase6_5_report_path: Path,
    phase7_report_path: Path,
    phase13_report_path: Path,
    strategy_manifest: dict,
    preregistration_path: Path = PREREGISTRATION_PATH,
) -> Phase14Report:
    # --- Stop conditions checked BEFORE any real analysis (spec section 29) ------
    config_check = verify_config_hash(phase6_5_report_path, phase7_report_path)
    if not config_check.matches:
        raise ConfigMismatchError(f"CONFIG_MISMATCH: {config_check}")

    strategy_hash_matches, strategy_hash_mismatches = verify_strategy_hashes_unchanged(
        strategy_manifest
    )
    if not strategy_hash_matches:
        raise StrategyHashMismatchError(
            f"STRATEGY_HASH_MISMATCH: {strategy_hash_mismatches}"
        )

    # --- Pre-registration: saved BEFORE any Signal/Score/Backtest/Forward Return
    # data is touched (spec section 28's ordering requirement) -------------------
    phase14_config = load_phase14_config()
    preregistration = save_preregistration(
        phase14_config, phase13_report_path, preregistration_path
    )

    # --- Load once, reuse everywhere (spec section 27) ---------------------------
    logger.info("Phase 14: loading Target Data (Signal/Forward Return/Backtest/Regime)")
    data = load_target_data(config, tickers)
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

    topix_returns = compute_topix_returns(topix)
    enriched = enrich_with_regime_and_topix(target_records, regime_df, topix_returns)
    target_trades_enriched = target_trades.merge(
        regime_df, left_on="signal_date", right_on="date", how="left", suffixes=("", "_regime")
    ).merge(
        topix_returns, left_on="signal_date", right_on="date", how="left", suffixes=("", "_topix")
    )
    core_trades = target_trades_enriched[core_condition_mask(target_trades_enriched)].copy()

    # --- Core / control condition buckets (spec section 1/11) --------------------
    logger.info("Phase 14: core/control condition buckets")
    core_keys = enriched[core_condition_mask(enriched)][["ticker", "date"]]
    control_keys = enriched[control_condition_mask(enriched)][["ticker", "date"]]
    core_condition_bucket = analyze_bucket(
        "core_condition", "BEAR_x_topix20d<=-10%", core_keys, target_panel, target_trades,
        population_forward_5d, tiers, bootstrap_config, phase9_config, permutation_config,
    )
    control_condition_bucket = analyze_bucket(
        "control_condition", "non_BEAR_x_topix20d>-5%", control_keys, target_panel,
        target_trades, population_forward_5d, tiers, bootstrap_config, phase9_config,
        permutation_config,
    )

    # --- Cross-tabs + threshold grid (spec section 2/8) ---------------------------
    logger.info("Phase 14: regime x dose-response cross-tab + pre-registered threshold grid")
    regime_x_dose_response = compute_regime_x_dose_response(
        enriched, target_panel, target_trades, population_forward_5d, tiers,
        bootstrap_config, phase9_config, permutation_config,
    )
    threshold_grid = compute_threshold_grid(
        enriched, phase14_config, target_panel, target_trades, population_forward_5d,
        tiers, bootstrap_config, phase9_config, permutation_config,
    )

    # --- Episodes, LOEO, LOYO, event exclusion (spec section 10/13/14/25) --------
    logger.info("Phase 14: episode analysis, LOEO, LOYO, event exclusion")
    bear_episodes = identify_regime_episodes(regime_df, BEAR_REGIME)
    episode_analysis = compute_episode_analysis(
        bear_episodes, core_trades, target_records, target_panel,
        phase14_config.major_episode_min_trades,
    )
    apr_2025_episode = find_bear_episode_in_month(bear_episodes, 2025, 4)
    loeo = compute_loeo(core_trades, episode_analysis, bootstrap_config)
    loyo = compute_loyo(core_trades, bootstrap_config)
    exclude_aug2024, exclude_apr2025, exclude_both = compute_event_exclusions(
        core_trades, apr_2025_episode, bootstrap_config
    )

    # --- Timing Placebo sweep (spec section 12) -----------------------------------
    logger.info(
        "Phase 14: timing placebo sweep over %d offsets", len(phase14_config.timing_placebo.offsets)
    )
    signal_cache = build_signal_ticker_cache(config, tickers, TARGET_DIRECTION, TARGET_SIGNAL_NAME)
    timing_sweep = run_timing_placebo_sweep(
        signal_cache, phase14_config.timing_placebo.offsets, regime_df, topix_returns,
        config.backtest,
    )

    # --- Dose-response, Score Independence, Forward Horizon, Cost (spec 9/24/17/18)
    logger.info("Phase 14: dose-response, score independence, forward horizon, cost sensitivity")
    dose_response = compute_dose_response(
        enriched, target_panel, target_trades, population_forward_5d, tiers,
        bootstrap_config, phase9_config, permutation_config,
    )
    scores_dir = Path(config.data.scores_dir)
    score_data = load_score_data(tickers, scores_dir)
    score_independence_core = compute_score_analysis(core_keys, target_panel, score_data)
    forward_horizon_comparison = compute_forward_horizon_comparison(enriched, target_panel)
    cost_sensitivity = compute_cost_sensitivity(
        core_trades, apr_2025_episode, episode_analysis, tiers
    )

    # --- Bootstrap battery + Permutation/FDR (spec section 15/16/20/21) ----------
    logger.info("Phase 14: 4-method bootstrap battery + permutation/FDR")
    bootstrap_battery = compute_bootstrap_battery(
        core_trades, tiers, bootstrap_config, phase9_config, phase14_config
    )
    permutation_battery = compute_permutation_battery(
        enriched, target_panel, population_forward_5d, phase14_config, permutation_config
    )
    fdr_results = benjamini_hochberg_correction(
        {key: r.p_value for key, r in permutation_battery.items()}
    )

    # --- Decision Framework (spec section 23) -------------------------------------
    decision_inputs = build_decision_inputs(
        core_condition_bucket, control_condition_bucket, exclude_aug2024, exclude_apr2025,
        loeo, loyo, timing_sweep, score_independence_core, fdr_results,
    )
    decision = classify_conditional_edge(decision_inputs)

    return Phase14Report(
        config_check=config_check,
        strategy_hash_matches=strategy_hash_matches,
        strategy_hash_mismatches=strategy_hash_mismatches,
        preregistration=preregistration,
        tickers=tickers,
        n_core_condition_trades=len(core_trades),
        core_condition_bucket=core_condition_bucket,
        control_condition_bucket=control_condition_bucket,
        regime_x_dose_response=regime_x_dose_response,
        threshold_grid=threshold_grid,
        bear_episodes=episode_analysis,
        apr_2025_episode=apr_2025_episode,
        loeo=loeo,
        loyo=loyo,
        exclude_aug2024=exclude_aug2024,
        exclude_apr2025=exclude_apr2025,
        exclude_both=exclude_both,
        timing_placebo=timing_sweep,
        dose_response=dose_response,
        score_independence_core=score_independence_core,
        forward_horizon_comparison=forward_horizon_comparison,
        cost_sensitivity=cost_sensitivity,
        bootstrap_battery=bootstrap_battery,
        permutation_battery=permutation_battery,
        fdr_results=fdr_results,
        decision=decision,
    )
