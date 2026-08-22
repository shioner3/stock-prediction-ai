"""Phase 8: BEAR regime reproducibility verification for
long_oversold_rebound - the only Signal that reached ACCEPT_CANDIDATE in
Phase 7. Every existing Signal/Score/Backtest/WFO/Cost/Bootstrap/
Permutation/Decision component is reused completely unchanged - this
module only SLICES already-computed results (by WFO window, by calendar
year, by BEAR episode, by ticker) and runs sensitivity checks
(Leave-One-Period-Out, a Placebo negative control) on top of them. It is
diagnostic, not optimization: no threshold, weight, or Signal condition
is read from or written by this module, and nothing here feeds back into
backtest/decision.py's Decision framework.

"Combined OOS" (spec section 3/4) is computed by calling
pipeline.run_walk_forward.run_walk_forward() with min_oos_start=None over
data/phase7/'s dataset, which already spans the FULL Phase 6.5 + Phase 7
period (2022-01-04..2026-08-20) continuously - see
scripts/run_phase7_report.py's docstring for why that dataset already
covers this range. This is a genuine recomputation (not a splice of the
two saved JSON reports), using ONE consistent ticker Universe throughout,
which avoids silently mixing Phase 6.5's 2,755-ticker Universe with
Phase 7's 2,880-ticker Universe in one "combined" statistic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.bootstrap import STATISTIC_NAMES, BootstrapResult, bootstrap_ci
from backtest.costs import apply_cost
from backtest.cross_sectional import (
    ConcentrationMetrics,
    TopKShare,
    compute_concentration,
    compute_top_k_shares,
)
from backtest.engine import run_backtest_for_ticker
from backtest.episode_analysis import (
    EpisodeMetrics,
    RegimeEpisode,
    compute_episode_metrics,
    identify_regime_episodes,
)
from backtest.leave_one_period_out import LeaveOneOutResult, leave_one_period_out
from backtest.market_regime import compute_market_regime
from backtest.metrics import BacktestMetrics, compute_metrics, compute_metrics_by_ticker
from backtest.permutation import PermutationResult, permutation_test
from backtest.regime_reproducibility import (
    RegimeWindowMetrics,
    RegimeYearMetrics,
    compute_regime_by_window,
    compute_regime_by_year,
)
from backtest.walk_forward import WalkForwardWindow
from common.hashing import hash_files
from config.loader import AppConfig
from pipeline.run_backtest import run_backtest
from pipeline.run_walk_forward import (
    CONFIG_FILES,
    SignalWalkForwardResult,
    WalkForwardReport,
    run_walk_forward,
)
from storage.parquet_store import load_feature_panel, load_ohlcv, load_signal_records
from targets.forward_returns import compute_forward_returns

logger = logging.getLogger(__name__)

BEAR_REGIME = "BEAR"
YEARS = (2022, 2023, 2024, 2025, 2026)
TOP_K_VALUES = (1, 5, 10, 20)
# Trading-day (not calendar-day) backward shift for the Placebo negative
# control - chosen to be well outside HOLD_DAYS=5 so placebo trades never
# overlap the real signal's own entry/exit windows, while staying small
# enough that most triggered tickers still have room to shift within
# their own history (rows that would shift before day 0 are dropped, not
# wrapped or clamped - see _shift_signal_records_backward()).
PLACEBO_SHIFT_TRADING_DAYS = 15


class ConfigMismatchError(RuntimeError):
    """Raised when the CURRENT config_hash does not match both Phase 6.5's
    and Phase 7's saved config_hash - spec section 2's CONFIG_MISMATCH
    stop condition. Phase 8 must never silently proceed with a changed
    Signal/Score/Backtest/WFO configuration.
    """


@dataclass
class ConfigCheckResult:
    matches: bool
    current_config_hash: str
    phase6_5_config_hash: str | None
    phase7_config_hash: str | None


def verify_config_hash(phase6_5_report_path: Path, phase7_report_path: Path) -> ConfigCheckResult:
    current_hash = hash_files(CONFIG_FILES)
    p65_hash = None
    p7_hash = None
    if phase6_5_report_path.exists():
        p65_hash = json.loads(phase6_5_report_path.read_text(encoding="utf-8"))["config_hash"]
    if phase7_report_path.exists():
        p7_hash = json.loads(phase7_report_path.read_text(encoding="utf-8"))["config_hash"]

    matches = p65_hash is not None and p7_hash is not None and current_hash == p65_hash == p7_hash
    return ConfigCheckResult(matches, current_hash, p65_hash, p7_hash)


@dataclass
class PlaceboResult:
    shift_trading_days: int
    n_placebo_trades_total: int
    n_placebo_trades_bear: int
    placebo_bear_metrics: BacktestMetrics
    real_bear_metrics: BacktestMetrics


@dataclass
class Phase8SignalReport:
    direction: str
    signal_name: str
    combined: SignalWalkForwardResult
    bear_by_window: list[RegimeWindowMetrics]
    bear_by_year: dict[int, RegimeYearMetrics]
    bear_episodes: list[EpisodeMetrics]
    bear_ticker_metrics: dict[str, BacktestMetrics]
    bear_concentration: ConcentrationMetrics
    bear_top_k_shares: dict[int, TopKShare]
    bear_cost_sensitivity: dict[str, BacktestMetrics]
    bear_bootstrap: dict[str, BootstrapResult]
    bear_permutation: PermutationResult | None
    lopo_by_year: list[LeaveOneOutResult]
    lopo_by_episode: list[LeaveOneOutResult]
    placebo: PlaceboResult


@dataclass
class OtherSignalSummary:
    direction: str
    signal_name: str
    combined_pf_base: float | None
    combined_expectancy_base: float | None
    phase6_5_pf_base: float | None
    phase7_pf_base: float | None
    combined_high_cost_pf: float | None
    decision_combined: str


@dataclass
class Phase8Report:
    config_check: ConfigCheckResult
    windows: list[WalkForwardWindow]
    bear_episodes_all: list[RegimeEpisode]
    target: Phase8SignalReport
    other_signals: list[OtherSignalSummary]
    tickers: list[str]
    data_start: date | None
    data_end: date | None


def _bear_signal_forward_returns(
    config: AppConfig,
    tickers: list[str],
    direction: str,
    signal_name: str,
    regime_df: pd.DataFrame,
    forward_window: int,
) -> np.ndarray:
    """Forward Return (targets/forward_returns.py, NOT Trade Return) for
    every (ticker, date) where this Signal triggered AND that date is
    BEAR-classified - the "signal" side of the BEAR-restricted
    Permutation Test (spec section 10).
    """
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)
    col = f"forward_return_{forward_window}d"
    values: list[float] = []
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
        merged = sig.merge(forward_with_date, on="date", how="left").merge(
            regime_df, on="date", how="left"
        )
        bear_rows = merged[merged["regime"] == BEAR_REGIME]
        values.extend(bear_rows[col].dropna().tolist())
    return np.array(values, dtype=float)


def _bear_population_forward_returns(
    config: AppConfig, tickers: list[str], regime_df: pd.DataFrame, forward_window: int
) -> np.ndarray:
    """Forward Return for EVERY row of every ticker's Feature panel
    (triggered or not) that falls on a BEAR-classified date - the
    "population" side of the BEAR-restricted Permutation Test, mirroring
    pipeline/run_walk_forward.py's _forward_return_population() but
    restricted to BEAR dates instead of OOS-window dates.
    """
    features_dir = Path(config.data.features_dir)
    col = f"forward_return_{forward_window}d"
    values: list[float] = []
    for ticker in tickers:
        panel = load_feature_panel(ticker, features_dir)
        forward = compute_forward_returns(panel)
        frame = panel[["date"]].join(forward).merge(regime_df, on="date", how="left")
        bear_rows = frame[frame["regime"] == BEAR_REGIME]
        values.extend(bear_rows[col].dropna().tolist())
    return np.array(values, dtype=float)


def _assign_episode_index(dates: pd.Series, episodes: list[RegimeEpisode]) -> pd.Series:
    """Maps each date to the (single) BEAR episode whose [start,end]
    range contains it - well-defined for BEAR-regime-filtered trades,
    since episodes partition every BEAR-classified trading day exactly
    once (see backtest/episode_analysis.py). Dates outside every episode
    (shouldn't occur for already-BEAR-filtered input) get None.
    """
    result: pd.Series = pd.Series([None] * len(dates), index=dates.index, dtype=object)
    for ep in episodes:
        mask = (dates >= ep.start_date) & (dates <= ep.end_date)
        result = result.mask(mask, ep.index)
    return result


def _shift_signal_records_backward(
    signal_records: pd.DataFrame, panel: pd.DataFrame, shift_days: int
) -> pd.DataFrame:
    """Shifts each row's date BACKWARD by `shift_days` TRADING-DAY index
    positions within this ticker's own date index (never forward, and
    never by calendar days - trading-day positions keep the shifted date
    a real trading day). A backward shift only ever produces a date that
    is chronologically <= the original, so it cannot introduce future
    information relative to what the real Signal used - this is the
    "safe direction" for a look-ahead-free Placebo (spec section 12).
    Rows that would shift before the start of the panel are dropped.
    """
    dates = panel["date"].tolist()
    pos_by_date = {d: i for i, d in enumerate(dates)}
    rows: list[dict] = []
    for row in signal_records.itertuples(index=False):
        pos = pos_by_date.get(row.date)
        if pos is None:
            continue
        new_pos = pos - shift_days
        if new_pos < 0:
            continue
        shifted = row._asdict()
        shifted["date"] = dates[new_pos]
        rows.append(shifted)
    if not rows:
        return signal_records.iloc[0:0].copy()
    return pd.DataFrame(rows)[list(signal_records.columns)]


def _run_placebo(
    config: AppConfig,
    tickers: list[str],
    direction: str,
    signal_name: str,
    regime_df: pd.DataFrame,
    real_bear_trades: pd.DataFrame,
    shift_days: int = PLACEBO_SHIFT_TRADING_DAYS,
) -> PlaceboResult:
    """Negative control: re-triggers the SAME Signal's dates shifted
    PLACEBO_SHIFT_TRADING_DAYS earlier (not a different Signal, not a
    random draw), re-runs them through the UNCHANGED Backtest Engine,
    and compares the resulting BEAR-regime performance to the real
    Signal's. If the placebo ALSO shows strong BEAR performance, that
    would suggest "being long during BEAR regime dates in general" (not
    the oversold-rebound setup specifically) drives the result.
    """
    signals_dir = Path(config.data.signals_dir)
    features_dir = Path(config.data.features_dir)
    placebo_trade_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        signal_records = load_signal_records(ticker, signals_dir)
        sig = signal_records[
            (signal_records["direction"] == direction)
            & (signal_records["signal_name"] == signal_name)
        ]
        if sig.empty:
            continue
        panel = load_feature_panel(ticker, features_dir)
        shifted = _shift_signal_records_backward(sig, panel, shift_days)
        if shifted.empty:
            continue
        result = run_backtest_for_ticker(shifted, panel, config.backtest)
        if not result.trades.empty:
            placebo_trade_frames.append(result.trades)

    if placebo_trade_frames:
        placebo_trades = pd.concat(placebo_trade_frames, ignore_index=True)
    else:
        placebo_trades = pd.DataFrame(columns=["ticker", "signal_date", "return"])

    merged = placebo_trades.merge(regime_df, left_on="signal_date", right_on="date", how="left")
    placebo_bear = merged[merged["regime"] == BEAR_REGIME]

    return PlaceboResult(
        shift_trading_days=shift_days,
        n_placebo_trades_total=len(placebo_trades),
        n_placebo_trades_bear=len(placebo_bear),
        placebo_bear_metrics=compute_metrics(placebo_bear),
        real_bear_metrics=compute_metrics(real_bear_trades),
    )


def _build_other_signals_summary(
    combined_by_signal: dict[tuple[str, str], SignalWalkForwardResult],
    phase6_5_report: dict,
    phase7_report: dict,
    target_key: tuple[str, str],
) -> list[OtherSignalSummary]:
    p65_by_key = {
        (s["direction"], s["signal_name"]): s for s in phase6_5_report.get("signal_results", [])
    }
    p7_by_key = {
        (s["direction"], s["signal_name"]): s for s in phase7_report.get("signal_results", [])
    }

    def _pf(entry: dict | None) -> float | None:
        if entry is None:
            return None
        tier = entry.get("oos_metrics_by_cost_tier", {}).get("base")
        return tier.get("profit_factor") if tier else None

    results = []
    for key, r in sorted(combined_by_signal.items()):
        if key == target_key:
            continue
        direction, name = key
        base = r.oos_metrics_by_cost_tier.get("base")
        high = r.oos_metrics_by_cost_tier.get("high")
        results.append(
            OtherSignalSummary(
                direction=direction,
                signal_name=name,
                combined_pf_base=base.profit_factor if base else None,
                combined_expectancy_base=base.expectancy if base else None,
                phase6_5_pf_base=_pf(p65_by_key.get(key)),
                phase7_pf_base=_pf(p7_by_key.get(key)),
                combined_high_cost_pf=high.profit_factor if high else None,
                decision_combined=r.decision.value,
            )
        )
    return results


def run_phase8_analysis(
    config: AppConfig,
    tickers: list[str],
    phase6_5_report_path: Path,
    phase7_report_path: Path,
    target_direction: str = "LONG",
    target_signal_name: str = "long_oversold_rebound",
    combined_report: WalkForwardReport | None = None,
) -> Phase8Report:
    """combined_report: an already-computed run_walk_forward(...,
    min_oos_start=None) result to reuse instead of recomputing it here.
    ALL 12 Signals' Combined OOS (incl. the dominant ~1-hour 12-signal
    Permutation Test) are produced by ONE run_walk_forward() call - Phase
    11's Research track (pipeline/run_phase11_research.py) calls this
    function once per remaining Signal and must not pay that cost 11
    times over, so it computes run_walk_forward() itself exactly once
    and passes the SAME result in here for every signal. Default None
    preserves Phase 8's original standalone behavior exactly (computes
    it internally, as every existing Phase 8 caller/test still expects).
    """
    config_check = verify_config_hash(phase6_5_report_path, phase7_report_path)
    if not config_check.matches:
        raise ConfigMismatchError(f"CONFIG_MISMATCH: {config_check}")

    if combined_report is None:
        logger.info(
            "Phase 8: computing Combined OOS (min_oos_start=None) over %d tickers", len(tickers)
        )
        combined_report = run_walk_forward(config, tickers=tickers, min_oos_start=None)
    combined_by_signal = {
        (r.direction, r.signal_name): r for r in combined_report.signal_results
    }

    target_key = (target_direction, target_signal_name)
    target_combined = combined_by_signal.get(target_key)
    if target_combined is None:
        raise RuntimeError(
            f"target signal {target_key} not found in the Combined OOS report - "
            "it never triggered over the full dataset"
        )

    backtest_summary = run_backtest(config, tickers=tickers)
    trades = backtest_summary.trades
    topix = load_ohlcv("TOPIX", config.data.processed_dir)
    regime_df = compute_market_regime(topix, config.validation.market_regime)

    signal_trades = trades[
        (trades["direction"] == target_direction) & (trades["signal_name"] == target_signal_name)
    ]

    windows = combined_report.windows
    bear_by_window = compute_regime_by_window(signal_trades, regime_df, windows, BEAR_REGIME)
    bear_by_year = compute_regime_by_year(signal_trades, regime_df, BEAR_REGIME, list(YEARS))

    bear_episodes_all = identify_regime_episodes(regime_df, BEAR_REGIME)
    bear_episode_metrics = compute_episode_metrics(signal_trades, bear_episodes_all)

    merged = signal_trades.merge(regime_df, left_on="signal_date", right_on="date", how="left")
    bear_trades = merged[merged["regime"] == BEAR_REGIME].copy()

    bear_ticker_metrics = compute_metrics_by_ticker(bear_trades)
    bear_concentration = compute_concentration(bear_trades)
    bear_top_k = compute_top_k_shares(bear_trades, TOP_K_VALUES)

    bear_cost_sensitivity: dict[str, BacktestMetrics] = {}
    for tier in config.validation.transaction_cost.tiers:
        costed = bear_trades.assign(**{"return": apply_cost(bear_trades["return"], tier)})
        bear_cost_sensitivity[tier.name] = compute_metrics(costed)

    base_tier = next(t for t in config.validation.transaction_cost.tiers if t.name == "base")
    base_returns = apply_cost(bear_trades["return"], base_tier)
    bear_bootstrap = {
        name: bootstrap_ci(base_returns.to_numpy(), name, config.validation.bootstrap)
        for name in STATISTIC_NAMES
    }

    perm_window = config.validation.permutation.forward_window
    logger.info(
        "Phase 8: building BEAR-restricted permutation population over %d tickers", len(tickers)
    )
    bear_signal_returns = _bear_signal_forward_returns(
        config, tickers, target_direction, target_signal_name, regime_df, perm_window
    )
    bear_population_returns = _bear_population_forward_returns(
        config, tickers, regime_df, perm_window
    )
    bear_permutation: PermutationResult | None = None
    if len(bear_signal_returns) > 0:
        bear_permutation = permutation_test(
            bear_signal_returns, bear_population_returns, config.validation.permutation
        )

    bear_trades_with_year = bear_trades.assign(
        year=bear_trades["signal_date"].apply(lambda d: d.year)
    )
    lopo_by_year = leave_one_period_out(bear_trades_with_year, "year")

    bear_trades_with_episode = bear_trades.assign(
        bear_episode_index=_assign_episode_index(bear_trades["signal_date"], bear_episodes_all)
    )
    lopo_by_episode = leave_one_period_out(bear_trades_with_episode, "bear_episode_index")

    logger.info("Phase 8: running Placebo negative control over %d tickers", len(tickers))
    placebo = _run_placebo(
        config, tickers, target_direction, target_signal_name, regime_df, bear_trades
    )

    target_report = Phase8SignalReport(
        direction=target_direction,
        signal_name=target_signal_name,
        combined=target_combined,
        bear_by_window=bear_by_window,
        bear_by_year=bear_by_year,
        bear_episodes=bear_episode_metrics,
        bear_ticker_metrics=bear_ticker_metrics,
        bear_concentration=bear_concentration,
        bear_top_k_shares=bear_top_k,
        bear_cost_sensitivity=bear_cost_sensitivity,
        bear_bootstrap=bear_bootstrap,
        bear_permutation=bear_permutation,
        lopo_by_year=lopo_by_year,
        lopo_by_episode=lopo_by_episode,
        placebo=placebo,
    )

    phase6_5_report = (
        json.loads(phase6_5_report_path.read_text(encoding="utf-8"))
        if phase6_5_report_path.exists()
        else {}
    )
    phase7_report = (
        json.loads(phase7_report_path.read_text(encoding="utf-8"))
        if phase7_report_path.exists()
        else {}
    )
    other_signals = _build_other_signals_summary(
        combined_by_signal, phase6_5_report, phase7_report, target_key
    )

    return Phase8Report(
        config_check=config_check,
        windows=windows,
        bear_episodes_all=bear_episodes_all,
        target=target_report,
        other_signals=other_signals,
        tickers=tickers,
        data_start=windows[0].train_start if windows else None,
        data_end=windows[-1].oos_end if windows else None,
    )
