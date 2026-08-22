"""Phase 11 Track B: independent verification of the 11 Signals not yet
covered by Phase 6.5-9 (everything except long_oversold_rebound, which
stays frozen as Strategy Version 1's Forward Test target - see
pipeline/run_forward_test.py). This is VERIFICATION, not "improvement":
no Signal condition, Score weight, or Decision threshold is read from or
written by this module.

Every statistical primitive reused here is ALREADY signal-agnostic
(parameterized by direction/signal_name) and unmodified:
- pipeline.run_walk_forward.run_walk_forward() - Combined-OOS Bootstrap/
  Permutation/FDR/cost-tier/regime/window/concentration/Decision for
  ALL 12 signals AT ONCE. This is the ~1-hour-dominated step (the
  12-signal Combined Permutation Test) and is called EXACTLY ONCE here,
  never per-signal - see pipeline/run_phase8_analysis.py's
  `combined_report` injection parameter, added specifically so this
  module can reuse that one result across all 11 signals instead of
  recomputing it 11 times (~11 hours) for no additional information (
  run_walk_forward() already evaluates every triggered signal in the
  trades set in one pass).
- pipeline.run_phase8_analysis.run_phase8_analysis() - BEAR-regime
  window/year/episode/concentration/cost-sensitivity/bootstrap/
  BEAR-restricted-permutation/Leave-One-Period-Out/Placebo, called once
  per signal with the shared combined_report (fast: no WFO recompute).
- pipeline.run_phase9_analysis.run_phase9_analysis() - Day Cluster/Block
  Bootstrap, Timing Placebo sweep, sector/liquidity breakdown, cost
  stress by scope, forward-holding-period profile, scenarios A/B/C/E/F.
  Never calls run_walk_forward() itself (see that module's own
  docstring), so calling it once per signal only repeats its own cheap
  ~45s run_backtest() call, not the expensive WFO/permutation step.
- backtest.event_concentration.compute_day_concentration() - called
  directly here on each signal's Combined-OOS trades (not BEAR-
  restricted) to complete the "event concentration" picture Phase 9 only
  computed for the BEAR subset.

Timing Placebo offsets are OVERRIDDEN from config/phase9_settings.yaml's
persisted default (which lacks 0/+1/+3) via a Phase9Config instance built
in-memory for this run only - config/phase9_settings.yaml itself, and
therefore its own hash/history, is left untouched (same config-isolation
pattern Phase 9 established: diagnostic-analysis parameters never touch
config_hash - see config/loader.py's Phase9Config docstring).

Dependency direction (spec section 19, mirroring Phase 9's own rule):
this module IMPORTS FROM signals/backtest/targets, and must never be
imported BY them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backtest.decision import Decision
from backtest.event_concentration import DayConcentrationMetrics, compute_day_concentration
from backtest.multiple_testing import FDRResult
from config.loader import AppConfig, Phase9Config, TimingPlaceboConfig
from pipeline.run_backtest import run_backtest
from pipeline.run_phase8_analysis import (
    ConfigCheckResult,
    ConfigMismatchError,
    Phase8Report,
    run_phase8_analysis,
    verify_config_hash,
)
from pipeline.run_phase9_analysis import Phase9Report, run_phase9_analysis
from pipeline.run_walk_forward import SignalWalkForwardResult, run_walk_forward

logger = logging.getLogger(__name__)

FROZEN_STRATEGY_SIGNAL: tuple[str, str] = ("LONG", "long_oversold_rebound")

# The 11 Signals NOT yet independently verified at Phase 6.5-9's rigor -
# every Signal registered in signals/long/ and signals/short/ except the
# one already frozen as Strategy Version 1's Forward Test target.
REMAINING_SIGNALS: list[tuple[str, str]] = [
    ("LONG", "long_breakout"),
    ("LONG", "long_ma_rebound"),
    ("LONG", "long_momentum_continuation"),
    ("LONG", "long_pullback"),
    ("LONG", "long_volume_breakout"),
    ("SHORT", "short_breakdown"),
    ("SHORT", "short_ma_rejection"),
    ("SHORT", "short_momentum_continuation"),
    ("SHORT", "short_overbought_reversal"),
    ("SHORT", "short_pullback"),
    ("SHORT", "short_volume_breakdown"),
]

# Phase 11 spec section 19's required Timing Placebo offset set - a
# superset of Phase 9's own default ([-15,-10,-5,-3,-1,5,10]) that adds
# 0/+1/+3 for finer resolution near the real signal date. Fixed here,
# before running any Phase 11 analysis, exactly like Phase 9's own
# offsets were fixed before its run (spec section 10/11: never tuned to
# the result).
RESEARCH_TIMING_PLACEBO_OFFSETS: list[int] = [-15, -10, -5, -3, -1, 0, 1, 3, 5, 10]


def research_phase9_config() -> Phase9Config:
    return Phase9Config(timing_placebo=TimingPlaceboConfig(offsets=RESEARCH_TIMING_PLACEBO_OFFSETS))


@dataclass
class Phase11SignalReport:
    direction: str
    signal_name: str
    combined: SignalWalkForwardResult | None
    decision: Decision | None
    fdr: FDRResult | None
    combined_day_concentration: DayConcentrationMetrics | None
    phase8: Phase8Report | None
    phase9: Phase9Report | None
    note: str | None = None


@dataclass
class Phase11ResearchReport:
    config_check: ConfigCheckResult
    tickers: list[str]
    windows_evaluated: int
    signals: list[Phase11SignalReport]


def run_phase11_research(
    config: AppConfig,
    tickers: list[str],
    phase6_5_report_path: Path,
    phase7_report_path: Path,
    jpx_master_path: Path,
    target_signals: list[tuple[str, str]] | None = None,
) -> Phase11ResearchReport:
    config_check = verify_config_hash(phase6_5_report_path, phase7_report_path)
    if not config_check.matches:
        raise ConfigMismatchError(f"CONFIG_MISMATCH: {config_check}")

    if target_signals is None:
        target_signals = REMAINING_SIGNALS

    logger.info(
        "Phase 11 Research: computing Combined OOS (all signals, min_oos_start=None) "
        "over %d tickers - this is the expensive step, run exactly once", len(tickers),
    )
    combined_report = run_walk_forward(config, tickers=tickers, min_oos_start=None)
    combined_by_signal = {
        (r.direction, r.signal_name): r for r in combined_report.signal_results
    }

    logger.info(
        "Phase 11 Research: computing Combined backtest trades over %d tickers", len(tickers)
    )
    backtest_summary = run_backtest(config, tickers=tickers)
    trades = backtest_summary.trades

    phase9_config = research_phase9_config()
    signal_reports: list[Phase11SignalReport] = []

    for direction, signal_name in target_signals:
        key = (direction, signal_name)
        combined = combined_by_signal.get(key)
        if combined is None:
            logger.warning(
                "Phase 11 Research: %s:%s never triggered in the Combined OOS trades - "
                "recorded with INSUFFICIENT data, no Phase 8/9 sub-analysis run",
                direction, signal_name,
            )
            signal_reports.append(
                Phase11SignalReport(
                    direction=direction, signal_name=signal_name,
                    combined=None, decision=None, fdr=None,
                    combined_day_concentration=None, phase8=None, phase9=None,
                    note="signal never triggered in the Combined OOS trades",
                )
            )
            continue

        logger.info(
            "Phase 11 Research: %s:%s - running Phase 8 sub-analysis", direction, signal_name
        )
        phase8 = run_phase8_analysis(
            config, tickers, phase6_5_report_path, phase7_report_path,
            target_direction=direction, target_signal_name=signal_name,
            combined_report=combined_report,
        )
        logger.info(
            "Phase 11 Research: %s:%s - running Phase 9 sub-analysis", direction, signal_name
        )
        phase9 = run_phase9_analysis(
            config, phase9_config, tickers, phase6_5_report_path, phase7_report_path,
            jpx_master_path, target_direction=direction, target_signal_name=signal_name,
        )

        signal_trades = trades[
            (trades["direction"] == direction) & (trades["signal_name"] == signal_name)
        ]
        day_concentration = compute_day_concentration(signal_trades)

        fdr = combined_report.fdr_results.get(f"{direction}:{signal_name}")

        signal_reports.append(
            Phase11SignalReport(
                direction=direction,
                signal_name=signal_name,
                combined=combined,
                decision=combined.decision,
                fdr=fdr,
                combined_day_concentration=day_concentration,
                phase8=phase8,
                phase9=phase9,
            )
        )

    return Phase11ResearchReport(
        config_check=config_check,
        tickers=tickers,
        windows_evaluated=len(combined_report.windows),
        signals=signal_reports,
    )
