"""Phase 8 section 7: segments a Market Regime series (backtest/
market_regime.py's output) into contiguous episodes of one regime, and
computes per-episode Trade metrics. Does not change the regime
classification logic itself (BEAR/BULL/NEUTRAL thresholds are untouched -
this module only groups the already-classified dates into runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd

from backtest.metrics import BacktestMetrics, compute_metrics


@dataclass(frozen=True)
class RegimeEpisode:
    regime: str
    index: int  # 0-based, chronological order among episodes of this regime
    start_date: date_type
    end_date: date_type
    trading_days: int


def identify_regime_episodes(regime_df: pd.DataFrame, regime: str) -> list[RegimeEpisode]:
    """regime_df: date/regime columns (backtest/market_regime.py's
    output). A run of CONSECUTIVE rows (sorted by date) with
    regime==target forms one episode; any other regime value (including
    NaN, during the trailing-return warmup) breaks the run. This treats
    "consecutive" as adjacent rows in the regime series, i.e. consecutive
    TRADING days - calendar gaps (weekends/holidays) between two trading
    days both classified BEAR do not split an episode.
    """
    df = regime_df.sort_values("date").reset_index(drop=True)

    episodes: list[RegimeEpisode] = []
    current_start: date_type | None = None
    current_dates: list[date_type] = []

    def _flush() -> None:
        if current_start is not None:
            episodes.append(
                RegimeEpisode(
                    regime=regime,
                    index=len(episodes),
                    start_date=current_start,
                    end_date=current_dates[-1],
                    trading_days=len(current_dates),
                )
            )

    for row in df.itertuples(index=False):
        if row.regime == regime:
            if current_start is None:
                current_start = row.date
            current_dates.append(row.date)
        else:
            _flush()
            current_start = None
            current_dates = []
    _flush()

    return episodes


@dataclass
class EpisodeMetrics:
    episode: RegimeEpisode
    metrics: BacktestMetrics
    cumulative_return: float | None


def compute_episode_metrics(
    trades: pd.DataFrame, episodes: list[RegimeEpisode]
) -> list[EpisodeMetrics]:
    """A trade belongs to an episode if its signal_date falls within
    [episode.start_date, episode.end_date] inclusive - the same
    signal-date-based windowing pipeline/run_walk_forward.py already uses
    for TRAIN/VALIDATION/OOS membership (backtest/walk_forward.py's
    contains_oos()), applied here to regime episodes instead of WFO
    windows.
    """
    results = []
    for ep in episodes:
        subset = trades[
            (trades["signal_date"] >= ep.start_date) & (trades["signal_date"] <= ep.end_date)
        ]
        metrics = compute_metrics(subset)
        cumulative_return = float(subset["return"].sum()) if not subset.empty else None
        results.append(
            EpisodeMetrics(episode=ep, metrics=metrics, cumulative_return=cumulative_return)
        )
    return results


@dataclass
class RichEpisodeMetrics:
    """Phase 9 section 5: EpisodeMetrics plus the additional per-episode
    breakdown fields Phase 9 asks for (unique tickers, median trade
    return, max drawdown of the within-episode cumulative-return curve,
    and this episode's share of TOTAL P&L across all episodes passed in)
    - added as a NEW dataclass/function rather than changing
    EpisodeMetrics/compute_episode_metrics(), which Phase 8 already
    depends on and tests.
    """

    episode: RegimeEpisode
    metrics: BacktestMetrics
    unique_tickers: int
    median_trade_return: float | None
    max_drawdown: float | None  # <= 0; peak-to-trough on the within-episode cumulative curve
    cumulative_return: float | None
    pnl_contribution_share: float | None  # this episode's cum_return / sum across ALL episodes


def _max_drawdown(returns_in_date_order: np.ndarray) -> float | None:
    if len(returns_in_date_order) == 0:
        return None
    cumulative = np.cumsum(returns_in_date_order)
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_peak
    return float(drawdowns.min())


def compute_rich_episode_metrics(
    trades: pd.DataFrame, episodes: list[RegimeEpisode]
) -> list[RichEpisodeMetrics]:
    basic = compute_episode_metrics(trades, episodes)
    total_cumulative = sum(
        (b.cumulative_return for b in basic if b.cumulative_return is not None), start=0.0
    )

    results = []
    for ep, base in zip(episodes, basic):
        subset = trades[
            (trades["signal_date"] >= ep.start_date) & (trades["signal_date"] <= ep.end_date)
        ].sort_values("signal_date")

        unique_tickers = int(subset["ticker"].nunique())
        median_return = float(subset["return"].median()) if not subset.empty else None
        max_dd = _max_drawdown(subset["return"].to_numpy(dtype=float))
        contribution = (
            base.cumulative_return / total_cumulative
            if base.cumulative_return is not None and total_cumulative != 0
            else None
        )

        results.append(
            RichEpisodeMetrics(
                episode=ep,
                metrics=base.metrics,
                unique_tickers=unique_tickers,
                median_trade_return=median_return,
                max_drawdown=max_dd,
                cumulative_return=base.cumulative_return,
                pnl_contribution_share=contribution,
            )
        )
    return results
