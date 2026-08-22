"""Backtest Engine: turns SignalRecord rows into Trade Records under one
fixed, non-negotiable rule:

    Signal date = t
    Entry       = Open[t+1]
    Exit        = Close[t + 1 + hold_days - 1]   (fixed holding period)

    LONG  return = Exit/Entry - 1
    SHORT return = (Entry-Exit)/Entry            (see the SHORT branch
                                                    below for why this,
                                                    not "Entry/Exit - 1")

Entry is NEVER Close[t] - see tests/test_backtest_no_lookahead.py's
Test C/D for the automated check.

Dependency direction: this module takes an already-computed SignalRecord
DataFrame (signals/pipeline.py's output) plus that ticker's OHLCV/Feature
panel - it never imports or calls a signals/long/*.py or signals/short/*.py
compute function directly. That is what lets the same Signal output be
evaluated under several different Backtest configurations (different
hold_days, different overlap rules) without recomputing Signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.loader import BacktestConfig

logger = logging.getLogger(__name__)

TRADE_COLUMNS = [
    "ticker", "signal_name", "direction", "signal_date",
    "entry_date", "exit_date", "entry_price", "exit_price", "return",
]


@dataclass
class SkippedSignal:
    """A SignalRecord row that could not become a Trade, and why."""

    ticker: str
    signal_name: str
    direction: str
    signal_date: Any
    reason: str


@dataclass
class BacktestRunResult:
    trades: pd.DataFrame
    skipped: list[SkippedSignal] = field(default_factory=list)


def _entry_exit_prices(
    ohlcv_by_date: pd.DataFrame, signal_date: Any, hold_days: int
) -> tuple[Any, Any, float | None, float | None, str | None]:
    """Returns (entry_date, exit_date, entry_price, exit_price, skip_reason).
    skip_reason is None exactly when a valid trade can be built.
    """
    dates = ohlcv_by_date.index

    try:
        pos = dates.get_loc(signal_date)
    except KeyError:
        return None, None, None, None, "signal_date not found in OHLCV"

    entry_pos = pos + 1
    if entry_pos >= len(dates):
        return None, None, None, None, "no t+1 data available"

    exit_pos = entry_pos + hold_days - 1
    if exit_pos >= len(dates):
        return None, None, None, None, "not enough data to reach exit (hold_days)"

    entry_date = dates[entry_pos]
    exit_date = dates[exit_pos]
    entry_price = ohlcv_by_date.loc[entry_date, "open"]
    exit_price = ohlcv_by_date.loc[exit_date, "close"]

    if pd.isna(entry_price):
        return entry_date, exit_date, None, None, "t+1 Open is NaN"
    if pd.isna(exit_price):
        return entry_date, exit_date, float(entry_price), None, "exit Close is NaN"
    if entry_price <= 0:
        return entry_date, exit_date, float(entry_price), float(exit_price), "entry price <= 0"
    if exit_price <= 0:
        return entry_date, exit_date, float(entry_price), float(exit_price), "exit price <= 0"

    return entry_date, exit_date, float(entry_price), float(exit_price), None


def run_backtest_for_ticker(
    signal_records: pd.DataFrame, ohlcv: pd.DataFrame, config: BacktestConfig
) -> BacktestRunResult:
    """signal_records: one ticker's rows in signals/pipeline.py's
    SignalRecord shape (ticker, date, signal_name, direction,
    triggered=True, signal_version). ohlcv: the SAME ticker's OHLCV
    history - any DataFrame with date/open/close columns works, so the
    Feature panel can be passed directly.
    """
    ohlcv_by_date = ohlcv.set_index("date").sort_index()
    ohlcv_by_date = ohlcv_by_date[~ohlcv_by_date.index.duplicated(keep="last")]

    trades: list[dict[str, Any]] = []
    skipped: list[SkippedSignal] = []
    # (ticker, signal_name, direction) -> exit_date of that combo's
    # currently-open trade, used to suppress overlapping signals.
    open_until: dict[tuple[str, str, str], Any] = {}

    for row in signal_records.sort_values("date").itertuples(index=False):
        key = (row.ticker, row.signal_name, row.direction)

        if config.suppress_overlapping_signals:
            still_open_exit = open_until.get(key)
            if still_open_exit is not None and row.date <= still_open_exit:
                skipped.append(
                    SkippedSignal(
                        row.ticker, row.signal_name, row.direction, row.date,
                        "overlapping signal - prior trade not yet exited",
                    )
                )
                continue

        entry_date, exit_date, entry_price, exit_price, reason = _entry_exit_prices(
            ohlcv_by_date, row.date, config.hold_days
        )
        if reason is not None:
            skipped.append(
                SkippedSignal(row.ticker, row.signal_name, row.direction, row.date, reason)
            )
            continue

        assert entry_price is not None and exit_price is not None
        if row.direction == "LONG":
            trade_return = exit_price / entry_price - 1
        else:
            # Standard short-return convention: profit relative to the
            # capital at risk at entry, (Entry-Exit)/Entry - e.g.
            # Entry=100, Exit=95 -> +5%. This matches the worked example
            # in the Phase 4 spec exactly; a literal "Entry/Exit - 1"
            # reading of that spec's formula section does NOT (it gives
            # +5.26% for the same numbers) - see README's Phase 4 notes.
            trade_return = (entry_price - exit_price) / entry_price

        trades.append(
            {
                "ticker": row.ticker,
                "signal_name": row.signal_name,
                "direction": row.direction,
                "signal_date": row.date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": trade_return,
            }
        )
        if config.suppress_overlapping_signals:
            open_until[key] = exit_date

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestRunResult(trades=trades_df, skipped=skipped)
