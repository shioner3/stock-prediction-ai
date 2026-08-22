"""Phase 10 section 14: daily integrity monitoring.

- Data integrity: reuses validation/ohlcv.py::validate_ohlcv() UNCHANGED
  (missing/duplicate/abnormal OHLC/negative volume/unsorted dates), plus
  one NEW check specific to a daily Forward Test - "stale prices"
  (today's fetch didn't actually advance past the last known date, e.g.
  the Provider silently returned yesterday's data again).
- Strategy integrity: forward_test/manifest.py::
  verify_strategy_hashes_unchanged() (imported by callers, not
  duplicated here).
- Trading integrity: sanity checks on Positions already in the Paper
  Portfolio (duplicate signal key, impossible entry/exit ordering,
  non-positive prices). These should already be structurally impossible
  by construction - backtest/engine.py refuses to build such a trade,
  and forward_test/portfolio.py's record_closed_trades() is
  append-only-by-key - but re-verifying them explicitly is what spec
  section 14 asks for as a standing monitoring check, not a one-time
  assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type

import pandas as pd

from forward_test.portfolio import Position
from validation.ohlcv import ValidationIssue, validate_ohlcv


@dataclass
class DataIntegrityResult:
    ticker: str
    issues: list[ValidationIssue]
    is_stale: bool
    latest_date: date_type | None
    expected_date: date_type

    @property
    def is_clean(self) -> bool:
        return not self.issues and not self.is_stale


def check_data_integrity(
    df: pd.DataFrame, ticker: str, expected_date: date_type
) -> DataIntegrityResult:
    """expected_date: the trading day this fetch was supposed to bring
    data up to (usually "today"). A fetch that returns data, but whose
    latest row is still an OLDER date, is flagged stale rather than
    silently treated as "up to date" - the run_date must never advance
    past what the data can actually support.
    """
    if df.empty:
        return DataIntegrityResult(ticker, [], True, None, expected_date)
    report = validate_ohlcv(df, ticker)
    latest_date = df["date"].max()
    is_stale = latest_date < expected_date
    return DataIntegrityResult(ticker, report.issues, is_stale, latest_date, expected_date)


@dataclass
class TradingIntegrityResult:
    duplicate_keys: list[str] = field(default_factory=list)
    impossible_entries: list[str] = field(default_factory=list)
    impossible_exits: list[str] = field(default_factory=list)
    non_positive_prices: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.duplicate_keys or self.impossible_entries
            or self.impossible_exits or self.non_positive_prices
        )


def check_trading_integrity(positions: list[Position]) -> TradingIntegrityResult:
    seen: set[str] = set()
    result = TradingIntegrityResult()

    for p in positions:
        key = f"{p.ticker}|{p.signal_name}|{p.direction}|{p.signal_date.isoformat()}"
        if key in seen:
            result.duplicate_keys.append(key)
        seen.add(key)

        if p.entry_price <= 0:
            result.non_positive_prices.append(f"{key}: entry_price={p.entry_price}")
        if p.exit_price <= 0:
            result.non_positive_prices.append(f"{key}: exit_price={p.exit_price}")
        if p.entry_date <= p.signal_date:
            result.impossible_entries.append(
                f"{key}: entry_date={p.entry_date} <= signal_date={p.signal_date}"
            )
        if p.exit_date <= p.entry_date:
            result.impossible_exits.append(
                f"{key}: exit_date={p.exit_date} <= entry_date={p.entry_date}"
            )

    return result
