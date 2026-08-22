"""Phase 10 section 6/13: Paper Portfolio.

Virtual position/equity tracking ONLY - no real orders, no real money
(spec section 13/22). Trade economics (Entry=Open[t+1], Exit=Close after
HOLD_DAYS, same "base" transaction cost tier) come VERBATIM from
already-computed Trade Records (backtest/engine.py::run_backtest_for_ticker,
unchanged) - this module only translates a percentage RETURN per trade
into a monetary P&L using one simple, documented, FIXED-notional
convention. The existing Backtest Engine has no position-sizing model of
its own (see backtest/metrics.py's docstring: "no position-sizing model
in Phase 4") - a concrete monetary translation is invented HERE, for
Paper Portfolio reporting only; it never feeds back into how Signals or
trade RETURNS are computed.

Position sizing convention: every position risks the SAME fixed notional
(initial_capital * per_trade_notional_fraction), no compounding, no cap
on the number of simultaneously open positions - the existing Backtest
Engine's only entry restriction is per-(ticker, signal, direction)
overlap suppression, not a portfolio-wide position-count cap, so none is
invented here either (spec section 6: "同時最大保有数は既存仕様を維持する").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass
class Position:
    ticker: str
    signal_name: str
    direction: str
    signal_date: date
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    trade_return: float  # from the frozen Backtest Engine, unmodified
    cost_bps: float
    notional: float
    quantity: float
    pnl: float
    status: str = "CLOSED"


@dataclass
class OpenPosition:
    """Phase 11 section 7/8: a signal that has ENTERED (Entry=Open[t+1]
    is known) but not yet reached its Exit date - marked to market using
    the latest available Close, for unrealized P&L / open_positions
    reporting. Never appears in closed_positions; this is purely
    descriptive (nothing here is "executed" - spec section 13/22).
    """

    ticker: str
    signal_name: str
    direction: str
    signal_date: date
    entry_date: date
    entry_price: float
    notional: float
    quantity: float
    mark_date: date
    mark_price: float
    unrealized_return: float
    unrealized_pnl: float
    status: str = "OPEN"


@dataclass
class PortfolioState:
    initial_capital: float
    per_trade_notional_fraction: float
    closed_positions: list[Position] = field(default_factory=list)

    @property
    def notional_per_trade(self) -> float:
        return self.initial_capital * self.per_trade_notional_fraction

    @property
    def realized_pnl(self) -> float:
        return sum(p.pnl for p in self.closed_positions)

    @property
    def equity(self) -> float:
        return self.initial_capital + self.realized_pnl

    def recorded_keys(self) -> set[tuple[str, str, str, str]]:
        return {
            (p.ticker, p.signal_name, p.direction, p.signal_date.isoformat())
            for p in self.closed_positions
        }

    def record_closed_trades(self, trades: pd.DataFrame, base_cost_bps: float) -> list[Position]:
        """trades: backtest/engine.py's Trade Record columns (ticker,
        signal_name, direction, signal_date, entry_date, exit_date,
        entry_price, exit_price, return). APPEND-ONLY: a (ticker,
        signal_name, direction, signal_date) key already present in
        closed_positions is skipped, never overwritten or recomputed
        (spec section 7/11's immutability rule) - this is what makes it
        safe to call this once per day with the FULL re-derived trade set
        each time (backtest/engine.py is a pure function of already-fixed
        Signal dates and accumulating OHLCV, so re-deriving is never
        "using hindsight", only resolving trades as their required
        future data becomes available).
        """
        existing = self.recorded_keys()
        notional = self.notional_per_trade
        newly_added: list[Position] = []

        for _, row in trades.iterrows():
            key = (
                row["ticker"], row["signal_name"], row["direction"],
                row["signal_date"].isoformat(),
            )
            if key in existing:
                continue
            entry_price = float(row["entry_price"])
            pnl = notional * float(row["return"])
            position = Position(
                ticker=row["ticker"],
                signal_name=row["signal_name"],
                direction=row["direction"],
                signal_date=row["signal_date"],
                entry_date=row["entry_date"],
                entry_price=entry_price,
                exit_date=row["exit_date"],
                exit_price=float(row["exit_price"]),
                trade_return=float(row["return"]),
                cost_bps=base_cost_bps,
                notional=notional,
                quantity=notional / entry_price if entry_price > 0 else 0.0,
                pnl=pnl,
            )
            self.closed_positions.append(position)
            newly_added.append(position)
        return newly_added


def compute_open_positions(
    pending_signals: list[tuple[str, str, str, date]],
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    notional_per_trade: float,
) -> list[OpenPosition]:
    """pending_signals: (ticker, signal_name, direction, signal_date)
    tuples for signals already in the Signal Log but NOT YET in
    closed_positions - i.e. every signal minus whatever
    record_closed_trades() has already resolved. For each one, marks the
    position to market using Entry=Open[t+1] (same rule as the frozen
    Backtest Engine) and the LATEST available Close - NOT a
    re-implementation of backtest/engine.py (which only ever returns
    fully-resolved trades and has no "still open" concept at all), just
    a read-only monetary snapshot layered on top of it for reporting.

    A signal whose Entry day (t+1) has no data yet is PENDING_ENTRY, not
    OPEN, and is silently omitted (not yet a position at all).
    """
    open_positions: list[OpenPosition] = []
    for ticker, signal_name, direction, signal_date in pending_signals:
        df = ohlcv_by_ticker.get(ticker)
        if df is None or df.empty:
            continue
        dates = df["date"].tolist()
        try:
            pos = dates.index(signal_date)
        except ValueError:
            continue
        entry_pos = pos + 1
        if entry_pos >= len(dates):
            continue  # PENDING_ENTRY - t+1 not reached yet
        entry_date = dates[entry_pos]
        entry_price = float(df.loc[df["date"] == entry_date, "open"].iloc[0])
        if entry_price <= 0 or pd.isna(entry_price):
            continue

        mark_date = dates[-1]
        if mark_date <= entry_date:
            continue  # entered today - no post-entry mark-to-market day yet
        mark_price = float(df.loc[df["date"] == mark_date, "close"].iloc[0])
        if mark_price <= 0 or pd.isna(mark_price):
            continue

        if direction == "LONG":
            unrealized_return = mark_price / entry_price - 1
        else:
            unrealized_return = (entry_price - mark_price) / entry_price
        quantity = notional_per_trade / entry_price

        open_positions.append(
            OpenPosition(
                ticker=ticker,
                signal_name=signal_name,
                direction=direction,
                signal_date=signal_date,
                entry_date=entry_date,
                entry_price=entry_price,
                notional=notional_per_trade,
                quantity=quantity,
                mark_date=mark_date,
                mark_price=mark_price,
                unrealized_return=unrealized_return,
                unrealized_pnl=notional_per_trade * unrealized_return,
            )
        )
    return open_positions


def _json_default(obj: object) -> object:
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_portfolio(state: PortfolioState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(state), default=_json_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_portfolio(path: Path) -> PortfolioState:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    positions = [
        Position(
            ticker=p["ticker"],
            signal_name=p["signal_name"],
            direction=p["direction"],
            signal_date=date.fromisoformat(p["signal_date"]),
            entry_date=date.fromisoformat(p["entry_date"]),
            entry_price=p["entry_price"],
            exit_date=date.fromisoformat(p["exit_date"]),
            exit_price=p["exit_price"],
            trade_return=p["trade_return"],
            cost_bps=p["cost_bps"],
            notional=p["notional"],
            quantity=p.get(
                "quantity", p["notional"] / p["entry_price"] if p["entry_price"] else 0.0
            ),
            pnl=p["pnl"],
            status=p.get("status", "CLOSED"),
        )
        for p in raw["closed_positions"]
    ]
    return PortfolioState(
        initial_capital=raw["initial_capital"],
        per_trade_notional_fraction=raw["per_trade_notional_fraction"],
        closed_positions=positions,
    )


def new_portfolio(initial_capital: float, per_trade_notional_fraction: float) -> PortfolioState:
    return PortfolioState(
        initial_capital=initial_capital, per_trade_notional_fraction=per_trade_notional_fraction
    )
