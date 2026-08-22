from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ensemble.portfolio_sim import (
    INITIAL_CAPITAL,
    PER_TRADE_NOTIONAL_FRACTION,
    compute_equity_curve_metrics,
    dedupe_trades_by_ticker_date_direction,
    select_top_n_candidates,
)


def _trade_row(
    ticker: str, signal_date: date, entry_date: date, exit_date: date, ret: float,
    direction: str = "LONG", signal_name: str = "long_pullback",
) -> dict:
    return {
        "ticker": ticker, "signal_name": signal_name, "direction": direction,
        "signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date,
        "entry_price": 100.0, "exit_price": 100.0 * (1 + ret), "return": ret,
    }


# --- dedupe -----------------------------------------------------------------


def test_dedupe_keeps_one_row_per_ticker_date_direction() -> None:
    d = date(2026, 1, 5)
    e, x = date(2026, 1, 6), date(2026, 1, 13)
    trades = pd.DataFrame(
        [
            _trade_row("7203", d, e, x, 0.05, signal_name="long_pullback"),
            _trade_row("7203", d, e, x, 0.05, signal_name="long_ma_rebound"),
        ]
    )
    out = dedupe_trades_by_ticker_date_direction(trades)
    assert len(out) == 1


def test_dedupe_keeps_distinct_tickers_dates_directions() -> None:
    d = date(2026, 1, 5)
    e, x = date(2026, 1, 6), date(2026, 1, 13)
    trades = pd.DataFrame(
        [
            _trade_row("7203", d, e, x, 0.05, direction="LONG"),
            _trade_row("7203", d, e, x, -0.03, direction="SHORT", signal_name="short_pullback"),
            _trade_row("6758", d, e, x, 0.02, direction="LONG"),
        ]
    )
    out = dedupe_trades_by_ticker_date_direction(trades)
    assert len(out) == 3


# --- select_top_n_candidates --------------------------------------------------


def test_select_top_n_candidates_ranks_by_count_then_score() -> None:
    d = date(2026, 1, 5)
    df = pd.DataFrame(
        [
            {"ticker": "A", "date": d, "dominant_direction": "LONG", "total_signal_count": 3},
            {"ticker": "B", "date": d, "dominant_direction": "LONG", "total_signal_count": 1},
            {"ticker": "C", "date": d, "dominant_direction": "LONG", "total_signal_count": 2},
        ]
    )
    scores = {("A", d): 50.0, ("B", d): 90.0, ("C", d): 10.0}
    out = select_top_n_candidates(df, scores, top_n=2)
    assert list(out["ticker"]) == ["A", "C"]  # count desc beats score


def test_select_top_n_candidates_excludes_neutral() -> None:
    d = date(2026, 1, 5)
    df = pd.DataFrame(
        [
            {"ticker": "A", "date": d, "dominant_direction": "NEUTRAL", "total_signal_count": 5},
            {"ticker": "B", "date": d, "dominant_direction": "LONG", "total_signal_count": 1},
        ]
    )
    out = select_top_n_candidates(df, {("B", d): 10.0}, top_n=5)
    assert list(out["ticker"]) == ["B"]


def test_select_top_n_candidates_caps_per_day() -> None:
    d = date(2026, 1, 5)
    rows = [
        {"ticker": t, "date": d, "dominant_direction": "LONG", "total_signal_count": 1}
        for t in "ABCDEFG"
    ]
    df = pd.DataFrame(rows)
    scores = {(t, d): float(i) for i, t in enumerate("ABCDEFG")}
    out = select_top_n_candidates(df, scores, top_n=5)
    assert len(out) == 5


# --- compute_equity_curve_metrics ---------------------------------------------


def test_equity_curve_metrics_empty() -> None:
    m = compute_equity_curve_metrics(pd.DataFrame())
    assert m.n_trades == 0
    assert m.total_return is None
    assert m.cagr is None


def test_equity_curve_metrics_single_winning_trade() -> None:
    trades = pd.DataFrame(
        [_trade_row("7203", date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 13), 0.05)]
    )
    m = compute_equity_curve_metrics(trades)
    assert m.n_trades == 1
    notional = INITIAL_CAPITAL * PER_TRADE_NOTIONAL_FRACTION
    expected_return = notional * 0.05 / INITIAL_CAPITAL
    assert m.total_return is not None
    assert np.isclose(m.total_return, expected_return)
    assert m.win_rate == 1.0
    assert m.max_drawdown == 0.0  # equity only ever goes up


def test_equity_curve_metrics_drawdown_detected() -> None:
    trades = pd.DataFrame(
        [
            _trade_row("A", date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 13), 0.10),
            _trade_row("B", date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 20), -0.20),
        ]
    )
    m = compute_equity_curve_metrics(trades)
    assert m.max_drawdown is not None and m.max_drawdown < 0
    assert m.n_trades == 2


def test_equity_curve_metrics_aggregates_same_day_exits() -> None:
    # Two trades exiting the SAME day should be summed into one equity
    # step, not treated as two separate curve points.
    exit_d = date(2026, 1, 13)
    trades = pd.DataFrame(
        [
            _trade_row("A", date(2026, 1, 5), date(2026, 1, 6), exit_d, 0.05),
            _trade_row("B", date(2026, 1, 5), date(2026, 1, 6), exit_d, 0.03),
        ]
    )
    m = compute_equity_curve_metrics(trades)
    assert m.n_trades == 2
    assert m.start_date == exit_d
    assert m.end_date == exit_d
