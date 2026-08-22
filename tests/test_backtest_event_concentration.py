from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from backtest.event_concentration import compute_day_concentration


def _trades(rows: list[tuple[str, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "signal_date": [r[1] for r in rows],
            "return": [r[2] for r in rows],
        }
    )


def _d(offset: int) -> date:
    return date(2024, 1, 1) + timedelta(days=offset)


def test_empty_trades() -> None:
    m = compute_day_concentration(pd.DataFrame(columns=["signal_date", "return"]))
    assert m.n_days == 0
    assert m.n_trades == 0
    assert m.pnl_share_by_k[1] is None
    assert m.gini_coefficient is None


def test_single_day_is_fully_concentrated() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(0), 0.03), ("C", _d(0), -0.01)])
    m = compute_day_concentration(trades, k_values=(1, 5))
    assert m.n_days == 1
    assert m.n_trades == 3
    assert m.pnl_share_by_k[1] == 1.0
    assert m.trade_share_by_k[1] == 1.0
    assert m.pnl_share_by_k[5] == 1.0  # k larger than n_days caps at 100%


def test_evenly_spread_across_many_days_low_concentration() -> None:
    trades = _trades([(f"T{i}", _d(i), 0.01) for i in range(20)])
    m = compute_day_concentration(trades, k_values=(1, 5, 10))
    assert m.n_days == 20
    assert m.trade_share_by_k[1] == pytest.approx(1 / 20)
    assert m.pnl_share_by_k[1] == pytest.approx(1 / 20)
    assert m.trade_share_by_k[10] == pytest.approx(0.5)


def test_matches_phase8_style_concentrated_event() -> None:
    # One dominant day (like Phase 8's 2024-08 episode) among several
    # small days.
    big_day = [("T", _d(0), 0.30)] * 10
    small_days = [(f"S{i}", _d(i + 1), 0.01) for i in range(5)]
    trades = _trades(big_day + small_days)

    m = compute_day_concentration(trades, k_values=(1,))
    assert m.n_days == 6
    assert m.pnl_share_by_k[1] > 0.9  # the big day dominates P&L
    assert m.trade_share_by_k[1] == pytest.approx(10 / 15)


def test_gini_zero_when_pnl_evenly_split_across_days() -> None:
    trades = _trades([(f"T{i}", _d(i), 0.01) for i in range(10)])
    m = compute_day_concentration(trades)
    assert m.gini_coefficient == pytest.approx(0.0, abs=1e-9)


def test_gini_higher_for_concentrated_than_even_distribution() -> None:
    even_trades = _trades([(f"T{i}", _d(i), 0.01) for i in range(10)])
    concentrated_trades = _trades(
        [("BIG", _d(0), 0.09)] + [(f"T{i}", _d(i + 1), 0.001) for i in range(9)]
    )
    even_gini = compute_day_concentration(even_trades).gini_coefficient
    concentrated_gini = compute_day_concentration(concentrated_trades).gini_coefficient
    assert even_gini is not None and concentrated_gini is not None
    assert concentrated_gini > even_gini


def test_zero_total_return_gives_none_pnl_share_but_valid_trade_share() -> None:
    trades = _trades([("A", _d(0), 0.05), ("B", _d(1), -0.05)])
    m = compute_day_concentration(trades, k_values=(1,))
    assert m.pnl_share_by_k[1] is None
    assert m.trade_share_by_k[1] == 0.5
