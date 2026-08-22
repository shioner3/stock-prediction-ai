from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from forward_test.portfolio import (
    compute_open_positions,
    load_portfolio,
    new_portfolio,
    save_portfolio,
)


def _trade_row(
    ticker: str,
    signal_date: date,
    entry_date: date,
    entry_price: float,
    exit_date: date,
    exit_price: float,
    trade_return: float,
    signal_name: str = "long_oversold_rebound",
    direction: str = "LONG",
) -> dict:
    return {
        "ticker": ticker, "signal_name": signal_name, "direction": direction,
        "signal_date": signal_date, "entry_date": entry_date, "entry_price": entry_price,
        "exit_date": exit_date, "exit_price": exit_price, "return": trade_return,
    }


def _trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_new_portfolio_starts_at_initial_capital() -> None:
    state = new_portfolio(10_000_000.0, 0.01)
    assert state.equity == 10_000_000.0
    assert state.realized_pnl == 0.0
    assert state.closed_positions == []


def test_record_closed_trades_updates_equity() -> None:
    state = new_portfolio(1_000_000.0, 0.10)  # notional_per_trade = 100,000
    trades = _trades(
        [_trade_row(
            "7203", date(2026, 1, 5), date(2026, 1, 6), 100.0, date(2026, 1, 13), 105.0, 0.05,
        )]
    )
    newly_added = state.record_closed_trades(trades, base_cost_bps=30.0)
    assert len(newly_added) == 1
    assert state.realized_pnl == 5_000.0  # 100,000 * 0.05
    assert state.equity == 1_005_000.0


def test_record_closed_trades_is_append_only_no_duplicates() -> None:
    state = new_portfolio(1_000_000.0, 0.10)
    trades = _trades(
        [_trade_row(
            "7203", date(2026, 1, 5), date(2026, 1, 6), 100.0, date(2026, 1, 13), 105.0, 0.05,
        )]
    )
    first = state.record_closed_trades(trades, base_cost_bps=30.0)
    second = state.record_closed_trades(trades, base_cost_bps=30.0)  # same trade re-submitted
    assert len(first) == 1
    assert len(second) == 0  # already recorded, skipped
    assert len(state.closed_positions) == 1


def test_record_closed_trades_only_adds_genuinely_new_ones() -> None:
    state = new_portfolio(1_000_000.0, 0.10)
    trade_a = _trades(
        [_trade_row(
            "7203", date(2026, 1, 5), date(2026, 1, 6), 100.0, date(2026, 1, 13), 105.0, 0.05,
        )]
    )
    state.record_closed_trades(trade_a, base_cost_bps=30.0)

    trade_b = _trade_row(
        "6758", date(2026, 1, 6), date(2026, 1, 7), 200.0, date(2026, 1, 14), 190.0, -0.05,
    )
    trade_a_and_b = pd.concat([trade_a, _trades([trade_b])], ignore_index=True)

    newly_added = state.record_closed_trades(trade_a_and_b, base_cost_bps=30.0)
    assert len(newly_added) == 1
    assert newly_added[0].ticker == "6758"
    assert len(state.closed_positions) == 2


def test_negative_return_reduces_equity() -> None:
    state = new_portfolio(1_000_000.0, 0.10)
    trades = _trades(
        [_trade_row(
            "6758", date(2026, 1, 6), date(2026, 1, 7), 200.0, date(2026, 1, 14), 190.0, -0.05,
        )]
    )
    state.record_closed_trades(trades, base_cost_bps=30.0)
    assert state.equity == 995_000.0


def test_save_and_load_portfolio_roundtrip(tmp_path: Path) -> None:
    state = new_portfolio(1_000_000.0, 0.10)
    trades = _trades(
        [_trade_row(
            "7203", date(2026, 1, 5), date(2026, 1, 6), 100.0, date(2026, 1, 13), 105.0, 0.05,
        )]
    )
    state.record_closed_trades(trades, base_cost_bps=30.0)

    path = tmp_path / "portfolio.json"
    save_portfolio(state, path)
    loaded = load_portfolio(path)

    assert loaded.initial_capital == state.initial_capital
    assert loaded.equity == state.equity
    assert len(loaded.closed_positions) == 1
    assert loaded.closed_positions[0].ticker == "7203"
    assert loaded.closed_positions[0].signal_date == date(2026, 1, 5)


def test_load_portfolio_missing_file_raises() -> None:
    try:
        load_portfolio(Path("this/does/not/exist.json"))
        raise AssertionError("should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


# --- compute_open_positions (Phase 11 section 7/8) --------------------------


def _panel(dates: list[date], opens: list[float], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "open": opens, "close": closes})


def test_compute_open_positions_marks_to_latest_close() -> None:
    dates = [date(2026, 1, d) for d in range(5, 10)]
    panel = _panel(dates, opens=[100, 102, 103, 104, 105], closes=[101, 103, 104, 105, 106])
    pending = [("7203", "long_oversold_rebound", "LONG", date(2026, 1, 5))]

    result = compute_open_positions(pending, {"7203": panel}, notional_per_trade=100_000.0)
    assert len(result) == 1
    p = result[0]
    assert p.entry_date == date(2026, 1, 6)
    assert p.entry_price == 102.0
    assert p.mark_date == date(2026, 1, 9)
    assert p.mark_price == 106.0
    assert p.unrealized_return == pytest.approx(106.0 / 102.0 - 1)
    assert p.unrealized_pnl == pytest.approx(100_000.0 * (106.0 / 102.0 - 1))


def test_compute_open_positions_short_direction_return_sign() -> None:
    dates = [date(2026, 1, d) for d in range(5, 8)]
    panel = _panel(dates, opens=[100, 90, 80], closes=[95, 85, 75])
    pending = [("6758", "short_breakdown", "SHORT", date(2026, 1, 5))]

    result = compute_open_positions(pending, {"6758": panel}, notional_per_trade=100_000.0)
    assert len(result) == 1
    p = result[0]
    # Entry=Open[1/6]=90, mark=Close[1/7]=75 -> price fell, SHORT profits.
    assert p.unrealized_return == pytest.approx((90.0 - 75.0) / 90.0)
    assert p.unrealized_pnl > 0


def test_compute_open_positions_pending_entry_omitted() -> None:
    # Signal on the LAST available date - t+1 (Entry) doesn't exist yet.
    dates = [date(2026, 1, d) for d in range(5, 8)]
    panel = _panel(dates, opens=[100, 101, 102], closes=[101, 102, 103])
    pending = [("7203", "long_oversold_rebound", "LONG", date(2026, 1, 7))]

    result = compute_open_positions(pending, {"7203": panel}, notional_per_trade=100_000.0)
    assert result == []


def test_compute_open_positions_entered_today_no_mark_yet_omitted() -> None:
    # Entry day IS the last available day - no post-entry Close to mark
    # against yet.
    dates = [date(2026, 1, d) for d in range(5, 7)]
    panel = _panel(dates, opens=[100, 101], closes=[101, 102])
    pending = [("7203", "long_oversold_rebound", "LONG", date(2026, 1, 5))]

    result = compute_open_positions(pending, {"7203": panel}, notional_per_trade=100_000.0)
    assert result == []


def test_compute_open_positions_unknown_ticker_skipped() -> None:
    pending = [("9999", "long_oversold_rebound", "LONG", date(2026, 1, 5))]
    result = compute_open_positions(pending, {}, notional_per_trade=100_000.0)
    assert result == []


def test_compute_open_positions_signal_date_not_in_panel_skipped() -> None:
    dates = [date(2026, 1, d) for d in range(5, 10)]
    panel = _panel(dates, opens=[100] * 5, closes=[101] * 5)
    pending = [("7203", "long_oversold_rebound", "LONG", date(2099, 1, 1))]
    result = compute_open_positions(pending, {"7203": panel}, notional_per_trade=100_000.0)
    assert result == []


def test_compute_open_positions_empty_pending_list() -> None:
    assert compute_open_positions([], {}, notional_per_trade=100_000.0) == []
