from __future__ import annotations

from datetime import date

import pandas as pd

from forward_test.integrity import check_data_integrity, check_trading_integrity
from forward_test.portfolio import Position


def _ohlcv(rows: list[tuple[date, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    ).assign(ticker="T")


def _position(
    ticker: str = "7203",
    signal_date: date = date(2026, 1, 5),
    entry_date: date = date(2026, 1, 6),
    entry_price: float = 100.0,
    exit_date: date = date(2026, 1, 13),
    exit_price: float = 105.0,
) -> Position:
    return Position(
        ticker=ticker, signal_name="long_oversold_rebound", direction="LONG",
        signal_date=signal_date, entry_date=entry_date, entry_price=entry_price,
        exit_date=exit_date, exit_price=exit_price, trade_return=0.05,
        cost_bps=30.0, notional=100_000.0, quantity=1_000.0, pnl=5_000.0,
    )


# --- check_data_integrity -------------------------------------------------------


def test_data_integrity_clean_up_to_date() -> None:
    df = _ohlcv([(date(2026, 1, 5), 100, 101, 99, 100.5, 10_000)])
    result = check_data_integrity(df, "T", expected_date=date(2026, 1, 5))
    assert result.is_clean
    assert result.is_stale is False
    assert result.issues == []


def test_data_integrity_empty_dataframe_is_stale() -> None:
    result = check_data_integrity(pd.DataFrame(), "T", expected_date=date(2026, 1, 5))
    assert result.is_stale is True
    assert result.latest_date is None
    assert not result.is_clean


def test_data_integrity_stale_when_latest_date_behind_expected() -> None:
    df = _ohlcv([(date(2026, 1, 4), 100, 101, 99, 100.5, 10_000)])
    result = check_data_integrity(df, "T", expected_date=date(2026, 1, 5))
    assert result.is_stale is True
    assert not result.is_clean


def test_data_integrity_flags_invalid_ohlc() -> None:
    df = _ohlcv([(date(2026, 1, 5), 100, 50, 99, 100.5, 10_000)])  # high < low
    result = check_data_integrity(df, "T", expected_date=date(2026, 1, 5))
    assert not result.is_clean
    assert any(i.rule == "high_below_low" for i in result.issues)


# --- check_trading_integrity ------------------------------------------------------


def test_trading_integrity_clean_position() -> None:
    result = check_trading_integrity([_position()])
    assert result.is_clean


def test_trading_integrity_detects_duplicate_key() -> None:
    p = _position()
    result = check_trading_integrity([p, p])
    assert not result.is_clean
    assert len(result.duplicate_keys) == 1


def test_trading_integrity_detects_non_positive_prices() -> None:
    bad_entry = _position(entry_price=0.0)
    bad_exit = _position(ticker="6758", exit_price=-1.0)
    result = check_trading_integrity([bad_entry, bad_exit])
    assert not result.is_clean
    assert len(result.non_positive_prices) == 2


def test_trading_integrity_detects_exit_before_entry() -> None:
    bad = _position(entry_date=date(2026, 1, 10), exit_date=date(2026, 1, 5))
    result = check_trading_integrity([bad])
    assert not result.is_clean
    assert len(result.impossible_exits) == 1


def test_trading_integrity_detects_entry_before_signal() -> None:
    bad = _position(signal_date=date(2026, 1, 10), entry_date=date(2026, 1, 5))
    result = check_trading_integrity([bad])
    assert not result.is_clean
    assert len(result.impossible_entries) == 1


def test_trading_integrity_empty_list_is_clean() -> None:
    result = check_trading_integrity([])
    assert result.is_clean
