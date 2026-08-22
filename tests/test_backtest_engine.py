from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import run_backtest_for_ticker
from config.loader import BacktestConfig


def _ohlcv(n: int = 15, open_price: float = 100.0, close_price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": [open_price] * n,
            "high": [max(open_price, close_price) + 1] * n,
            "low": [min(open_price, close_price) - 1] * n,
            "close": [close_price] * n,
            "volume": [10_000.0] * n,
        }
    )


def _signal_record(ticker: str, date, signal_name: str, direction: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker, "date": date, "signal_name": signal_name,
                "direction": direction, "triggered": True, "signal_version": "v1",
            }
        ]
    )


# --- Synthetic hand-computed Entry/Exit/Return (section 20) ----------------


def test_long_trade_hand_computed_return() -> None:
    n = 15
    dates = pd.bdate_range("2024-01-01", periods=n)
    ohlcv = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": [50.0] * n,
            "high": [110.0] * n,
            "low": [40.0] * n,
            "close": [50.0] * n,
            "volume": [10_000.0] * n,
        }
    )
    signal_date = ohlcv["date"].iloc[2]
    entry_idx = 3
    exit_idx = entry_idx + 5 - 1  # hold_days=5
    ohlcv.loc[entry_idx, "open"] = 100.0
    ohlcv.loc[exit_idx, "close"] = 105.0

    signals = _signal_record("TEST", signal_date, "test_signal", "LONG")
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=True)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 105.0
    assert np.isclose(trade["return"], 0.05)
    assert trade["entry_date"] == ohlcv["date"].iloc[entry_idx]
    assert trade["exit_date"] == ohlcv["date"].iloc[exit_idx]


def test_short_trade_hand_computed_return() -> None:
    n = 15
    dates = pd.bdate_range("2024-01-01", periods=n)
    ohlcv = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": [50.0] * n,
            "high": [110.0] * n,
            "low": [40.0] * n,
            "close": [50.0] * n,
            "volume": [10_000.0] * n,
        }
    )
    signal_date = ohlcv["date"].iloc[2]
    entry_idx = 3
    exit_idx = entry_idx + 5 - 1
    ohlcv.loc[entry_idx, "open"] = 100.0
    ohlcv.loc[exit_idx, "close"] = 95.0

    signals = _signal_record("TEST", signal_date, "test_signal", "SHORT")
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=True)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 95.0
    assert np.isclose(trade["return"], 0.05)


# --- Entry cannot be built: section 13 ---------------------------------------


def test_no_trade_when_t_plus_1_data_missing() -> None:
    n = 5
    ohlcv = _ohlcv(n)
    signal_date = ohlcv["date"].iloc[n - 1]  # last row -> no t+1 at all
    signals = _signal_record("TEST", signal_date, "sig", "LONG")
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert result.trades.empty
    assert len(result.skipped) == 1
    assert "t+1" in result.skipped[0].reason


def test_no_trade_when_exit_data_missing() -> None:
    n = 5  # t+1 (index 1) exists, but exit at entry_pos+hold_days-1=5 is out of range
    ohlcv = _ohlcv(n)
    signal_date = ohlcv["date"].iloc[0]
    signals = _signal_record("TEST", signal_date, "sig", "LONG")
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert result.trades.empty
    assert len(result.skipped) == 1
    assert "hold_days" in result.skipped[0].reason or "exit" in result.skipped[0].reason


def test_no_trade_when_t_plus_1_open_is_nan() -> None:
    n = 15
    ohlcv = _ohlcv(n)
    entry_idx = 3
    ohlcv.loc[entry_idx, "open"] = np.nan
    signal_date = ohlcv["date"].iloc[entry_idx - 1]
    signals = _signal_record("TEST", signal_date, "sig", "LONG")
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert result.trades.empty
    assert "Open is NaN" in result.skipped[0].reason


def test_no_trade_when_exit_close_is_nan() -> None:
    n = 15
    ohlcv = _ohlcv(n)
    entry_idx = 3
    exit_idx = entry_idx + 5 - 1
    ohlcv.loc[exit_idx, "close"] = np.nan
    signal_date = ohlcv["date"].iloc[entry_idx - 1]
    signals = _signal_record("TEST", signal_date, "sig", "LONG")
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert result.trades.empty
    assert "Close is NaN" in result.skipped[0].reason


def test_no_trade_when_entry_price_non_positive() -> None:
    n = 15
    ohlcv = _ohlcv(n)
    entry_idx = 3
    ohlcv.loc[entry_idx, "open"] = 0.0
    signal_date = ohlcv["date"].iloc[entry_idx - 1]
    signals = _signal_record("TEST", signal_date, "sig", "LONG")
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert result.trades.empty
    assert "entry price" in result.skipped[0].reason


def test_no_trade_when_exit_price_non_positive() -> None:
    n = 15
    ohlcv = _ohlcv(n)
    entry_idx = 3
    exit_idx = entry_idx + 5 - 1
    ohlcv.loc[exit_idx, "close"] = -1.0
    signal_date = ohlcv["date"].iloc[entry_idx - 1]
    signals = _signal_record("TEST", signal_date, "sig", "LONG")
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert result.trades.empty
    assert "exit price" in result.skipped[0].reason


# --- Overlapping signal suppression (section 14) ----------------------------


def test_overlapping_same_signal_is_suppressed_by_default() -> None:
    n = 20
    ohlcv = _ohlcv(n, open_price=100.0, close_price=100.0)
    d0, d1 = ohlcv["date"].iloc[0], ohlcv["date"].iloc[1]  # 1 day apart - well within hold_days=5
    signals = pd.concat(
        [
            _signal_record("TEST", d0, "sig", "LONG"),
            _signal_record("TEST", d1, "sig", "LONG"),
        ],
        ignore_index=True,
    )
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=True)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 1
    assert any("overlapping" in s.reason for s in result.skipped)


def test_overlapping_signal_allowed_when_suppression_disabled() -> None:
    n = 20
    ohlcv = _ohlcv(n, open_price=100.0, close_price=100.0)
    d0, d1 = ohlcv["date"].iloc[0], ohlcv["date"].iloc[1]
    signals = pd.concat(
        [
            _signal_record("TEST", d0, "sig", "LONG"),
            _signal_record("TEST", d1, "sig", "LONG"),
        ],
        ignore_index=True,
    )
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=False)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 2


def test_non_overlapping_signals_both_produce_trades() -> None:
    n = 30
    ohlcv = _ohlcv(n, open_price=100.0, close_price=100.0)
    d0 = ohlcv["date"].iloc[0]
    d_later = ohlcv["date"].iloc[10]  # well past the first trade's exit (hold_days=5)
    signals = pd.concat(
        [
            _signal_record("TEST", d0, "sig", "LONG"),
            _signal_record("TEST", d_later, "sig", "LONG"),
        ],
        ignore_index=True,
    )
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=True)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 2


def test_different_signal_names_are_never_suppressed_against_each_other() -> None:
    n = 20
    ohlcv = _ohlcv(n, open_price=100.0, close_price=100.0)
    d0, d1 = ohlcv["date"].iloc[0], ohlcv["date"].iloc[1]
    signals = pd.concat(
        [
            _signal_record("TEST", d0, "sig_a", "LONG"),
            _signal_record("TEST", d1, "sig_b", "LONG"),
        ],
        ignore_index=True,
    )
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=True)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 2
