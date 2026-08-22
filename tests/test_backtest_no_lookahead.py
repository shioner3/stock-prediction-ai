"""No-lookahead verification for the Backtest layer (Phase 4's section 19):

    A. Signal(t) is unaffected by changes to t+1-and-later data.
    B. Trade Return may change when t+1 Open changes, but the Signal
       decision itself (which dates triggered) must not.
    C. Entry Price is always exactly Open[t+1].
    D. Close[t] is never used as an Entry Price, even when it's a
       deliberately extreme, hard-to-miss value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from backtest.engine import run_backtest_for_ticker
from config.loader import BacktestConfig, SignalsConfig
from features.pipeline import compute_feature_panel
from signals.pipeline import compute_signal_panel

# --- Test A: Signal(t) unaffected by t+1+ changes ---------------------------


def test_signal_at_t_unaffected_by_future_data_changes() -> None:
    n = 300
    base = make_synthetic_ohlcv(n, seed=60)
    panel_base = compute_feature_panel(base)
    signals_base = compute_signal_panel(panel_base, SignalsConfig())

    rng = np.random.default_rng(61)
    t = 200
    perturbed = base.copy()
    future_mask = perturbed.index > t
    n_future = int(future_mask.sum())
    for col in ("open", "high", "low", "close"):
        perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] * rng.uniform(
            0.5, 1.5, size=n_future
        )
    perturbed.loc[future_mask, "volume"] = perturbed.loc[future_mask, "volume"] * rng.uniform(
        0.5, 3.0, size=n_future
    )

    panel_perturbed = compute_feature_panel(perturbed)
    signals_perturbed = compute_signal_panel(panel_perturbed, SignalsConfig())

    signal_cols = [c for c in signals_base.columns if c not in ("ticker", "date")]
    row_base = signals_base.loc[t, signal_cols]
    row_perturbed = signals_perturbed.loc[t, signal_cols]
    pd.testing.assert_series_equal(row_base, row_perturbed, check_names=False)


# --- Test B: changing t+1 Open changes Trade Return, not the Signal --------


def _ohlcv_with_entry_open(n: int, entry_idx: int, entry_open: float) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": [50.0] * n,
            "high": [120.0] * n,
            "low": [10.0] * n,
            "close": [50.0] * n,
            "volume": [10_000.0] * n,
        }
    )
    df.loc[entry_idx, "open"] = entry_open
    df.loc[entry_idx + 4, "close"] = 110.0  # hold_days=5 -> exit at entry_idx+4
    return df


def test_changing_entry_open_changes_return_but_not_signal_decision() -> None:
    n = 15
    entry_idx = 3
    signal_date_idx = entry_idx - 1
    ohlcv_a = _ohlcv_with_entry_open(n, entry_idx, entry_open=100.0)
    ohlcv_b = _ohlcv_with_entry_open(n, entry_idx, entry_open=50.0)

    signal_date = ohlcv_a["date"].iloc[signal_date_idx]
    signals = pd.DataFrame(
        [
            {
                "ticker": "TEST", "date": signal_date, "signal_name": "sig",
                "direction": "LONG", "triggered": True, "signal_version": "v1",
            }
        ]
    )
    config = BacktestConfig(hold_days=5)

    result_a = run_backtest_for_ticker(signals, ohlcv_a, config)
    result_b = run_backtest_for_ticker(signals, ohlcv_b, config)

    # Same Signal input -> same number of trades and the same entry/exit
    # DATES either way (the Signal decision, i.e. which dates trade, is
    # untouched by the Entry price).
    assert len(result_a.trades) == len(result_b.trades) == 1
    assert result_a.trades.iloc[0]["entry_date"] == result_b.trades.iloc[0]["entry_date"]
    assert result_a.trades.iloc[0]["signal_date"] == result_b.trades.iloc[0]["signal_date"]

    # But the Trade Return differs, because Entry Price differs.
    assert result_a.trades.iloc[0]["entry_price"] != result_b.trades.iloc[0]["entry_price"]
    assert result_a.trades.iloc[0]["return"] != result_b.trades.iloc[0]["return"]


# --- Test C: Entry Price is always exactly Open[t+1] -------------------------


def test_entry_price_always_equals_t_plus_1_open() -> None:
    n = 200
    ohlcv = make_synthetic_ohlcv(n, seed=62)
    dates = ohlcv["date"].tolist()
    # Trigger a synthetic signal on every date from index 5 to n-10.
    signal_dates = dates[5 : n - 10]
    signals = pd.DataFrame(
        {
            "ticker": ["TEST"] * len(signal_dates),
            "date": signal_dates,
            "signal_name": ["sig"] * len(signal_dates),
            "direction": ["LONG"] * len(signal_dates),
            "triggered": [True] * len(signal_dates),
            "signal_version": ["v1"] * len(signal_dates),
        }
    )
    config = BacktestConfig(hold_days=5, suppress_overlapping_signals=False)

    result = run_backtest_for_ticker(signals, ohlcv, config)
    ohlcv_by_date = ohlcv.set_index("date")

    assert len(result.trades) > 0
    for _, trade in result.trades.iterrows():
        expected_open = ohlcv_by_date.loc[trade["entry_date"], "open"]
        assert trade["entry_price"] == expected_open


# --- Test D: Close[t] is never used as an Entry Price -------------------------


def test_signal_date_close_never_used_as_entry_price() -> None:
    n = 15
    dates = pd.bdate_range("2024-01-01", periods=n)
    ohlcv = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": [50.0] * n,
            "high": [999_999.0] * n,
            "low": [1.0] * n,
            "close": [50.0] * n,
            "volume": [10_000.0] * n,
        }
    )
    signal_idx = 3
    entry_idx = signal_idx + 1
    ohlcv.loc[signal_idx, "close"] = 999_999.0  # an unmistakable, wrong Entry Price if used
    ohlcv.loc[entry_idx, "open"] = 42.0  # the correct Entry Price

    signal_date = ohlcv["date"].iloc[signal_idx]
    signals = pd.DataFrame(
        [
            {
                "ticker": "TEST", "date": signal_date, "signal_name": "sig",
                "direction": "LONG", "triggered": True, "signal_version": "v1",
            }
        ]
    )
    config = BacktestConfig(hold_days=5)

    result = run_backtest_for_ticker(signals, ohlcv, config)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["entry_price"] == 42.0
    assert result.trades.iloc[0]["entry_price"] != 999_999.0
