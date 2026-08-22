from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.regime_reproducibility import compute_regime_by_window, compute_regime_by_year
from backtest.walk_forward import WalkForwardWindow


def _window(index: int, oos_start: date, oos_end: date) -> WalkForwardWindow:
    return WalkForwardWindow(
        index=index,
        train_start=date(2020, 1, 1), train_end=date(2020, 6, 1),
        validation_start=date(2020, 6, 1), validation_end=date(2020, 9, 1),
        oos_start=oos_start, oos_end=oos_end, oos_truncated=False,
    )


def _regime_df(rows: list[tuple[date, str]]) -> pd.DataFrame:
    return pd.DataFrame({"date": [r[0] for r in rows], "regime": [r[1] for r in rows]})


def _trades(rows: list[tuple[str, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "signal_date": [r[1] for r in rows],
            "return": [r[2] for r in rows],
        }
    )


# --- compute_regime_by_window -------------------------------------------------


def test_regime_by_window_restricts_to_window_oos_and_regime() -> None:
    windows = [
        _window(0, date(2024, 1, 1), date(2024, 3, 31)),
        _window(1, date(2024, 4, 1), date(2024, 6, 30)),
    ]
    regime_df = _regime_df(
        [(date(2024, 1, 15), "BEAR"), (date(2024, 2, 1), "BULL"), (date(2024, 4, 15), "BEAR")]
    )
    trades = _trades(
        [
            ("A", date(2024, 1, 15), 0.05),  # window 0, BEAR
            ("B", date(2024, 2, 1), 0.02),  # window 0, BULL - excluded
            ("C", date(2024, 4, 15), -0.01),  # window 1, BEAR
        ]
    )
    results = compute_regime_by_window(trades, regime_df, windows, "BEAR")
    assert len(results) == 2
    assert results[0].window_index == 0
    assert results[0].metrics.n_trades == 1
    assert results[1].window_index == 1
    assert results[1].metrics.n_trades == 1


def test_regime_by_window_includes_windows_with_zero_matching_trades() -> None:
    windows = [_window(0, date(2024, 1, 1), date(2024, 3, 31))]
    regime_df = _regime_df([(date(2024, 1, 15), "BULL")])  # no BEAR days at all
    trades = _trades([("A", date(2024, 1, 15), 0.05)])

    results = compute_regime_by_window(trades, regime_df, windows, "BEAR")
    assert len(results) == 1  # window not dropped
    assert results[0].metrics.n_trades == 0


def test_regime_by_window_trade_outside_oos_range_excluded() -> None:
    windows = [_window(0, date(2024, 4, 1), date(2024, 6, 30))]
    regime_df = _regime_df([(date(2024, 1, 15), "BEAR")])
    trades = _trades([("A", date(2024, 1, 15), 0.05)])  # before this window's OOS

    results = compute_regime_by_window(trades, regime_df, windows, "BEAR")
    assert results[0].metrics.n_trades == 0


# --- compute_regime_by_year ----------------------------------------------------


def test_regime_by_year_groups_by_calendar_year() -> None:
    regime_df = _regime_df(
        [(date(2024, 3, 1), "BEAR"), (date(2024, 6, 1), "BEAR"), (date(2025, 2, 1), "BEAR")]
    )
    trades = _trades(
        [
            ("A", date(2024, 3, 1), 0.05),
            ("B", date(2024, 6, 1), -0.02),
            ("C", date(2025, 2, 1), 0.03),
        ]
    )
    result = compute_regime_by_year(trades, regime_df, "BEAR", [2024, 2025, 2026])
    assert result[2024].metrics is not None
    assert result[2024].metrics.n_trades == 2
    assert result[2025].metrics is not None
    assert result[2025].metrics.n_trades == 1


def test_regime_by_year_no_bear_data_gives_none_metrics() -> None:
    regime_df = _regime_df([(date(2024, 3, 1), "BULL")])  # no BEAR days anywhere
    trades = _trades([("A", date(2024, 3, 1), 0.05)])

    result = compute_regime_by_year(trades, regime_df, "BEAR", [2024])
    assert result[2024].metrics is None  # NO_BEAR_DATA, not a zero-trade result


def test_regime_by_year_regime_present_but_signal_never_triggered() -> None:
    # BEAR days exist this year, but no trade's signal_date falls on one -
    # this must be a real zero-trade BacktestMetrics, NOT None (distinct
    # from "the regime never occurred").
    regime_df = _regime_df([(date(2024, 3, 1), "BEAR")])
    trades = _trades([("A", date(2024, 5, 1), 0.05)])  # trade on a non-BEAR day

    result = compute_regime_by_year(trades, regime_df, "BEAR", [2024])
    assert result[2024].metrics is not None
    assert result[2024].metrics.n_trades == 0
