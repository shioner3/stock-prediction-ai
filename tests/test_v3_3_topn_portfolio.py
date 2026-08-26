from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v3.validation.topn_portfolio import (
    compute_daily_top_n_tickers,
    compute_topn_portfolio_metrics,
    compute_turnover,
)


def _panel(n_days: int = 20, n_tickers: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append((d, f"T{i}", rng.uniform(0, 1), rng.normal(0.001, 0.01)))
    return pd.DataFrame(rows, columns=["date", "ticker", "prediction", "actual"])


def test_cumulative_return_matches_manual_compounding() -> None:
    panel = _panel()
    metrics = compute_topn_portfolio_metrics(panel, 5, "actual", window_days=5)
    daily_returns = np.array(
        [
            d.equal_weight_return
            for d in metrics.base.daily_returns
            if d.equal_weight_return is not None
        ]
    )
    expected = float(np.prod(1 + daily_returns) - 1)
    assert metrics.cumulative_return is not None
    assert abs(metrics.cumulative_return - expected) < 1e-9


def test_max_drawdown_is_nonpositive() -> None:
    panel = _panel()
    metrics = compute_topn_portfolio_metrics(panel, 10, "actual", window_days=5)
    assert metrics.max_drawdown is not None
    assert metrics.max_drawdown <= 0


def test_constant_positive_returns_give_zero_drawdown() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    rows = [(d, f"T{i}", float(i), 0.01) for d in dates for i in range(5)]
    panel = pd.DataFrame(rows, columns=["date", "ticker", "prediction", "actual"])
    metrics = compute_topn_portfolio_metrics(panel, 3, "actual", window_days=5)
    assert metrics.max_drawdown is not None
    assert abs(metrics.max_drawdown) < 1e-9
    assert metrics.sharpe is None  # zero std -> undefined Sharpe


def test_turnover_is_zero_when_selection_never_changes() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    rows = [(d, f"T{i}", float(4 - i), 0.01) for d in dates for i in range(5)]
    panel = pd.DataFrame(rows, columns=["date", "ticker", "prediction", "actual"])
    daily_tickers = compute_daily_top_n_tickers(panel, 3)
    turnover = compute_turnover(daily_tickers)
    assert turnover == 0.0


def test_turnover_is_one_when_selection_fully_replaced_each_day() -> None:
    daily_tickers = {
        date(2024, 1, 1): {"A", "B"},
        date(2024, 1, 2): {"C", "D"},
        date(2024, 1, 3): {"A", "B"},
    }
    turnover = compute_turnover(daily_tickers)
    assert turnover == 1.0
