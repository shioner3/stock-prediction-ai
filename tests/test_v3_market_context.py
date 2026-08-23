from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from conftest import make_synthetic_ohlcv

from v3.features.market_context import compute_market_breadth, compute_market_context_panel


def test_market_context_panel_has_expected_columns() -> None:
    market = make_synthetic_ohlcv(200, seed=1, ticker="TOPIX")
    out = compute_market_context_panel(market)
    for col in (
        "topix_return_1d", "topix_return_20d", "topix_return_60d",
        "topix_volatility_20d", "topix_drawdown_20d",
    ):
        assert col in out.columns
    assert len(out) == len(market)


def test_market_breadth_all_positive_gives_ratio_one() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    rows = [(d, f"T{i}", 0.01) for d in dates for i in range(10)]
    panel = pd.DataFrame(rows, columns=["date", "ticker", "return_1d"])
    breadth = compute_market_breadth(panel)
    assert (breadth["advancing_ratio"] == 1.0).all()
    assert (breadth["declining_ratio"] == 0.0).all()
    assert (breadth["market_breadth"] == 1.0).all()


def test_market_breadth_mixed_signs() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    rows = []
    for d in dates:
        for i in range(20):
            ret = 0.01 if i < 12 else -0.01  # 60% advancing, 40% declining
            rows.append((d, f"T{i}", ret))
    panel = pd.DataFrame(rows, columns=["date", "ticker", "return_1d"])
    breadth = compute_market_breadth(panel)
    assert (abs(breadth["advancing_ratio"] - 0.6) < 1e-9).all()
    assert (abs(breadth["declining_ratio"] - 0.4) < 1e-9).all()
    assert (abs(breadth["market_breadth"] - 0.2) < 1e-9).all()
