from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.market_regime import compute_market_regime
from config.loader import MarketRegimeConfig


def _market_df(close: list[float]) -> pd.DataFrame:
    n = len(close)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "ticker": ["TOPIX"] * n, "date": [d.date() for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": [1000.0] * n,
        }
    )


def test_strong_uptrend_gives_bull() -> None:
    n = 80
    close = list(np.linspace(100.0, 130.0, n))  # +30% over the window
    df = _market_df(close)
    out = compute_market_regime(df, MarketRegimeConfig(lookback_days=60))
    assert out["regime"].iloc[-1] == "BULL"


def test_strong_downtrend_gives_bear() -> None:
    n = 80
    close = list(np.linspace(130.0, 100.0, n))
    df = _market_df(close)
    out = compute_market_regime(df, MarketRegimeConfig(lookback_days=60))
    assert out["regime"].iloc[-1] == "BEAR"


def test_flat_price_gives_neutral() -> None:
    n = 80
    df = _market_df([100.0] * n)
    out = compute_market_regime(df, MarketRegimeConfig(lookback_days=60))
    assert out["regime"].iloc[-1] == "NEUTRAL"


def test_warmup_rows_are_none() -> None:
    n = 80
    df = _market_df([100.0] * n)
    out = compute_market_regime(df, MarketRegimeConfig(lookback_days=60))
    assert out["regime"].iloc[:60].isna().all()
    assert out["regime"].iloc[60:].notna().all()


def test_threshold_boundaries() -> None:
    base = 100.0
    # Exactly at the +5% bull threshold - return_60d == 0.05 must NOT
    # count as BULL (strict > per the module's docstring formula).
    close = [base] * 20 + list(np.linspace(base, base * 1.05, 61))[1:]
    df = _market_df(close)
    out = compute_market_regime(df, MarketRegimeConfig(lookback_days=60, bull_threshold=0.05))
    assert out["regime"].iloc[-1] == "NEUTRAL"


def test_invalid_lookback_raises() -> None:
    df = _market_df([100.0] * 10)
    with pytest.raises(ValueError, match="lookback_days"):
        compute_market_regime(df, MarketRegimeConfig(lookback_days=17))


def test_date_column_passed_through() -> None:
    n = 70
    df = _market_df([100.0] * n)
    out = compute_market_regime(df, MarketRegimeConfig(lookback_days=60))
    assert (out["date"] == df["date"]).all()
