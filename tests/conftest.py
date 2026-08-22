from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(
    n: int, seed: int = 0, base_price: float = 1000.0, ticker: str = "TEST"
) -> pd.DataFrame:
    """A random-walk OHLCV series with a fresh 0..n-1 index, used across
    the Phase 2 feature tests (no-lookahead checks need enough rows for
    every feature's warmup to clear, hence a generator rather than a
    small hand-written fixture).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    daily_returns = rng.normal(0, 0.01, size=n)
    close = base_price * np.cumprod(1 + daily_returns)
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    high_raw = close * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    low_raw = close * (1 - np.abs(rng.normal(0, 0.003, size=n)))
    high = np.maximum.reduce([high_raw, open_, close])
    low = np.minimum.reduce([low_raw, open_, close])
    volume = rng.integers(10_000, 100_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": [d.date() for d in dates],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def make_panel(n: int = 1, ticker: str = "TEST", **columns: object) -> pd.DataFrame:
    """A minimal synthetic Feature-panel-shaped DataFrame built directly
    from explicit column values (bypassing compute_feature_panel
    entirely) - used by signals/ boundary tests, where what's being
    tested is "does this Signal trigger at exactly this feature value",
    not "does Phase 2 compute that feature value correctly" (that's
    already covered by tests/test_features_*.py).
    """
    base: dict[str, object] = {
        "ticker": [ticker] * n,
        "date": list(pd.bdate_range("2024-01-01", periods=n)),
    }
    base.update(columns)
    return pd.DataFrame(base)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    return pd.DataFrame(
        {
            "ticker": ["7203"] * 10,
            "date": dates,
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [10_000 + i * 100 for i in range(10)],
        }
    )


@pytest.fixture
def sample_master_csv_path() -> str:
    return "data/reference/jpx_listed_companies.sample.csv"
