"""Rolling per-ticker beta (v3/robustness/beta.py) - verifies the
trailing-window covariance/variance estimate recovers a KNOWN synthetic
beta, and that early (warmup) rows stay NaN before enough history exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v3.robustness.beta import (
    BETA_WINDOW,
    MAX_PLAUSIBLE_BETA,
    MIN_BETA_PERIODS,
    compute_rolling_beta,
)


def _synthetic_beta_dataset(true_beta: float, n_days: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    market_return = rng.normal(0, 0.01, n_days)
    # Deterministic relationship, zero idiosyncratic noise, so the
    # recovered beta should be extremely close to true_beta.
    stock_return = true_beta * market_return
    return pd.DataFrame(
        {
            "date": dates, "ticker": "T0",
            "return_1d": stock_return, "topix_return_1d": market_return,
        }
    )


def test_recovers_known_beta_with_zero_noise() -> None:
    dataset = _synthetic_beta_dataset(true_beta=1.8)
    result = compute_rolling_beta(dataset)
    last_beta = result["beta"].iloc[-1]
    assert abs(last_beta - 1.8) < 1e-6


def test_warmup_period_is_nan_before_min_periods() -> None:
    dataset = _synthetic_beta_dataset(true_beta=1.0, n_days=MIN_BETA_PERIODS - 5)
    result = compute_rolling_beta(dataset)
    assert result["beta"].isna().all()


def test_beta_available_once_min_periods_reached() -> None:
    dataset = _synthetic_beta_dataset(true_beta=1.0, n_days=MIN_BETA_PERIODS + 5)
    result = compute_rolling_beta(dataset)
    assert result["beta"].isna().sum() == MIN_BETA_PERIODS - 1
    assert result["beta"].notna().sum() == 6


def test_near_zero_market_variance_produces_nan_not_extreme_beta() -> None:
    # Regression test for the real Full Universe bug (see beta.py's own
    # "Bug found and fixed" docstring note): TOPIX barely moves over the
    # window (tiny but nonzero variance - NOT caught by the existing
    # var==0 guard) while the stock moves normally, so the raw cov/var
    # ratio would be astronomically large (observed up to ~7.8M on real
    # data) without MAX_PLAUSIBLE_BETA bounding it to NaN.
    n_days = BETA_WINDOW + 10
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    market_return = np.array([1e-9 if i % 2 == 0 else -1e-9 for i in range(n_days)])
    stock_return = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n_days)])
    dataset = pd.DataFrame(
        {"date": dates, "ticker": "T0", "return_1d": stock_return, "topix_return_1d": market_return}
    )
    result = compute_rolling_beta(dataset)
    last_beta = result["beta"].iloc[-1]
    assert pd.isna(last_beta)
    assert (result["beta"].dropna().abs() <= MAX_PLAUSIBLE_BETA).all()


def test_multiple_tickers_computed_independently() -> None:
    a = _synthetic_beta_dataset(true_beta=2.0, n_days=BETA_WINDOW + 10, seed=1)
    b = _synthetic_beta_dataset(true_beta=0.5, n_days=BETA_WINDOW + 10, seed=1)
    b["ticker"] = "T1"
    dataset = pd.concat([a, b], ignore_index=True)
    result = compute_rolling_beta(dataset)
    beta_a = result[result["ticker"] == "T0"]["beta"].iloc[-1]
    beta_b = result[result["ticker"] == "T1"]["beta"].iloc[-1]
    assert abs(beta_a - 2.0) < 1e-6
    assert abs(beta_b - 0.5) < 1e-6
