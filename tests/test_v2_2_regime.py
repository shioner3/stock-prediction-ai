from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from config.loader import MarketRegimeConfig
from v2.validation.regime import REGIMES, analyze_by_regime, build_regime_series


def test_build_regime_series_reuses_v1_classification() -> None:
    market = make_synthetic_ohlcv(200, seed=1, ticker="TOPIX")
    regime_df = build_regime_series(market, MarketRegimeConfig())
    assert list(regime_df.columns) == ["date", "regime"]
    assert set(regime_df["regime"].dropna().unique()) <= {"BULL", "NEUTRAL", "BEAR"}


def test_analyze_by_regime_returns_all_three_regimes() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    regime_df = pd.DataFrame(
        {"date": dates, "regime": (["BULL"] * 10 + ["NEUTRAL"] * 10 + ["BEAR"] * 10)}
    )
    rng = np.random.default_rng(1)
    rows = []
    for d in dates:
        for i in range(20):
            rows.append((d, f"T{i}", float(i) / 20, rng.normal(0, 0.02)))
    scored = pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "total_score": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )
    scored["score_bucket"] = pd.qcut(
        scored["total_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    results = analyze_by_regime(scored, regime_df, "ret", window_days=5)
    assert {r.regime for r in results} == set(REGIMES)
    for r in results:
        assert r.n > 0
