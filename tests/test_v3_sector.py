from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from v3.features.sector import JPX_MASTER_CACHE_PATH, attach_sector_features, load_ticker_sector_map


def test_load_ticker_sector_map_reads_local_cache() -> None:
    assert Path(JPX_MASTER_CACHE_PATH).exists()
    sector_map = load_ticker_sector_map()
    assert {"ticker", "sector33"} <= set(sector_map.columns)
    assert len(sector_map) > 1000


def test_attach_sector_features_computes_within_day_within_sector_mean() -> None:
    rng = np.random.default_rng(2)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    tickers = [f"T{i}" for i in range(6)]
    rows = []
    for d in dates:
        for i, t in enumerate(tickers):
            rows.append((d, t, rng.normal(0, 0.01), rng.normal(0, 0.02)))
    panel = pd.DataFrame(rows, columns=["date", "ticker", "return_1d", "return_20d"])
    sector_map = pd.DataFrame(
        {"ticker": tickers, "sector33": ["A", "A", "A", "B", "B", "B"]}
    )
    out = attach_sector_features(panel, sector_map)
    assert "industry_return" in out.columns
    assert "stock_vs_industry" in out.columns
    assert "industry_relative_strength" in out.columns

    for d in dates:
        day = out[out["date"] == d]
        sector_a = day[day["sector33"] == "A"]
        expected_mean = sector_a["return_1d"].mean()
        assert (abs(sector_a["industry_return"] - expected_mean) < 1e-9).all()
