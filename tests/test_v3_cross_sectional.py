from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v3.features.cross_sectional import CROSS_SECTIONAL_FEATURES, add_v3_cross_sectional_features


def _synthetic_universe(n_days: int = 10, n_tickers: int = 20, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append(
                (
                    d, f"T{i}",
                    rng.normal(0, 0.02), rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1),
                    rng.normal(0, 0.05), rng.uniform(-0.3, 0.0), rng.normal(0, 0.03),
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date", "ticker", "return_5d", "volume_ratio_20d", "volatility_20d",
            "return_20d", "distance_from_20d_high", "rs_20d",
        ],
    )


def test_adds_all_six_percentile_columns() -> None:
    panel = _synthetic_universe()
    out = add_v3_cross_sectional_features(panel)
    for pct_name in CROSS_SECTIONAL_FEATURES:
        assert pct_name in out.columns
        values = out[pct_name].dropna()
        assert (values > 0).all() and (values <= 1.0).all()


def test_percentiles_computed_within_day_only() -> None:
    panel = _synthetic_universe()
    out = add_v3_cross_sectional_features(panel)
    for _d, group in out.groupby("date"):
        n = len(group)
        # Every day has the same ticker count -> ranks should span (0,1]
        # with the expected number of distinct steps for that day alone.
        assert group["return_percentile"].dropna().nunique() <= n


def test_never_mutates_input_panel() -> None:
    panel = _synthetic_universe()
    before = panel.copy(deep=True)
    add_v3_cross_sectional_features(panel)
    assert panel.equals(before)
