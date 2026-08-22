from __future__ import annotations

import numpy as np
import pytest
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel, feature_metadata

N_ROWS = 300  # long enough to clear every feature's warmup (max is macd_signal at 34)


def test_missing_required_columns_raises() -> None:
    df = make_synthetic_ohlcv(20, seed=1).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_feature_panel(df)


def test_output_preserves_row_count_and_order() -> None:
    df = make_synthetic_ohlcv(N_ROWS, seed=13)
    panel = compute_feature_panel(df)
    assert len(panel) == len(df)
    assert panel["date"].tolist() == df["date"].tolist()


def test_ohlcv_columns_pass_through_unchanged() -> None:
    df = make_synthetic_ohlcv(N_ROWS, seed=13)
    panel = compute_feature_panel(df)
    for col in ["ticker", "date", "open", "high", "low", "close", "volume"]:
        assert (panel[col] == df[col]).all()


@pytest.mark.parametrize("meta", feature_metadata(), ids=lambda m: m.name)
def test_feature_warmup_matches_declared_metadata(meta) -> None:
    df = make_synthetic_ohlcv(N_ROWS, seed=14)
    # Same seed range/dates as df (just different prices, via a different
    # seed) so the market benchmark has full date coverage - otherwise
    # rs_Nd's actual NaN count could exceed its declared warmup_period
    # for reasons unrelated to what this test is checking (see
    # features/relative_strength.py's warmup_period caveat).
    market_df = make_synthetic_ohlcv(N_ROWS, seed=99, ticker="TOPIX")
    panel = compute_feature_panel(df, recent_high_window=20, market_df=market_df)

    assert meta.name in panel.columns, f"{meta.name} missing from panel"
    column = panel[meta.name]

    expected_nan_rows = meta.warmup_period - 1
    assert column.iloc[:expected_nan_rows].isna().all(), (
        f"{meta.name}: expected all of the first {expected_nan_rows} rows to be NaN"
    )
    assert column.iloc[expected_nan_rows:].notna().all(), (
        f"{meta.name}: expected no NaN from row {expected_nan_rows} onward "
        "(warmup_period is declared too high)"
    )


def test_no_feature_column_contains_infinite_values() -> None:
    df = make_synthetic_ohlcv(N_ROWS, seed=15)
    # Force some zero-volume and flat-price stretches, which is where
    # division-by-zero would show up if safe_divide() were not used.
    df.loc[df.index[50:60], "volume"] = 0.0
    df.loc[df.index[100:120], "close"] = df["close"].iloc[100]
    df.loc[df.index[100:120], "open"] = df["close"].iloc[100]
    df.loc[df.index[100:120], "high"] = df["close"].iloc[100]
    df.loc[df.index[100:120], "low"] = df["close"].iloc[100]

    market_df = make_synthetic_ohlcv(N_ROWS, seed=16, ticker="TOPIX")
    panel = compute_feature_panel(df, market_df=market_df)
    feature_cols = [m.name for m in feature_metadata()]
    for col in feature_cols:
        values = panel[col].dropna()
        assert not np.isinf(values).any(), col
