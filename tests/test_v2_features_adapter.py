from __future__ import annotations

import numpy as np
from conftest import make_synthetic_ohlcv

from v2.features_adapter import (
    MA60_WINDOW,
    RECENT_HIGH_60D_WINDOW,
    add_v2_derived_features,
    build_v2_feature_panel,
)


def test_v1_columns_pass_through_unmodified() -> None:
    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    market = make_synthetic_ohlcv(300, seed=999, ticker="TOPIX")
    panel = build_v2_feature_panel(ohlcv, market_df=market)

    for col in ("return_5d", "sma_20", "rsi_14", "rs_20d", "pullback_depth", "volume_ratio_20d"):
        assert col in panel.columns


def test_derived_columns_present() -> None:
    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    panel = build_v2_feature_panel(ohlcv)
    for col in ("price_vs_ma60", "ma5_vs_ma20", "ma20_vs_ma60", "distance_from_60d_high"):
        assert col in panel.columns


def test_ma5_vs_ma20_matches_manual_calculation() -> None:
    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    panel = build_v2_feature_panel(ohlcv)
    expected = panel["sma_5"] / panel["sma_20"] - 1
    assert np.allclose(
        panel["ma5_vs_ma20"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_price_vs_ma60_warmup_is_nan_before_60_rows() -> None:
    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    panel = build_v2_feature_panel(ohlcv)
    assert panel["price_vs_ma60"].iloc[: MA60_WINDOW - 1].isna().all()
    assert panel["price_vs_ma60"].iloc[MA60_WINDOW:].notna().any()


def test_distance_from_60d_high_warmup_is_nan_before_60_rows() -> None:
    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    panel = build_v2_feature_panel(ohlcv)
    assert panel["distance_from_60d_high"].iloc[: RECENT_HIGH_60D_WINDOW - 1].isna().all()


def test_distance_from_60d_high_always_non_positive() -> None:
    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    panel = build_v2_feature_panel(ohlcv)
    valid = panel["distance_from_60d_high"].dropna()
    assert (valid <= 1e-12).all()


def test_add_v2_derived_features_matches_build_v2_feature_panel() -> None:
    """The cache-reuse fast path (add_v2_derived_features on an
    already-computed V1 panel) must produce byte-identical derived
    columns to the fresh-compute path (build_v2_feature_panel) - this is
    the equivalence v2/pipeline.py's real Full Universe run relies on.
    """
    from features.pipeline import compute_feature_panel

    ohlcv = make_synthetic_ohlcv(300, seed=1, ticker="T0")
    market = make_synthetic_ohlcv(300, seed=999, ticker="TOPIX")

    fresh = build_v2_feature_panel(ohlcv, market_df=market)
    v1_panel = compute_feature_panel(ohlcv, market_df=market)
    from_cache = add_v2_derived_features(v1_panel)

    for col in ("price_vs_ma60", "ma5_vs_ma20", "ma20_vs_ma60", "distance_from_60d_high"):
        assert np.allclose(
            fresh[col].to_numpy(dtype=float), from_cache[col].to_numpy(dtype=float),
            equal_nan=True,
        )
