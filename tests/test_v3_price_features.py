from __future__ import annotations

from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from v3.features.price_features import add_v3_price_features


def _panel(n_days: int = 200, seed: int = 1):
    ohlcv = make_synthetic_ohlcv(n_days, seed=seed, ticker="T0")
    return compute_feature_panel(ohlcv)


def test_adds_return_120d() -> None:
    panel = add_v3_price_features(_panel(200))
    assert "return_120d" in panel.columns
    assert panel["return_120d"].notna().sum() > 0


def test_adds_new_sma_windows_and_ratios() -> None:
    panel = add_v3_price_features(_panel(200))
    for col in (
        "close_to_sma_10", "close_to_sma_60", "close_to_sma_120",
        "ma5_to_ma20", "ma20_to_ma60", "ma60_to_ma120",
    ):
        assert col in panel.columns


def test_adds_rsi5_and_rsi20_bounded_0_100() -> None:
    panel = add_v3_price_features(_panel(200))
    for col in ("rsi_5", "rsi_20"):
        values = panel[col].dropna()
        assert len(values) > 0
        assert (values >= 0).all() and (values <= 100).all()


def test_turnover_is_close_times_volume() -> None:
    panel = add_v3_price_features(_panel(50))
    expected = panel["close"] * panel["volume"]
    assert (panel["turnover"].dropna() == expected.dropna()).all()


def test_turnover_ratio_is_finite_and_positive_where_defined() -> None:
    panel = add_v3_price_features(_panel(50))
    values = panel["turnover_ratio"].dropna()
    assert len(values) > 0
    assert (values > 0).all()


def test_drawdown_from_high_is_nonnegative_and_distance_is_nonpositive() -> None:
    panel = add_v3_price_features(_panel(200))
    for window in (20, 60, 120):
        distance = panel[f"distance_from_{window}d_high"].dropna()
        drawdown = panel[f"drawdown_from_{window}d_high"].dropna()
        assert (distance <= 1e-12).all()
        assert (drawdown >= -1e-12).all()
        # Same magnitude, opposite sign.
        common_idx = distance.index.intersection(drawdown.index)
        assert (
            (distance.loc[common_idx] + drawdown.loc[common_idx]).abs() < 1e-9
        ).all()


def test_volatility_change_uses_volatility_20d() -> None:
    panel = add_v3_price_features(_panel(60))
    assert "volatility_change" in panel.columns
    assert panel["volatility_change"].notna().sum() > 0


def test_never_mutates_input_panel() -> None:
    base = _panel(60)
    before = base.copy(deep=True)
    add_v3_price_features(base)
    assert base.equals(before)
