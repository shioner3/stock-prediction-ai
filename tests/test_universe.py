from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from config.loader import UniverseFilterConfig
from universe.build import apply_static_filters, classify_instrument_type, load_master_list
from universe.filters import check_price_and_liquidity


def test_classify_instrument_type_etf() -> None:
    assert classify_instrument_type("ETF・ETN") == "ETF"


def test_classify_instrument_type_reit() -> None:
    label = "REIT・ベンチャーファンド・カントリーファンド・インフラファンド"
    assert classify_instrument_type(label) == "REIT"


def test_classify_instrument_type_stock() -> None:
    assert classify_instrument_type("輸送用機器") == "STOCK"


@pytest.fixture
def master_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["7203", "1306", "8951", "9999"],
            "name": ["Toyota", "TOPIX ETF", "Nippon Building Fund", "Foreign Co"],
            "market_segment": ["Prime", "Prime", "Prime", "TSE Other"],
            "sector33": ["輸送用機器", "ETF・ETN", "REIT・ベンチャーファンド", "銀行業"],
        }
    )


def test_apply_static_filters_excludes_etf_reit_and_other_segments(master_df: pd.DataFrame) -> None:
    result = apply_static_filters(
        master_df, segments=["Prime", "Standard", "Growth"], exclude_etf=True, exclude_reit=True
    )
    assert result.included["code"].tolist() == ["7203"]
    assert set(result.excluded["code"]) == {"1306", "8951", "9999"}


def test_apply_static_filters_can_keep_etf_when_disabled(master_df: pd.DataFrame) -> None:
    result = apply_static_filters(
        master_df, segments=["Prime", "Standard", "Growth"], exclude_etf=False, exclude_reit=True
    )
    assert "1306" in result.included["code"].tolist()


def test_load_master_list_sample_fixture_is_readable(sample_master_csv_path: str) -> None:
    df = load_master_list(sample_master_csv_path)
    assert {"code", "name", "market_segment", "sector33"}.issubset(df.columns)
    assert len(df) > 0


def test_load_master_list_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_master_list("data/reference/does_not_exist.csv")


def _liquid_ohlcv_df(avg_volume: float = 50_000, avg_close: float = 500.0) -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(20)]
    return pd.DataFrame(
        {
            "ticker": ["7203"] * 20,
            "date": dates,
            "open": [avg_close] * 20,
            "high": [avg_close + 1] * 20,
            "low": [avg_close - 1] * 20,
            "close": [avg_close] * 20,
            "volume": [avg_volume] * 20,
        }
    )


def test_check_price_and_liquidity_passes_for_liquid_stock() -> None:
    df = _liquid_ohlcv_df()
    config = UniverseFilterConfig.model_validate(
        {"price": {"min_close_price": 100}, "liquidity": {"min_avg_volume_20d": 10_000}}
    )
    result = check_price_and_liquidity(df, "7203", config)
    assert result.passed


def test_check_price_and_liquidity_fails_on_low_price() -> None:
    df = _liquid_ohlcv_df(avg_close=50.0)
    config = UniverseFilterConfig.model_validate({"price": {"min_close_price": 100}})
    result = check_price_and_liquidity(df, "7203", config)
    assert not result.passed
    assert "avg_close" in result.reason


def test_check_price_and_liquidity_fails_on_low_volume() -> None:
    df = _liquid_ohlcv_df(avg_volume=100)
    config = UniverseFilterConfig.model_validate({"liquidity": {"min_avg_volume_20d": 10_000}})
    result = check_price_and_liquidity(df, "7203", config)
    assert not result.passed
    assert "avg_volume" in result.reason


def test_check_price_and_liquidity_uses_earliest_rows_not_latest() -> None:
    """Phase 6.5 Data/Universe Leakage Fix: a ticker illiquid at the
    START of its history but liquid by the END must still be EXCLUDED -
    using the tail (as this function did before the fix) would let
    "today's" liquidity leak into a decision that governs the ticker's
    entire history, including Walk Forward windows years earlier.
    """
    n = 40
    dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    # First 20 rows: illiquid (volume=100). Last 20 rows: liquid (volume=50_000).
    volume = [100] * 20 + [50_000] * 20
    df = pd.DataFrame(
        {
            "ticker": ["7203"] * n, "date": dates,
            "open": [500.0] * n, "high": [505.0] * n, "low": [495.0] * n,
            "close": [500.0] * n, "volume": volume,
        }
    )
    config = UniverseFilterConfig.model_validate({"liquidity": {"min_avg_volume_20d": 10_000}})

    result = check_price_and_liquidity(df, "7203", config, lookback_days=20)

    assert not result.passed  # decided by the EARLY illiquid period, not the later liquid one
    assert "avg_volume" in result.reason


def test_check_price_and_liquidity_early_liquidity_is_not_overridden_by_later_illiquidity() -> None:
    """The mirror case: liquid at the start, illiquid by the end - still
    PASSES, because eligibility is decided at the start of the period.
    """
    n = 40
    dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    volume = [50_000] * 20 + [100] * 20
    df = pd.DataFrame(
        {
            "ticker": ["7203"] * n, "date": dates,
            "open": [500.0] * n, "high": [505.0] * n, "low": [495.0] * n,
            "close": [500.0] * n, "volume": volume,
        }
    )
    config = UniverseFilterConfig.model_validate({"liquidity": {"min_avg_volume_20d": 10_000}})

    result = check_price_and_liquidity(df, "7203", config, lookback_days=20)

    assert result.passed


def test_check_price_and_liquidity_fails_on_empty_data() -> None:
    config = UniverseFilterConfig()
    result = check_price_and_liquidity(pd.DataFrame(), "7203", config)
    assert not result.passed
