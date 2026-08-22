from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from config.loader import AppConfig, UniverseFilterConfig
from pipeline.ingest import run_ingest
from providers.base import (
    FetchResult,
    FetchStatus,
    MarketIndexMeta,
    MarketIndexProvider,
    OHLCVProvider,
)


def _ohlcv(ticker: str, n_days: int = 20, volume: int = 50_000) -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "ticker": [ticker] * n_days,
            "date": dates,
            "open": [500.0] * n_days,
            "high": [505.0] * n_days,
            "low": [495.0] * n_days,
            "close": [500.0] * n_days,
            "volume": [volume] * n_days,
        }
    )


class FakeOHLCVProvider(OHLCVProvider):
    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses

    def fetch(self, ticker: str, start: date, end: date) -> FetchResult:
        return self.responses[ticker]


class FakeMarketIndexProvider(MarketIndexProvider):
    def __init__(self, result: FetchResult) -> None:
        self.result = result

    def fetch_index(self, start: date, end: date) -> FetchResult:
        return self.result

    def describe(self) -> MarketIndexMeta:
        return MarketIndexMeta(
            name="Fake Index", symbol=self.result.ticker, type="TEST", source="fake"
        )


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


def test_run_ingest_with_fake_providers(base_config: AppConfig) -> None:
    responses = {
        "7203": FetchResult("7203", FetchStatus.SUCCESS, _ohlcv("7203")),
        "9999": FetchResult("9999", FetchStatus.FAILED, None, error="no data returned"),
        "1332": FetchResult("1332", FetchStatus.SUCCESS, _ohlcv("1332", volume=10)),  # illiquid
    }
    provider = FakeOHLCVProvider(responses)
    index_provider = FakeMarketIndexProvider(
        FetchResult("^TOPX", FetchStatus.SUCCESS, _ohlcv("^TOPX"))
    )

    summary = run_ingest(
        base_config,
        UniverseFilterConfig.model_validate({"liquidity": {"min_avg_volume_20d": 10_000}}),
        tickers_override=["7203", "9999", "1332"],
        provider=provider,
        index_provider=index_provider,
    )

    assert summary.fetch_success == 2
    assert summary.fetch_failed == 1
    assert summary.processed_count == 1  # only 7203 passes the liquidity filter
    assert summary.excluded_by_liquidity == 1
    assert summary.topix_available is True
    assert (Path(base_config.data.raw_dir) / "7203.parquet").exists()
    assert (Path(base_config.data.processed_dir) / "7203.parquet").exists()
    assert not (Path(base_config.data.processed_dir) / "1332.parquet").exists()


def test_run_ingest_marks_topix_unavailable_on_failure(base_config: AppConfig) -> None:
    provider = FakeOHLCVProvider({"7203": FetchResult("7203", FetchStatus.SUCCESS, _ohlcv("7203"))})
    index_provider = FakeMarketIndexProvider(
        FetchResult("^TOPX", FetchStatus.FAILED, None, error="network down")
    )

    summary = run_ingest(
        base_config,
        UniverseFilterConfig(),
        tickers_override=["7203"],
        provider=provider,
        index_provider=index_provider,
    )

    assert summary.topix_available is False
