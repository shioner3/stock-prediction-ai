from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from config.loader import AppConfig, UniverseFilterConfig
from pipeline.ingest import run_ingest
from pipeline.universe_ingest import run_universe_ingest
from providers.base import (
    FetchResult,
    FetchStatus,
    MarketIndexMeta,
    MarketIndexProvider,
    OHLCVProvider,
)


class FakeMarketIndexProvider(MarketIndexProvider):
    def fetch_index(self, start: date, end: date) -> FetchResult:
        return FetchResult("TOPIX", FetchStatus.FAILED, None, error="not needed for this test")

    def describe(self) -> MarketIndexMeta:
        return MarketIndexMeta(name="Fake", symbol="TOPIX", type="TEST", source="fake")


class CountingFakeMarketIndexProvider(MarketIndexProvider):
    """Like FakeMarketIndexProvider, but SUCCEEDS and counts calls - used
    to prove Phase 7's bug fix (run_universe_ingest now fetches TOPIX)
    and its cache-skip behavior.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def fetch_index(self, start: date, end: date) -> FetchResult:
        self.call_count += 1
        return FetchResult("TOPIX", FetchStatus.SUCCESS, _ohlcv("TOPIX", n_days=20))

    def describe(self) -> MarketIndexMeta:
        return MarketIndexMeta(name="Fake", symbol="TOPIX", type="TEST", source="fake")


def _ohlcv(ticker: str, n_days: int = 20, volume: int = 50_000) -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "ticker": [ticker] * n_days, "date": dates,
            "open": [500.0] * n_days, "high": [505.0] * n_days,
            "low": [495.0] * n_days, "close": [500.0] * n_days,
            "volume": [volume] * n_days,
        }
    )


class CountingFakeProvider(OHLCVProvider):
    """Like test_pipeline_ingest.py's FakeOHLCVProvider, but counts calls
    per ticker - used to prove the checkpoint/resume wrapper actually
    skips already-fetched tickers rather than re-fetching them.
    """

    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses
        self.call_count: dict[str, int] = dict.fromkeys(responses, 0)

    def fetch(self, ticker: str, start: date, end: date) -> FetchResult:
        self.call_count[ticker] = self.call_count.get(ticker, 0) + 1
        return self.responses[ticker]


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2024-01-01",
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


def _responses() -> dict[str, FetchResult]:
    return {
        "7203": FetchResult("7203", FetchStatus.SUCCESS, _ohlcv("7203")),
        "6758": FetchResult("6758", FetchStatus.SUCCESS, _ohlcv("6758")),
        "9999": FetchResult("9999", FetchStatus.FAILED, None, error="no data returned"),
        "1332": FetchResult("1332", FetchStatus.SUCCESS, _ohlcv("1332", volume=10)),  # illiquid
    }


def test_first_run_attempts_every_ticker(base_config: AppConfig, tmp_path: Path) -> None:
    provider = CountingFakeProvider(_responses())
    manifest = tmp_path / "manifest.json"
    tickers = list(_responses().keys())

    filter_config = UniverseFilterConfig.model_validate(
        {"liquidity": {"min_avg_volume_20d": 10_000}}
    )
    summary = run_universe_ingest(
        base_config, filter_config, tickers, manifest,
        provider=provider, index_provider=FakeMarketIndexProvider(),
    )

    assert summary.attempted == 4
    assert summary.skipped_cached == 0
    assert summary.fetch_success == 3
    assert summary.fetch_failed == 1
    assert summary.processed_count == 2  # 7203, 6758 (1332 excluded by liquidity)
    assert summary.excluded_by_liquidity == 1
    for ticker in tickers:
        assert provider.call_count[ticker] == 1


def test_second_run_skips_already_successful_tickers(
    base_config: AppConfig, tmp_path: Path
) -> None:
    filter_config = UniverseFilterConfig.model_validate(
        {"liquidity": {"min_avg_volume_20d": 10_000}}
    )
    tickers = list(_responses().keys())
    manifest = tmp_path / "manifest.json"

    provider_a = CountingFakeProvider(_responses())
    run_universe_ingest(
        base_config, filter_config, tickers, manifest,
        provider=provider_a, index_provider=FakeMarketIndexProvider(),
    )

    provider_b = CountingFakeProvider(_responses())
    summary_b = run_universe_ingest(
        base_config, filter_config, tickers, manifest,
        provider=provider_b, index_provider=FakeMarketIndexProvider(),
    )

    # 7203, 6758, 1332 succeeded (or were fully evaluated) on the first
    # run and must be skipped; only 9999 (failed -> not cache-eligible)
    # gets retried.
    assert summary_b.skipped_cached == 3
    assert summary_b.attempted == 1
    assert provider_b.call_count["7203"] == 0
    assert provider_b.call_count["6758"] == 0
    assert provider_b.call_count["1332"] == 0
    assert provider_b.call_count["9999"] == 1


def test_resumed_run_gives_same_aggregate_counts_as_uninterrupted_run(
    base_config: AppConfig, tmp_path: Path
) -> None:
    filter_config = UniverseFilterConfig.model_validate(
        {"liquidity": {"min_avg_volume_20d": 10_000}}
    )
    tickers = list(_responses().keys())

    # Uninterrupted: one call processes everything.
    manifest_a = tmp_path / "manifest_a.json"
    provider_a = CountingFakeProvider(_responses())
    summary_a = run_universe_ingest(
        base_config, filter_config, tickers, manifest_a,
        provider=provider_a, index_provider=FakeMarketIndexProvider(),
    )

    # "Interrupted" and resumed: split into two calls against a shared manifest.
    manifest_b = tmp_path / "manifest_b.json"
    provider_b1 = CountingFakeProvider(_responses())
    run_universe_ingest(
        base_config, filter_config, tickers[:2], manifest_b,
        provider=provider_b1, index_provider=FakeMarketIndexProvider(),
    )
    provider_b2 = CountingFakeProvider(_responses())
    summary_b2 = run_universe_ingest(
        base_config, filter_config, tickers, manifest_b,
        provider=provider_b2, index_provider=FakeMarketIndexProvider(),
    )

    assert summary_a.fetch_success == summary_b2.fetch_success
    assert summary_a.fetch_failed == summary_b2.fetch_failed
    assert summary_a.processed_count == summary_b2.processed_count
    assert summary_a.excluded_by_liquidity == summary_b2.excluded_by_liquidity


def test_force_refetch_bypasses_cache(base_config: AppConfig, tmp_path: Path) -> None:
    filter_config = UniverseFilterConfig()
    tickers = ["7203"]
    manifest = tmp_path / "manifest.json"

    response = {"7203": FetchResult("7203", FetchStatus.SUCCESS, _ohlcv("7203"))}
    provider_a = CountingFakeProvider(response)
    run_universe_ingest(
        base_config, filter_config, tickers, manifest,
        provider=provider_a, index_provider=FakeMarketIndexProvider(),
    )

    provider_b = CountingFakeProvider(response)
    run_universe_ingest(
        base_config, filter_config, tickers, manifest,
        provider=provider_b, index_provider=FakeMarketIndexProvider(), force_refetch=True,
    )

    assert provider_b.call_count["7203"] == 1


def test_manifest_persisted_with_expected_shape(base_config: AppConfig, tmp_path: Path) -> None:
    filter_config = UniverseFilterConfig.model_validate(
        {"liquidity": {"min_avg_volume_20d": 10_000}}
    )
    manifest = tmp_path / "manifest.json"
    provider = CountingFakeProvider(_responses())
    run_universe_ingest(
        base_config, filter_config, list(_responses().keys()), manifest,
        provider=provider, index_provider=FakeMarketIndexProvider(),
    )

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "tickers" in data
    assert data["tickers"]["7203"]["status"] == "success"
    assert data["tickers"]["7203"]["included_in_universe"] is True
    assert data["tickers"]["9999"]["status"] == "failed"
    assert data["tickers"]["1332"]["included_in_universe"] is False


def test_universe_ingest_output_matches_run_ingest_for_same_tickers(
    base_config: AppConfig, tmp_path: Path
) -> None:
    """Phase 6.5 spec section 4.8: changing the FETCH ORCHESTRATION
    (checkpoint/resume) must not change what gets written for a given
    ticker - the raw/processed Parquet files produced via
    run_universe_ingest() must be byte-identical to run_ingest()'s.
    """
    from storage.parquet_store import load_ohlcv

    responses = {"7203": FetchResult("7203", FetchStatus.SUCCESS, _ohlcv("7203"))}
    filter_config = UniverseFilterConfig()

    config_a = base_config.model_copy(deep=True)
    config_a.data.raw_dir = tmp_path / "a_raw"
    config_a.data.processed_dir = tmp_path / "a_processed"
    run_ingest(
        config_a, filter_config, tickers_override=["7203"],
        provider=CountingFakeProvider(responses), index_provider=FakeMarketIndexProvider(),
    )

    config_b = base_config.model_copy(deep=True)
    config_b.data.raw_dir = tmp_path / "b_raw"
    config_b.data.processed_dir = tmp_path / "b_processed"
    run_universe_ingest(
        config_b, filter_config, ["7203"], tmp_path / "manifest.json",
        provider=CountingFakeProvider(responses), index_provider=FakeMarketIndexProvider(),
    )

    a = load_ohlcv("7203", config_a.data.processed_dir)
    b = load_ohlcv("7203", config_b.data.processed_dir)
    pd.testing.assert_frame_equal(a, b)


# --- Phase 7 bug fix: run_universe_ingest now fetches the market index -----


def _single_success_provider(ticker: str = "7203") -> CountingFakeProvider:
    return CountingFakeProvider({ticker: FetchResult(ticker, FetchStatus.SUCCESS, _ohlcv(ticker))})


def test_run_universe_ingest_fetches_and_saves_topix(
    base_config: AppConfig, tmp_path: Path
) -> None:
    filter_config = UniverseFilterConfig()
    manifest = tmp_path / "manifest.json"
    index_provider = CountingFakeMarketIndexProvider()

    summary = run_universe_ingest(
        base_config, filter_config, ["7203"], manifest,
        provider=_single_success_provider(), index_provider=index_provider,
    )

    assert summary.topix_available is True
    assert index_provider.call_count == 1
    assert (Path(base_config.data.processed_dir) / "TOPIX.parquet").exists()
    assert (Path(base_config.data.raw_dir) / "TOPIX.parquet").exists()


def test_run_universe_ingest_reuses_cached_topix_without_refetching(
    base_config: AppConfig, tmp_path: Path
) -> None:
    filter_config = UniverseFilterConfig()
    manifest = tmp_path / "manifest.json"

    index_provider_a = CountingFakeMarketIndexProvider()
    run_universe_ingest(
        base_config, filter_config, ["7203"], manifest,
        provider=_single_success_provider(), index_provider=index_provider_a,
    )

    index_provider_b = CountingFakeMarketIndexProvider()
    summary_b = run_universe_ingest(
        base_config, filter_config, ["7203"], manifest,
        provider=_single_success_provider(), index_provider=index_provider_b,
    )

    assert summary_b.topix_available is True
    assert index_provider_b.call_count == 0  # cached TOPIX.parquet reused, not refetched


def test_run_universe_ingest_force_refetch_also_refetches_topix(
    base_config: AppConfig, tmp_path: Path
) -> None:
    filter_config = UniverseFilterConfig()
    manifest = tmp_path / "manifest.json"

    index_provider_a = CountingFakeMarketIndexProvider()
    run_universe_ingest(
        base_config, filter_config, ["7203"], manifest,
        provider=_single_success_provider(), index_provider=index_provider_a,
    )

    index_provider_b = CountingFakeMarketIndexProvider()
    run_universe_ingest(
        base_config, filter_config, ["7203"], manifest,
        provider=_single_success_provider(), index_provider=index_provider_b,
        force_refetch=True,
    )

    assert index_provider_b.call_count == 1


def test_run_universe_ingest_records_topix_unavailable_on_index_failure(
    base_config: AppConfig, tmp_path: Path
) -> None:
    filter_config = UniverseFilterConfig()
    manifest = tmp_path / "manifest.json"

    summary = run_universe_ingest(
        base_config, filter_config, ["7203"], manifest,
        provider=_single_success_provider(), index_provider=FakeMarketIndexProvider(),
    )

    assert summary.topix_available is False
    assert not (Path(base_config.data.processed_dir) / "TOPIX.parquet").exists()
