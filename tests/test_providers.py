from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from providers.base import FetchStatus
from providers.market_index import YFinanceMarketIndexProvider
from providers.yfinance_provider import (
    YFinanceProvider,
    fetch_yf_history,
    to_yahoo_symbol,
)


def test_to_yahoo_symbol_appends_suffix() -> None:
    assert to_yahoo_symbol("7203") == "7203.T"


def test_to_yahoo_symbol_passes_through_existing_suffix() -> None:
    assert to_yahoo_symbol("7203.T") == "7203.T"


def test_to_yahoo_symbol_passes_through_index_symbol() -> None:
    assert to_yahoo_symbol("^TOPX") == "^TOPX"


def _fake_history(n_days: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "Open": [100.0] * n_days,
            "High": [101.0] * n_days,
            "Low": [99.0] * n_days,
            "Close": [100.5] * n_days,
            "Volume": [10_000] * n_days,
        },
        index=pd.Index(dates, name="Date"),
    )


def test_yfinance_provider_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(*args, **kwargs):
        return _fake_history(10)

    monkeypatch.setattr("providers.yfinance_provider.fetch_yf_history", fake_fetch)
    provider = YFinanceProvider(min_expected_coverage=0.5)
    result = provider.fetch("7203", date(2024, 1, 1), date(2024, 1, 12))

    assert result.status == FetchStatus.SUCCESS
    assert result.data is not None
    assert list(result.data.columns) == ["ticker", "date", "open", "high", "low", "close", "volume"]
    assert (result.data["ticker"] == "7203").all()


def test_yfinance_provider_partial_on_low_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(*args, **kwargs):
        return _fake_history(2)

    monkeypatch.setattr("providers.yfinance_provider.fetch_yf_history", fake_fetch)
    provider = YFinanceProvider(min_expected_coverage=0.8)
    result = provider.fetch("7203", date(2024, 1, 1), date(2024, 1, 31))

    assert result.status == FetchStatus.PARTIAL
    assert result.data is not None


def test_yfinance_provider_failed_on_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("providers.yfinance_provider.fetch_yf_history", fake_fetch)
    provider = YFinanceProvider()
    result = provider.fetch("9999", date(2024, 1, 1), date(2024, 1, 12))

    assert result.status == FetchStatus.FAILED
    assert result.data is None
    assert result.error is not None


def test_yfinance_provider_failed_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("providers.yfinance_provider.fetch_yf_history", fake_fetch)
    provider = YFinanceProvider(max_retries=2)
    result = provider.fetch("7203", date(2024, 1, 1), date(2024, 1, 12))

    assert result.status == FetchStatus.FAILED
    assert "network down" in result.error


def test_fetch_yf_history_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return _fake_history(5)

    monkeypatch.setattr("providers.yfinance_provider.yf.Ticker", FakeTicker)
    monkeypatch.setattr("providers.yfinance_provider.time.sleep", lambda _: None)

    result = fetch_yf_history("7203.T", date(2024, 1, 1), date(2024, 1, 10), max_retries=5)

    assert calls["n"] == 3
    assert not result.empty


def test_fetch_yf_history_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs):
            raise ConnectionError("still down")

    monkeypatch.setattr("providers.yfinance_provider.yf.Ticker", FakeTicker)
    monkeypatch.setattr("providers.yfinance_provider.time.sleep", lambda _: None)

    with pytest.raises(ConnectionError):
        fetch_yf_history("7203.T", date(2024, 1, 1), date(2024, 1, 10), max_retries=3)


def test_market_index_provider_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(*args, **kwargs):
        return _fake_history(10)

    monkeypatch.setattr("providers.market_index.fetch_yf_history", fake_fetch)
    provider = YFinanceMarketIndexProvider(index_symbol="^TOPX")
    result = provider.fetch_index(date(2024, 1, 1), date(2024, 1, 12))

    assert result.status == FetchStatus.SUCCESS
    assert result.data is not None
    assert (result.data["ticker"] == "^TOPX").all()


def test_market_index_provider_failed_logs_and_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr("providers.market_index.fetch_yf_history", fake_fetch)
    provider = YFinanceMarketIndexProvider(index_symbol="^TOPX")
    result = provider.fetch_index(date(2024, 1, 1), date(2024, 1, 12))

    assert result.status == FetchStatus.FAILED


def test_market_index_provider_describe_reflects_constructor_args() -> None:
    provider = YFinanceMarketIndexProvider(
        index_symbol="1306.T", name="TOPIX Proxy", index_type="ETF_PROXY"
    )
    meta = provider.describe()

    assert meta.name == "TOPIX Proxy"
    assert meta.symbol == "1306.T"
    assert meta.type == "ETF_PROXY"
    assert meta.source == "yfinance"


@pytest.mark.network
def test_yfinance_provider_real_network_call() -> None:
    """Real network smoke test - deselected by default (see pyproject addopts).

    Run explicitly with: pytest -m network
    """
    provider = YFinanceProvider(max_retries=1, timeout_seconds=10.0)
    result = provider.fetch("7203", date(2024, 1, 1), date(2024, 1, 31))
    assert result.status in (FetchStatus.SUCCESS, FetchStatus.PARTIAL, FetchStatus.FAILED)
