from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_synthetic_ohlcv

from config.loader import AppConfig, UniverseFilterConfig
from forward_test.manifest import build_manifest, save_manifest
from pipeline.run_backtest import run_backtest
from pipeline.run_forward_test import (
    SafeAbortError,
    StrategyHashMismatchError,
    run_forward_test_day,
)
from providers.base import (
    FetchResult,
    FetchStatus,
    MarketIndexMeta,
    MarketIndexProvider,
    OHLCVProvider,
)


def _write_fake_jpx_xlsx(path: Path, tickers: list[str]) -> None:
    rows = [
        {
            "日付": 20260731,
            "コード": t,
            "銘柄名": f"Test {t}",
            "市場・商品区分": "プライム（内国株式）",
            "33業種コード": "9050",
            "33業種区分": "銀行業",
            "17業種コード": "16",
            "17業種区分": "金融（除く銀行）",
            "規模コード": "6",
            "規模区分": "TOPIX Small 1",
        }
        for t in tickers
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, index=False)


class FakeProvider(OHLCVProvider):
    """Returns the SAME fixed synthetic history regardless of the
    requested date range - matching the established pattern used
    throughout this project's other fetch-orchestration tests.
    """

    def __init__(self, responses: dict[str, pd.DataFrame]) -> None:
        self.responses = responses
        self.call_count: dict[str, int] = dict.fromkeys(responses, 0)

    def fetch(self, ticker: str, start: date, end: date) -> FetchResult:
        self.call_count[ticker] = self.call_count.get(ticker, 0) + 1
        if ticker not in self.responses:
            return FetchResult(ticker, FetchStatus.FAILED, None, error="no data")
        df = self.responses[ticker]
        subset = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if subset.empty:
            return FetchResult(ticker, FetchStatus.FAILED, None, error="no data in range")
        return FetchResult(ticker, FetchStatus.SUCCESS, subset)


class FakeMarketIndexProvider(MarketIndexProvider):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fetch_index(self, start: date, end: date) -> FetchResult:
        subset = self.df[(self.df["date"] >= start) & (self.df["date"] <= end)].reset_index(
            drop=True
        )
        return FetchResult("TOPIX", FetchStatus.SUCCESS, subset)

    def describe(self) -> MarketIndexMeta:
        return MarketIndexMeta(name="Fake", symbol="TOPIX", type="TEST", source="fake")


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "start_date": "2020-01-01",
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "features_dir": str(tmp_path / "features"),
                "signals_dir": str(tmp_path / "signals"),
                "scores_dir": str(tmp_path / "scores"),
            },
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


@pytest.fixture
def filter_config() -> UniverseFilterConfig:
    return UniverseFilterConfig()


def _setup_manifest(config: AppConfig, tmp_path: Path, t0: date) -> Path:
    manifest = build_manifest(
        config, "v1", t0, "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    path = tmp_path / "strategy_manifest.json"
    save_manifest(manifest, path)
    return path


def test_run_forward_test_day_missing_manifest_raises(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="no Strategy manifest"):
        run_forward_test_day(
            base_config, filter_config, date(2026, 8, 20),
            tmp_path / "nope.json", tmp_path / "fetch_manifest.json",
            tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
            tmp_path / "performance_log.jsonl", tmp_path / "jpx.xls",
        )


def test_run_forward_test_day_detects_hash_mismatch(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    strategy_path = _setup_manifest(base_config, tmp_path, date(2026, 8, 20))
    raw = json.loads(strategy_path.read_text(encoding="utf-8"))
    raw["hashes"]["features_hash"] = "0" * 64  # simulate a code change since T0
    strategy_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(StrategyHashMismatchError):
        run_forward_test_day(
            base_config, filter_config, date(2026, 8, 20),
            strategy_path, tmp_path / "fetch_manifest.json",
            tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
            tmp_path / "performance_log.jsonl", tmp_path / "jpx.xls",
        )


def test_run_forward_test_day_end_to_end_dry_run(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    t0 = date(2026, 8, 20)
    strategy_path = _setup_manifest(base_config, tmp_path, t0)

    tickers = ["T0", "T1", "T2"]
    jpx_path = tmp_path / "jpx.xls"
    _write_fake_jpx_xlsx(jpx_path, tickers)

    n = 300  # enough for SMA200 warmup
    responses = {t: make_synthetic_ohlcv(n, seed=100 + i, ticker=t) for i, t in enumerate(tickers)}
    # Anchor the synthetic series' last row at T0 itself so run_date's
    # data is actually present (make_synthetic_ohlcv always starts at
    # 2020-01-01, so shift dates forward to end exactly on t0).
    for t, df in responses.items():
        offset_days = (pd.Timestamp(t0) - pd.Timestamp(df["date"].iloc[-1])).days
        df["date"] = df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))
    market_df = make_synthetic_ohlcv(n, seed=999, ticker="TOPIX")
    offset_days = (pd.Timestamp(t0) - pd.Timestamp(market_df["date"].iloc[-1])).days
    market_df["date"] = market_df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))

    provider = FakeProvider(responses)
    index_provider = FakeMarketIndexProvider(market_df)

    perf_log_path = tmp_path / "performance_log.jsonl"
    result = run_forward_test_day(
        base_config, filter_config, t0,
        strategy_path, tmp_path / "fetch_manifest.json",
        tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
        perf_log_path, jpx_path,
        provider=provider, index_provider=index_provider,
    )

    assert result.run_date == t0
    assert result.universe_candidate_count == 3
    assert result.strategy_hash_unchanged is True
    assert result.strategy_hash_mismatches == []
    # No trade can resolve on day 1 (Entry needs t+1, Exit needs +hold_days
    # more) - equity must still equal the untouched initial capital.
    assert result.portfolio_equity == 10_000_000.0
    assert result.new_closed_positions == []
    assert result.open_positions == []  # not even an OPEN position yet (no t+1 close)
    assert result.trading_integrity.is_clean
    assert (tmp_path / "portfolio.json").exists()
    assert result.performance_log_written is True
    assert perf_log_path.exists()

    # Running the SAME day again must not duplicate signal log entries.
    log_path = tmp_path / "signal_log.jsonl"

    def _log_lines() -> list[str]:
        return log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []

    lines_after_first = _log_lines()
    perf_lines_after_first = perf_log_path.read_text(encoding="utf-8").splitlines()

    result2 = run_forward_test_day(
        base_config, filter_config, t0,
        strategy_path, tmp_path / "fetch_manifest.json",
        tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
        perf_log_path, jpx_path,
        provider=provider, index_provider=index_provider,
    )
    lines_after_second = _log_lines()

    assert result2.new_signal_log_entries == []  # nothing NEW today, already logged
    assert lines_after_second == lines_after_first  # no duplicate rows appended
    # Same run_date re-run: Performance Log stays a no-op append too.
    assert result2.performance_log_written is False
    assert perf_log_path.read_text(encoding="utf-8").splitlines() == perf_lines_after_first

    # Section 20 item 5: daily snapshot reproducibility - the two runs
    # (identical inputs) must agree on every structural/economic field.
    assert result2.run_date == result.run_date
    assert result2.final_universe_count == result.final_universe_count
    assert result2.portfolio_equity == result.portfolio_equity
    assert result2.trading_integrity == result.trading_integrity


def test_run_forward_test_day_never_fetches_beyond_run_date(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    """Section 20 item 3: no future data. A Provider that receives a
    fetch request with end > run_date would indicate the engine is
    reaching past what should be knowable "as of" that day.
    """
    t0 = date(2026, 8, 20)
    strategy_path = _setup_manifest(base_config, tmp_path, t0)
    jpx_path = tmp_path / "jpx.xls"
    _write_fake_jpx_xlsx(jpx_path, ["T0"])

    n = 300
    df = make_synthetic_ohlcv(n, seed=500, ticker="T0")
    offset_days = (pd.Timestamp(t0) - pd.Timestamp(df["date"].iloc[-1])).days
    df["date"] = df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))
    market_df = make_synthetic_ohlcv(n, seed=999, ticker="TOPIX")
    market_df["date"] = market_df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))

    class RecordingProvider(FakeProvider):
        def __init__(self, responses: dict[str, pd.DataFrame]) -> None:
            super().__init__(responses)
            self.requested_ranges: list[tuple[date, date]] = []

        def fetch(self, ticker: str, start: date, end: date) -> FetchResult:
            self.requested_ranges.append((start, end))
            return super().fetch(ticker, start, end)

    provider = RecordingProvider({"T0": df})
    index_provider = FakeMarketIndexProvider(market_df)

    run_forward_test_day(
        base_config, filter_config, t0,
        strategy_path, tmp_path / "fetch_manifest.json",
        tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
        tmp_path / "performance_log.jsonl", jpx_path,
        provider=provider, index_provider=index_provider,
    )

    assert provider.requested_ranges  # the fetch actually happened
    for _start, end in provider.requested_ranges:
        assert end <= t0


def test_run_forward_test_day_portfolio_has_no_live_order_fields() -> None:
    """Section 20 item 8: virtual execution only. Position must be a
    pure computed record (economics only) - no broker/order/account
    fields that would suggest a real execution path exists.
    """
    from forward_test.portfolio import Position

    field_names = {f.name for f in __import__("dataclasses").fields(Position)}
    forbidden_substrings = ("order", "broker", "account", "api_key")
    for name in field_names:
        for bad in forbidden_substrings:
            assert bad not in name.lower(), f"Position.{name} looks like a live-execution field"


def test_run_forward_test_day_excludes_pre_t0_signals_and_trades(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    """Phase 11 section 3: explicit regression test for the Phase 10
    T0 pre-signal-contamination bug - signals/trades dated BEFORE T0
    (present in the fetched history only for Feature warmup) must never
    reach the Signal Log or Paper Portfolio, even though the frozen
    Backtest Engine itself resolves them as ordinary completed trades
    over the FULL fetched history (it has no T0 concept at all - that
    filtering is entirely run_forward_test_day's responsibility).
    """
    n = 300
    run_date = date(2026, 8, 20)
    # T0 sits well inside the fetched window, leaving 200+ pre-T0
    # calendar days of history - long enough for long_oversold_rebound
    # to have fully resolved (entry+exit) many times before T0.
    t0 = run_date - timedelta(days=60)

    strategy_path = _setup_manifest(base_config, tmp_path, t0)
    jpx_path = tmp_path / "jpx.xls"
    _write_fake_jpx_xlsx(jpx_path, ["T0"])

    df = make_synthetic_ohlcv(n, seed=42, ticker="T0")
    offset_days = (pd.Timestamp(run_date) - pd.Timestamp(df["date"].iloc[-1])).days
    df["date"] = df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))
    market_df = make_synthetic_ohlcv(n, seed=999, ticker="TOPIX")
    market_df["date"] = market_df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))

    provider = FakeProvider({"T0": df})
    index_provider = FakeMarketIndexProvider(market_df)

    result = run_forward_test_day(
        base_config, filter_config, run_date,
        strategy_path, tmp_path / "fetch_manifest.json",
        tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
        tmp_path / "performance_log.jsonl", jpx_path,
        provider=provider, index_provider=index_provider,
    )

    for entry in result.new_signal_log_entries:
        assert date.fromisoformat(entry.signal_date) >= t0
    for pos in result.new_closed_positions:
        assert pos.signal_date >= t0

    log_path = tmp_path / "signal_log.jsonl"
    all_logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    for entry in all_logged:
        assert date.fromisoformat(entry["signal_date"]) >= t0

    portfolio_raw = json.loads((tmp_path / "portfolio.json").read_text(encoding="utf-8"))
    for p in portfolio_raw["closed_positions"]:
        assert date.fromisoformat(p["signal_date"]) >= t0

    # Sanity: the frozen Backtest Engine itself DOES see (and would
    # resolve) pre-T0 trades when not filtered - proving this test
    # would actually catch a regression, not just trivially pass
    # because the synthetic setup happens to have no pre-T0 trades.
    full_backtest = run_backtest(base_config, tickers=["T0"])
    pre_t0_trades = full_backtest.trades[full_backtest.trades["signal_date"] < t0]
    assert not pre_t0_trades.empty, (
        "test setup must produce at least one pre-T0 resolved trade, "
        "otherwise this regression test cannot actually detect the bug"
    )


def test_run_forward_test_day_universe_data_incomplete_aborts(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    """SAFE_ABORT (Phase 11 section 29): every candidate excluded by the
    frozen Universe filter (unrecognized market segment -> "OTHER",
    never in config.universe.segments) must abort loudly rather than
    silently running a Forward Test day against an empty Universe.
    """
    t0 = date(2026, 8, 20)
    strategy_path = _setup_manifest(base_config, tmp_path, t0)
    jpx_path = tmp_path / "jpx.xls"
    rows = [
        {
            "日付": 20260731, "コード": "T0", "銘柄名": "Test T0",
            "市場・商品区分": "PROマーケット（内国株式）",  # unmapped -> "OTHER"
            "33業種コード": "9050", "33業種区分": "銀行業",
            "17業種コード": "16", "17業種区分": "金融（除く銀行）",
            "規模コード": "6", "規模区分": "TOPIX Small 1",
        }
    ]
    jpx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(jpx_path, index=False)

    with pytest.raises(SafeAbortError) as exc_info:
        run_forward_test_day(
            base_config, filter_config, t0,
            strategy_path, tmp_path / "fetch_manifest.json",
            tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
            tmp_path / "performance_log.jsonl", jpx_path,
        )
    assert exc_info.value.reason == "UNIVERSE_DATA_INCOMPLETE"


def test_run_forward_test_day_market_data_unavailable_aborts(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    """SAFE_ABORT: a failed TOPIX Proxy fetch must abort rather than
    silently computing Market Regime (and therefore Signal Log entries)
    against no market data at all.
    """
    t0 = date(2026, 8, 20)
    strategy_path = _setup_manifest(base_config, tmp_path, t0)
    jpx_path = tmp_path / "jpx.xls"
    _write_fake_jpx_xlsx(jpx_path, ["T0"])

    n = 300
    df = make_synthetic_ohlcv(n, seed=1, ticker="T0")
    offset_days = (pd.Timestamp(t0) - pd.Timestamp(df["date"].iloc[-1])).days
    df["date"] = df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))

    class FailingMarketIndexProvider(FakeMarketIndexProvider):
        def fetch_index(self, start: date, end: date) -> FetchResult:
            return FetchResult("TOPIX", FetchStatus.FAILED, None, error="market data down")

    provider = FakeProvider({"T0": df})
    index_provider = FailingMarketIndexProvider(pd.DataFrame())

    with pytest.raises(SafeAbortError) as exc_info:
        run_forward_test_day(
            base_config, filter_config, t0,
            strategy_path, tmp_path / "fetch_manifest.json",
            tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
            tmp_path / "performance_log.jsonl", jpx_path,
            provider=provider, index_provider=index_provider,
        )
    assert exc_info.value.reason == "MARKET_DATA_UNAVAILABLE"


def test_run_forward_test_day_stale_threshold_exceeded_aborts(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    """SAFE_ABORT: if most of the Universe's fetch didn't actually reach
    run_date (e.g. a Provider-wide outage returning old cached data),
    the run must abort rather than silently proceeding on stale data.
    """
    t0 = date(2026, 8, 20)
    strategy_path = _setup_manifest(base_config, tmp_path, t0)
    tickers = ["T0", "T1", "T2"]
    jpx_path = tmp_path / "jpx.xls"
    _write_fake_jpx_xlsx(jpx_path, tickers)

    n = 300
    responses = {t: make_synthetic_ohlcv(n, seed=100 + i, ticker=t) for i, t in enumerate(tickers)}
    stale_cutoff = t0 - timedelta(days=5)  # every ticker's fetch stops 5 days short of t0
    for t, df in responses.items():
        offset_days = (pd.Timestamp(stale_cutoff) - pd.Timestamp(df["date"].iloc[-1])).days
        df["date"] = df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))
    market_df = make_synthetic_ohlcv(n, seed=999, ticker="TOPIX")
    offset_days = (pd.Timestamp(t0) - pd.Timestamp(market_df["date"].iloc[-1])).days
    market_df["date"] = market_df["date"].apply(lambda d: d + pd.Timedelta(days=offset_days))

    provider = FakeProvider(responses)
    index_provider = FakeMarketIndexProvider(market_df)

    with pytest.raises(SafeAbortError) as exc_info:
        run_forward_test_day(
            base_config, filter_config, t0,
            strategy_path, tmp_path / "fetch_manifest.json",
            tmp_path / "portfolio.json", tmp_path / "signal_log.jsonl",
            tmp_path / "performance_log.jsonl", jpx_path,
            provider=provider, index_provider=index_provider,
        )
    assert exc_info.value.reason == "STALE_THRESHOLD_EXCEEDED"


def test_run_forward_test_day_portfolio_corruption_aborts(
    base_config: AppConfig, filter_config: UniverseFilterConfig, tmp_path: Path
) -> None:
    """SAFE_ABORT: an unparseable Paper Portfolio file must abort
    rather than silently starting over from a fresh empty portfolio,
    which would quietly erase prior Forward Test history.
    """
    t0 = date(2026, 8, 20)
    strategy_path = _setup_manifest(base_config, tmp_path, t0)
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SafeAbortError) as exc_info:
        run_forward_test_day(
            base_config, filter_config, t0,
            strategy_path, tmp_path / "fetch_manifest.json",
            portfolio_path, tmp_path / "signal_log.jsonl",
            tmp_path / "performance_log.jsonl", tmp_path / "jpx.xls",
        )
    assert exc_info.value.reason == "PORTFOLIO_STATE_CORRUPTION"


def test_forward_test_cli_default_directories_are_isolated_from_other_phases() -> None:
    """Section 20 item 10: historical result isolation. The Forward
    Test's default working directories must never coincide with any
    directory Phase 6.5/7/8/9 already used.
    """
    from forward_test import FORWARD_TEST_ROOT

    other_phase_roots = {Path("data"), Path("data/phase7")}
    assert FORWARD_TEST_ROOT not in other_phase_roots
    assert str(FORWARD_TEST_ROOT).startswith("data")
    assert "forward_test" in str(FORWARD_TEST_ROOT)
