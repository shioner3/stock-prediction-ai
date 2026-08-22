from __future__ import annotations

from pathlib import Path

import pytest

from forward_test.performance_log import (
    PerformanceLogEntry,
    append_entry,
    compute_cumulative_return,
    compute_daily_return,
    compute_max_drawdown,
    load_entries,
)


def _entry(day: str, equity: float = 10_000_000.0) -> PerformanceLogEntry:
    return PerformanceLogEntry(
        date=day, strategy_version="v1", strategy_hash="abc123",
        universe_size=2780, data_timestamp="2026-08-20T09:00:00",
        market_regime_summary="NEUTRAL", signal_count=4, candidate_count=2780,
        open_positions=0, closed_positions=0, realized_pnl=0.0, unrealized_pnl=0.0,
        equity=equity, daily_return=None, cumulative_return=0.0, max_drawdown=0.0,
        data_quality_status="OK",
    )


def test_append_entry_writes_new_row(tmp_path: Path) -> None:
    path = tmp_path / "perf.jsonl"
    written = append_entry(path, _entry("2026-08-20"))
    assert written is True
    entries = load_entries(path)
    assert len(entries) == 1
    assert entries[0]["date"] == "2026-08-20"


def test_append_entry_same_date_is_idempotent_noop(tmp_path: Path) -> None:
    path = tmp_path / "perf.jsonl"
    append_entry(path, _entry("2026-08-20", equity=10_000_000.0))
    written_again = append_entry(path, _entry("2026-08-20", equity=99_999.0))
    assert written_again is False
    entries = load_entries(path)
    assert len(entries) == 1
    assert entries[0]["equity"] == 10_000_000.0  # untouched, not overwritten


def test_append_entry_different_dates_both_kept(tmp_path: Path) -> None:
    path = tmp_path / "perf.jsonl"
    append_entry(path, _entry("2026-08-20"))
    append_entry(path, _entry("2026-08-21"))
    entries = load_entries(path)
    assert [e["date"] for e in entries] == ["2026-08-20", "2026-08-21"]


def test_load_entries_missing_file_returns_empty() -> None:
    assert load_entries(Path("nope/does/not/exist.jsonl")) == []


def test_compute_cumulative_return() -> None:
    assert compute_cumulative_return(11_000_000.0, 10_000_000.0) == pytest.approx(0.10)
    assert compute_cumulative_return(9_000_000.0, 10_000_000.0) == pytest.approx(-0.10)


def test_compute_cumulative_return_zero_initial_capital_safe() -> None:
    assert compute_cumulative_return(100.0, 0.0) == 0.0


def test_compute_daily_return() -> None:
    assert compute_daily_return(10_100_000.0, 10_000_000.0) == pytest.approx(0.01)


def test_compute_daily_return_no_previous_gives_none() -> None:
    assert compute_daily_return(10_000_000.0, None) is None


def test_compute_max_drawdown_monotonic_rise_gives_zero() -> None:
    assert compute_max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_compute_max_drawdown_detects_worst_decline() -> None:
    # Peak 120, trough 90 -> -25%
    dd = compute_max_drawdown([100.0, 120.0, 90.0, 110.0])
    assert dd == pytest.approx(-0.25)


def test_compute_max_drawdown_empty_series() -> None:
    assert compute_max_drawdown([]) == 0.0
