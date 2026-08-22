from __future__ import annotations

import pandas as pd

from backtest.leave_one_period_out import leave_one_period_out


def _trades(rows: list[tuple[str, float, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "return": [r[1] for r in rows],
            "period": [r[2] for r in rows],
        }
    )


def test_leave_one_out_stable_when_pf_stays_above_one_without_each_period() -> None:
    # 3 periods, each contributing a mix of small wins/losses that keep
    # PF > 1 in aggregate AND when any single one is removed.
    trades = _trades(
        [
            ("A", 0.02, "p1"), ("B", -0.005, "p1"),
            ("C", 0.02, "p2"), ("D", -0.005, "p2"),
            ("E", 0.02, "p3"), ("F", -0.005, "p3"),
        ]
    )
    results = leave_one_period_out(trades, "period")
    assert len(results) == 3
    for r in results:
        assert r.classification == "STABLE"
        assert r.full_sample_pf is not None and r.full_sample_pf > 1.0
        assert r.leave_one_out_pf is not None and r.leave_one_out_pf > 1.0


def test_leave_one_out_flags_period_dependent_when_removal_flips_pf_below_one() -> None:
    # All the profit is concentrated in "p1" - removing it should flip
    # PF from >1 (full sample) to <1 (without p1).
    trades = _trades(
        [
            ("A", 0.30, "p1"),  # all the profit
            ("B", -0.05, "p2"), ("C", -0.05, "p2"),
            ("D", -0.05, "p3"), ("E", -0.05, "p3"),
        ]
    )
    results = leave_one_period_out(trades, "period")
    by_label = {r.period_label: r for r in results}

    assert by_label["p1"].full_sample_pf is not None and by_label["p1"].full_sample_pf > 1.0
    assert by_label["p1"].classification == "PERIOD_DEPENDENT"
    # Removing p2 or p3 alone shouldn't flip an already-losing remainder.
    assert by_label["p2"].classification == "STABLE"
    assert by_label["p3"].classification == "STABLE"


def test_leave_one_out_n_trades_removed_matches_group_size() -> None:
    trades = _trades([("A", 0.01, "p1"), ("B", 0.02, "p1"), ("C", -0.01, "p2")])
    results = leave_one_period_out(trades, "period")
    by_label = {r.period_label: r for r in results}
    assert by_label["p1"].n_trades_removed == 2
    assert by_label["p2"].n_trades_removed == 1


def test_leave_one_out_rows_with_null_group_are_excluded_entirely() -> None:
    trades = _trades([("A", 0.05, "p1"), ("B", -0.02, None), ("C", 0.01, "p1")])
    results = leave_one_period_out(trades, "period")
    assert len(results) == 1
    assert results[0].period_label == "p1"
    # Full sample excludes the null-group row - only the 2 "p1" trades count.
    assert results[0].n_trades_removed == 2


def test_leave_one_out_empty_input_gives_empty_result() -> None:
    assert leave_one_period_out(pd.DataFrame(columns=["return", "period"]), "period") == []


def test_leave_one_out_all_rows_null_group_gives_empty_result() -> None:
    trades = _trades([("A", 0.05, None), ("B", -0.02, None)])
    assert leave_one_period_out(trades, "period") == []


def test_leave_one_out_single_period_removes_everything() -> None:
    trades = _trades([("A", 0.05, "p1"), ("B", -0.02, "p1")])
    results = leave_one_period_out(trades, "period")
    assert len(results) == 1
    assert results[0].leave_one_out_pf is None  # nothing left after removing the only period
