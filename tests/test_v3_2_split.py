from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from v3.dataset import time_split


def _panel(n_days: int = 40) -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame({"date": dates, "ticker": ["T0"] * n_days, "x": range(n_days)})


def test_time_split_train_is_strictly_before_or_equal_train_end() -> None:
    panel = _panel()
    train_end = date(2024, 1, 20)
    train, _test = time_split(panel, train_end=train_end, test_start=date(2024, 2, 10))
    assert (train["date"] <= train_end).all()


def test_time_split_test_is_strictly_after_or_equal_test_start() -> None:
    panel = _panel()
    test_start = date(2024, 2, 10)
    _train, test = time_split(panel, train_end=date(2024, 1, 20), test_start=test_start)
    assert (test["date"] >= test_start).all()


def test_time_split_embargo_gap_is_excluded_from_both() -> None:
    panel = _panel()
    train_end = date(2024, 1, 20)
    test_start = date(2024, 1, 25)
    train, test = time_split(panel, train_end=train_end, test_start=test_start)
    embargo_dates = panel[(panel["date"] > train_end) & (panel["date"] < test_start)]["date"]
    assert not embargo_dates.empty  # sanity: there really is a gap to exclude
    assert not any(d in set(train["date"]) for d in embargo_dates)
    assert not any(d in set(test["date"]) for d in embargo_dates)


def test_time_split_never_mutates_input() -> None:
    panel = _panel()
    before = panel.copy(deep=True)
    time_split(panel, train_end=date(2024, 1, 20), test_start=date(2024, 2, 1))
    assert panel.equals(before)
