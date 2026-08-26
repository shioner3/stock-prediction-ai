from __future__ import annotations

from datetime import date

import pandas as pd

from pipeline.run_phase9_analysis import AUG_2024_EVENT_END, AUG_2024_EVENT_START
from v2.validation.event_year_analysis import (
    analyze_event_exclusion,
    analyze_max_contribution_day_exclusion,
    analyze_year_by_year,
    find_max_contribution_day,
)


def _scored(rows: list[tuple[date, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "score_bucket": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )


def test_find_max_contribution_day_identifies_largest_q5_day() -> None:
    rows = [
        (date(2024, 1, 1), "T0", "Q5", 0.01),
        (date(2024, 1, 2), "T0", "Q5", 0.50),  # dominant day
        (date(2024, 1, 2), "T1", "Q5", 0.30),
        (date(2024, 1, 3), "T0", "Q5", 0.02),
    ]
    max_day = find_max_contribution_day(_scored(rows), "ret")
    assert max_day == date(2024, 1, 2)


def test_find_max_contribution_day_empty_bucket_returns_none() -> None:
    rows = [(date(2024, 1, 1), "T0", "Q1", 0.01)]
    assert find_max_contribution_day(_scored(rows), "ret", bucket="Q5") is None


def test_max_contribution_day_exclusion_removes_that_date() -> None:
    rows = [
        (date(2024, 1, 1), "T0", "Q1", -0.01),
        (date(2024, 1, 1), "T0", "Q5", 0.01),
        (date(2024, 1, 2), "T0", "Q1", -0.01),
        (date(2024, 1, 2), "T0", "Q5", 0.50),
        (date(2024, 1, 3), "T0", "Q1", -0.01),
        (date(2024, 1, 3), "T0", "Q5", 0.02),
    ]
    result = analyze_max_contribution_day_exclusion(_scored(rows), "ret", window_days=5)
    assert result.n == 4  # 2024-01-02's 2 rows excluded


def test_aug_2024_event_exclusion() -> None:
    rows = [
        (AUG_2024_EVENT_START, "T0", "Q5", 0.50),
        (AUG_2024_EVENT_END, "T0", "Q5", 0.30),
        (date(2024, 9, 1), "T0", "Q5", 0.01),
        (date(2024, 9, 1), "T0", "Q1", -0.01),
    ]
    results = analyze_event_exclusion(_scored(rows), "ret", window_days=5)
    assert results["full_period"].n == 4
    assert results["excl_2024_08"].n == 2


def test_year_by_year_groups_correctly() -> None:
    rows = [
        (date(2022, 1, 1), "T0", "Q5", 0.01),
        (date(2023, 1, 1), "T0", "Q5", 0.02),
        (date(2023, 6, 1), "T0", "Q1", -0.01),
        (date(2024, 1, 1), "T0", "Q5", 0.03),
    ]
    results = analyze_year_by_year(_scored(rows), "ret", window_days=5)
    assert set(results.keys()) == {2022, 2023, 2024}
    assert results[2023].n == 2


def test_year_by_year_skips_years_with_no_data() -> None:
    rows = [(date(2022, 1, 1), "T0", "Q5", 0.01)]
    results = analyze_year_by_year(_scored(rows), "ret", window_days=5)
    assert set(results.keys()) == {2022}
