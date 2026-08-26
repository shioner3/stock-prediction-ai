from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from v2.validation.ic import compute_daily_spearman_ic, summarize_ic


def _panel(rows: list[tuple[date, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "score": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )


def test_perfect_positive_relationship_gives_ic_near_1() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", float(i), float(i)) for i in range(10)]
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    assert len(daily) == 1
    assert daily[0].ic is not None
    assert daily[0].ic > 0.99


def test_perfect_negative_relationship_gives_ic_near_minus_1() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", float(i), float(10 - i)) for i in range(10)]
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    assert daily[0].ic < -0.99


def test_ic_computed_independently_per_day() -> None:
    d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
    rows = (
        [(d1, f"T{i}", float(i), float(i)) for i in range(5)]
        + [(d2, f"T{i}", float(i), float(4 - i)) for i in range(5)]
    )
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    by_date = {d.date: d.ic for d in daily}
    assert by_date[d1] > 0.9
    assert by_date[d2] < -0.9


def test_single_pair_day_gives_none_ic() -> None:
    d = date(2024, 1, 1)
    rows = [(d, "T0", 1.0, 0.01)]
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    assert daily[0].ic is None
    assert daily[0].n == 1


def test_constant_score_gives_none_ic() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", 1.0, float(i)) for i in range(5)]
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    assert daily[0].ic is None


def test_nan_pairs_excluded_from_ic() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", float(i), float(i)) for i in range(5)]
    df = _panel(rows)
    df.loc[0, "ret"] = np.nan
    daily = compute_daily_spearman_ic(df, "score", "ret")
    assert daily[0].n == 4


def test_summarize_ic_aggregates_correctly() -> None:
    base = date(2024, 1, 1)
    rows = []
    for day_offset in range(5):
        d = base + timedelta(days=day_offset)
        rows.extend((d, f"T{i}", float(i), float(i)) for i in range(10))
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    summary = summarize_ic(daily, window_days=5)
    assert summary.n_days_with_ic == 5
    assert summary.mean_ic is not None
    assert summary.mean_ic > 0.99
    assert summary.positive_ic_ratio == 1.0
    assert summary.std_ic is not None


def test_summarize_ic_empty_when_no_valid_days() -> None:
    d = date(2024, 1, 1)
    rows = [(d, "T0", 1.0, 0.01)]
    daily = compute_daily_spearman_ic(_panel(rows), "score", "ret")
    summary = summarize_ic(daily, window_days=5)
    assert summary.n_days_with_ic == 0
    assert summary.mean_ic is None
    assert summary.information_ratio is None
