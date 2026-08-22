from __future__ import annotations

import numpy as np
import pandas as pd

from v2.ranking.cross_sectional import average_category_rank, percentile_rank_by_day


def _panel(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [r[0] for r in rows],
            "ticker": [r[1] for r in rows],
            "value": [r[2] for r in rows],
        }
    )


def test_percentile_rank_is_bounded_0_to_1() -> None:
    df = _panel([("D1", f"T{i}", float(i)) for i in range(10)])
    ranks = percentile_rank_by_day(df, "value")
    assert ranks.min() > 0
    assert ranks.max() <= 1.0


def test_higher_value_gets_higher_rank_when_higher_is_better() -> None:
    df = _panel([("D1", "A", 1.0), ("D1", "B", 2.0), ("D1", "C", 3.0)])
    ranks = percentile_rank_by_day(df, "value", higher_is_better=True)
    assert ranks.iloc[2] > ranks.iloc[1] > ranks.iloc[0]


def test_lower_value_gets_higher_rank_when_lower_is_better() -> None:
    df = _panel([("D1", "A", 1.0), ("D1", "B", 2.0), ("D1", "C", 3.0)])
    ranks = percentile_rank_by_day(df, "value", higher_is_better=False)
    assert ranks.iloc[0] > ranks.iloc[1] > ranks.iloc[2]


def test_ranking_is_independent_per_day() -> None:
    """A ticker's rank on day D1 must not depend on what values exist on
    day D2 - each date group is ranked in complete isolation.
    """
    df = _panel(
        [
            ("D1", "A", 1.0), ("D1", "B", 2.0),
            ("D2", "A", 100.0), ("D2", "B", 1.0), ("D2", "C", 2.0), ("D2", "D", 3.0),
        ]
    )
    ranks = percentile_rank_by_day(df, "value", higher_is_better=True)
    d1_ranks = ranks[df["date"] == "D1"].to_numpy()
    # With only 2 tickers on D1, ranks must be exactly {0.5, 1.0} -
    # unaffected by D2 having 4 tickers with very different values.
    assert set(np.round(d1_ranks, 6)) == {0.5, 1.0}


def test_nan_value_stays_nan_and_is_excluded_from_ranking() -> None:
    df = _panel([("D1", "A", 1.0), ("D1", "B", float("nan")), ("D1", "C", 3.0)])
    ranks = percentile_rank_by_day(df, "value", higher_is_better=True)
    assert pd.isna(ranks.iloc[1])
    # A and C rank among themselves only (B excluded), so C > A at 1.0/0.5.
    assert ranks.iloc[2] == 1.0
    assert ranks.iloc[0] == 0.5


def test_tied_values_get_the_same_average_rank() -> None:
    df = _panel([("D1", "A", 1.0), ("D1", "B", 1.0), ("D1", "C", 3.0)])
    ranks = percentile_rank_by_day(df, "value", higher_is_better=True)
    assert ranks.iloc[0] == ranks.iloc[1]
    assert ranks.iloc[2] > ranks.iloc[0]


def test_average_category_rank_skips_nan_members() -> None:
    members = pd.DataFrame({"f1": [0.2, np.nan], "f2": [0.8, 0.6]})
    result = average_category_rank(members)
    assert np.isclose(result.iloc[0], 0.5)
    assert np.isclose(result.iloc[1], 0.6)


def test_average_category_rank_all_nan_row_is_nan() -> None:
    members = pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})
    result = average_category_rank(members)
    assert pd.isna(result.iloc[0])
