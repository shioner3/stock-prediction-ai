from __future__ import annotations

from datetime import date

import pandas as pd

from v2.validation.topn import compute_topn_daily_returns, summarize_topn


def _panel(rows: list[tuple[date, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "total_score": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )


def test_top_n_selects_highest_score_tickers() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", float(i), float(i) * 0.01) for i in range(10)]
    daily = compute_topn_daily_returns(_panel(rows), n=3, return_col="ret")
    assert len(daily) == 1
    # top 3 scores are T9, T8, T7 -> returns 0.09, 0.08, 0.07 -> mean 0.08
    assert daily[0].equal_weight_return is not None
    assert abs(daily[0].equal_weight_return - 0.08) < 1e-9
    assert daily[0].n_available == 3


def test_top_n_fewer_than_n_available() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", float(i), float(i) * 0.01) for i in range(2)]
    daily = compute_topn_daily_returns(_panel(rows), n=5, return_col="ret")
    assert daily[0].n_available == 2


def test_top_n_excludes_nan_score_rows() -> None:
    d = date(2024, 1, 1)
    rows = [(d, f"T{i}", float(i), float(i) * 0.01) for i in range(5)]
    df = _panel(rows)
    df.loc[4, "total_score"] = float("nan")  # T4 (highest score) excluded
    daily = compute_topn_daily_returns(df, n=3, return_col="ret")
    # remaining top 3 by score: T3, T2, T1 -> 0.03, 0.02, 0.01 -> mean 0.02
    assert abs(daily[0].equal_weight_return - 0.02) < 1e-9


def test_summarize_topn_computes_stats_over_daily_series() -> None:
    rows = []
    for day in range(5):
        d = date(2024, 1, 1 + day)
        rows.extend((d, f"T{i}", float(i), 0.01) for i in range(5))
    daily = compute_topn_daily_returns(_panel(rows), n=2, return_col="ret")
    result = summarize_topn(daily, n=2, window_days=5)
    assert result.n == 2
    assert result.stats.n == 5
    assert abs(result.stats.mean_return - 0.01) < 1e-9
