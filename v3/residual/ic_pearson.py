"""Spec section 10/14: Pearson IC alongside Rank IC. `v2/validation/ic.py`
only computes Spearman (rank-based) IC - no existing V1/V2/V3 primitive
computes a per-day PEARSON (raw-value) cross-sectional correlation, so
this is genuinely new, but reuses `v2.validation.ic`'s own `DailyIC`/
`ICSummary` dataclasses and `summarize_ic()` aggregation UNCHANGED -
only the per-day correlation itself (`np.corrcoef` on raw values instead
of on ranks) is new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v2.validation.ic import DailyIC, ICSummary, summarize_ic


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None
    corr = np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1]
    return float(corr) if corr == corr else None


def compute_daily_pearson_ic(
    panel: pd.DataFrame, score_col: str, return_col: str, date_col: str = "date"
) -> list[DailyIC]:
    results: list[DailyIC] = []
    for day, group in panel.groupby(date_col, sort=True):
        valid = group[[score_col, return_col]].dropna()
        results.append(
            DailyIC(date=day, ic=_pearson(valid[score_col], valid[return_col]), n=len(valid))
        )
    return results


def summarize_pearson_ic(
    panel: pd.DataFrame, window_days: int, score_col: str = "prediction",
    return_col: str = "actual", date_col: str = "date",
) -> ICSummary:
    daily = compute_daily_pearson_ic(panel, score_col, return_col, date_col)
    return summarize_ic(daily, window_days)
