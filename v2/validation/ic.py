"""Cross-sectional Information Coefficient (spec section 8): per-day
Spearman correlation between V2 Score and Forward Return, aggregated
across days. Genuinely NEW - V1 has no equivalent (its Signals are
discrete triggers, not a continuous cross-sectional ranking), so this
is not a reuse of any existing primitive.

Spearman is computed as Pearson correlation on RANK-transformed values
(the standard, exact definition of Spearman's rho) via pandas' rank()
+ numpy's corrcoef() - the same "rank then corrcoef" pattern
pipeline/run_phase13_conditional_analysis.py's own Score rank-
correlation already uses - rather than scipy.stats.spearmanr(), since
scipy is not a project dependency and this project avoids adding new
dependencies without a specific need (the built-in rank+corrcoef
approach is numerically identical for data with no ties beyond
double-precision differences, and pandas' rank() already handles ties
via average-rank, matching scipy's default tie-handling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyIC:
    date: object
    ic: float | None  # NaN/None if fewer than 2 valid (score, return) pairs that day
    n: int


@dataclass(frozen=True)
class ICSummary:
    window_days: int
    daily_ic: list[DailyIC]
    mean_ic: float | None
    median_ic: float | None
    std_ic: float | None
    information_ratio: float | None  # mean_ic / std_ic
    positive_ic_ratio: float | None  # fraction of days with ic > 0
    n_days_with_ic: int


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None
    corr = np.corrcoef(x.rank().to_numpy(), y.rank().to_numpy())[0, 1]
    return float(corr) if corr == corr else None


def compute_daily_spearman_ic(
    panel: pd.DataFrame, score_col: str, return_col: str, date_col: str = "date"
) -> list[DailyIC]:
    """One Spearman correlation per date, computed ONLY from that date's
    own rows (groupby) - never pooled across days, same isolation
    guarantee v2/ranking/cross_sectional.py's percentile ranking already
    relies on.
    """
    results: list[DailyIC] = []
    for day, group in panel.groupby(date_col, sort=True):
        valid = group[[score_col, return_col]].dropna()
        results.append(
            DailyIC(date=day, ic=_spearman(valid[score_col], valid[return_col]), n=len(valid))
        )
    return results


def summarize_ic(daily_ic: list[DailyIC], window_days: int) -> ICSummary:
    values = np.array([d.ic for d in daily_ic if d.ic is not None], dtype=float)
    if len(values) == 0:
        return ICSummary(window_days, daily_ic, None, None, None, None, None, 0)

    mean_ic = float(values.mean())
    std_ic = float(values.std(ddof=1)) if len(values) > 1 else None
    return ICSummary(
        window_days=window_days,
        daily_ic=daily_ic,
        mean_ic=mean_ic,
        median_ic=float(np.median(values)),
        std_ic=std_ic,
        information_ratio=(mean_ic / std_ic) if std_ic not in (None, 0.0) else None,
        positive_ic_ratio=float((values > 0).mean()),
        n_days_with_ic=len(values),
    )
