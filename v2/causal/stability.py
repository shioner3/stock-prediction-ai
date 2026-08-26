"""Cross-sectional Stability of the Q1 effect (spec section 27): is Q1's
negative-relationship pattern a broadly stable daily phenomenon, or driven
by a handful of extreme days?

Gini coefficient is a new, small, generic computation (no existing V1/V2
primitive computes one) - applied to the MAGNITUDE of days where Q1's own
daily mean return fell below Q1's overall mean ("loss days" relative to
Q1's own average, spec's "Worst day" framing), since a textbook Gini
coefficient requires non-negative values and a Forward Return series is
signed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd


def _gini(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = float(sorted_vals.sum())
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n)


@dataclass(frozen=True)
class DailyStabilityResult:
    window_days: int
    n_days: int
    median_daily_return: float | None
    mean_daily_return: float | None
    worst_day: date_type | None
    worst_day_return: float | None
    best_day: date_type | None
    best_day_return: float | None
    positive_day_ratio: float | None
    gini_of_below_average_days: float | None


def compute_daily_stability(
    scored: pd.DataFrame,
    return_col: str,
    window_days: int,
    bucket_col: str = "score_bucket",
    bucket_label: str = "Q1",
    date_col: str = "date",
) -> DailyStabilityResult:
    subset = scored[scored[bucket_col] == bucket_label].dropna(subset=[return_col])
    daily_mean = subset.groupby(date_col)[return_col].mean()
    n_days = len(daily_mean)
    if n_days == 0:
        return DailyStabilityResult(window_days, 0, None, None, None, None, None, None, None, None)

    below_average = daily_mean[daily_mean < daily_mean.mean()]
    below_average_magnitude = (daily_mean.mean() - below_average).to_numpy()

    return DailyStabilityResult(
        window_days=window_days,
        n_days=n_days,
        median_daily_return=float(daily_mean.median()),
        mean_daily_return=float(daily_mean.mean()),
        worst_day=daily_mean.idxmin(),
        worst_day_return=float(daily_mean.min()),
        best_day=daily_mean.idxmax(),
        best_day_return=float(daily_mean.max()),
        positive_day_ratio=float((daily_mean > 0).mean()),
        gini_of_below_average_days=_gini(below_average_magnitude),
    )
