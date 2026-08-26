"""Gini coefficient / Lorenz curve (spec section 7) - genuinely new, no
existing V1/V2/V3 primitive computes inequality-of-contribution this way
(`v2/validation/concentration.py::compute_concentration()` only reports
Top-K SHARE-of-total, not the full-distribution Gini index).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gini_coefficient(values: pd.Series | np.ndarray) -> float | None:
    """Gini index of a non-negative contribution series (e.g. per-day or
    per-ticker summed return contribution within Q5). Values are shifted
    to be non-negative first (Gini is only well-defined for non-negative
    quantities) - a documented, order-preserving transform, not a
    magnitude change: shifting every value by the same constant does not
    alter which days/tickers "contribute more than others," only removes
    negative contributions from breaking the standard Gini formula.
    0 = perfectly equal contribution, 1 = maximally concentrated (one
    day/ticker holds everything).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    shifted = arr - arr.min() if arr.min() < 0 else arr
    total = shifted.sum()
    if total == 0:
        return 0.0
    sorted_vals = np.sort(shifted)
    n = len(sorted_vals)
    # Standard discrete Gini via the Lorenz curve area:
    # G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n, i = 1..n
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n
    return float(gini)


def lorenz_curve(values: pd.Series | np.ndarray) -> tuple[list[float], list[float]]:
    """Returns (cumulative_population_share, cumulative_value_share),
    both starting at (0, 0) and ending at (1, 1) - the underlying curve
    `gini_coefficient()` summarizes into one number.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return [0.0], [0.0]
    shifted = arr - arr.min() if arr.min() < 0 else arr
    sorted_vals = np.sort(shifted)
    total = sorted_vals.sum()
    n = len(sorted_vals)
    cum_value = np.cumsum(sorted_vals) / total if total != 0 else np.linspace(0, 1, n)
    cum_pop = np.arange(1, n + 1) / n
    return [0.0, *cum_pop.tolist()], [0.0, *cum_value.tolist()]


def compute_contribution_ranking(
    bucket_predictions: pd.DataFrame, group_col: str, return_col: str = "actual"
) -> pd.Series:
    """Sum of `return_col` per `group_col` value (e.g. per ticker, or per
    date) within an already-filtered-to-one-bucket DataFrame, sorted
    descending - the identity-preserving counterpart to `v2.validation.
    concentration.compute_concentration()`'s anonymous Top-K SHARES,
    needed here because leave-top-K-out analysis must know WHICH
    tickers/days to exclude, not just what fraction they represent.
    """
    return bucket_predictions.groupby(group_col)[return_col].sum().sort_values(ascending=False)
