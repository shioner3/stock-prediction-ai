"""Cross-sectional (same-day, Universe-wide) percentile ranking (spec
section 8).

Every rank here is computed WITHIN a single trading day's Universe
snapshot - never across the full history - so a value's rank only ever
depends on other tickers' values on that SAME day, never on any other
day's data (forward or backward). This is what keeps ranking itself
leakage-free even though the underlying Feature values are already
point-in-time correct: pooling multiple days before ranking would let a
later day's distribution influence an earlier day's percentile, which
is a subtler form of the same forward-information problem
(tests/test_v2_leakage.py checks this directly, not just via V1's
existing per-Feature no-lookahead guarantee).

0.95 = today's top 5% of the Universe (by whichever direction counts as
"better"), 0.05 = today's bottom 5%, matching spec section 8's own
convention.
"""

from __future__ import annotations

import pandas as pd


def percentile_rank_by_day(
    df: pd.DataFrame, value_col: str, date_col: str = "date", higher_is_better: bool = True
) -> pd.Series:
    """Percentile rank in (0, 1] of `value_col`, computed independently
    within each `date_col` group. NaN values (e.g. a ticker still in
    Feature warmup) are excluded from ranking and stay NaN in the
    result - they never silently count as the worst (or best) rank.
    """
    return df.groupby(date_col)[value_col].rank(pct=True, ascending=higher_is_better)


def average_category_rank(rank_columns: pd.DataFrame) -> pd.Series:
    """Row-wise mean of several already-computed percentile-rank columns
    (one category's member features) - NaN member columns are skipped
    (pandas' default skipna mean), and a row where EVERY member is NaN
    correctly stays NaN (no category signal available that day).
    """
    return rank_columns.mean(axis=1, skipna=True)
