"""V2 Score Q1-Q5 / Holding Period statistics (spec section 10/11/14).

A small, fresh implementation rather than a reuse of backtest/metrics.py's
BacktestMetrics: that dataclass describes actual TRADE returns (Entry at
Open[t+1], Exit after hold_days - backtest/engine.py's convention) and
has no std-dev/max-loss fields, while V2's inputs here are raw Forward
Return values (Close[t+n]/Close[t]-1, targets/forward_returns.py's
convention - price drift, never a simulated trade) and spec section 14
explicitly asks for standard deviation and maximum loss on top of the
usual win rate/PF/expectancy set. Reuses scoring.validation.
assign_quantile_buckets() (unmodified V1 code) for the Q1-Q5 split
itself - only the per-bucket statistics are new.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# A fixed, physically-motivated sanity bound - NOT tuned against any
# result - for excluding forward returns that cannot reflect a real
# trading outcome: Japanese stock daily price-limit rules ("値幅制限")
# make a >500% move over a handful of trading days essentially
# impossible without an unadjusted corporate action (split/reverse-
# split) or a raw data feed error. A single such row (a mean-return
# statistic is highly outlier-sensitive) can otherwise dominate an
# entire Q1-Q5 bucket's mean/PF - see research/phase_v2_1_report.md's
# own data-integrity finding (ticker 8303, 2023-09, an isolated ~4-row
# artifact) for the concrete case this constant exists to guard
# against. Applied identically to every forward window - never chosen
# per-window to produce a more favorable spread.
MAX_PLAUSIBLE_FORWARD_RETURN = 5.0


def exclude_implausible_returns(
    df: pd.DataFrame, return_col: str, max_abs_return: float = MAX_PLAUSIBLE_FORWARD_RETURN
) -> pd.DataFrame:
    """Drops rows whose `return_col` magnitude exceeds max_abs_return -
    an explicit, opt-in data-integrity step (see MAX_PLAUSIBLE_FORWARD_RETURN's
    own docstring). NEVER applied silently inside compute_return_stats()/
    compute_quantile_bucket_stats() - callers decide whether and when to
    use this, so a raw/unfiltered statistic is always still obtainable.
    """
    return df[df[return_col].abs() <= max_abs_return]


@dataclass(frozen=True)
class ReturnStats:
    n: int
    mean_return: float | None
    median_return: float | None
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    std_dev: float | None
    max_loss: float | None


def compute_return_stats(returns: pd.Series) -> ReturnStats:
    values = returns.dropna()
    n = len(values)
    if n == 0:
        return ReturnStats(0, None, None, None, None, None, None, None)

    wins = values[values > 0]
    losses = values[values < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())  # <= 0
    profit_factor: float | None
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = float("inf") if gross_profit > 0 else None

    return ReturnStats(
        n=n,
        mean_return=float(values.mean()),
        median_return=float(values.median()),
        win_rate=float((values > 0).mean()),
        profit_factor=profit_factor,
        expectancy=float(values.mean()),
        std_dev=float(values.std()) if n > 1 else None,
        max_loss=float(values.min()),
    )


@dataclass(frozen=True)
class QuantileBucketStats:
    bucket: str
    window_days: int
    stats: ReturnStats


def compute_quantile_bucket_stats(
    scored: pd.DataFrame, bucket_col: str, return_col: str, window_days: int
) -> list[QuantileBucketStats]:
    """One ReturnStats per Q1-Q5 bucket value present in scored[bucket_col],
    for the Forward Return window `window_days` (return_col e.g.
    "forward_return_5d") - the per-bucket, per-holding-period breakdown
    spec section 10/11 both ask for.
    """
    results = []
    for bucket, group in scored.groupby(bucket_col, observed=True):
        results.append(
            QuantileBucketStats(
                bucket=str(bucket),
                window_days=window_days,
                stats=compute_return_stats(group[return_col]),
            )
        )
    return results


def compute_q5_q1_spread(bucket_stats: list[QuantileBucketStats]) -> float | None:
    by_bucket = {b.bucket: b.stats for b in bucket_stats}
    q5, q1 = by_bucket.get("Q5"), by_bucket.get("Q1")
    if q5 is None or q1 is None or q5.mean_return is None or q1.mean_return is None:
        return None
    return q5.mean_return - q1.mean_return
