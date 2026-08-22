"""Phase 8 section 6: Leave-One-Period-Out sensitivity analysis - "does
removing any single period (a calendar year, a BEAR episode, ...) flip
the aggregate result from PF>1 to PF<1?" This is explicitly a diagnostic
SENSITIVITY check, not an optimization: it never selects, drops, or
reweights any period based on its own result - every period is tested
for removal exactly once, and all results are reported regardless of
outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.metrics import compute_metrics


@dataclass(frozen=True)
class LeaveOneOutResult:
    period_label: str
    n_trades_removed: int
    full_sample_pf: float | None
    leave_one_out_pf: float | None
    full_sample_expectancy: float | None
    leave_one_out_expectancy: float | None
    # "PERIOD_DEPENDENT" if removing this period alone flips PF from >1
    # to <1 (spec section 6's "特定の1期間を除外しただけでPFが大幅に1未満
    # になる場合"), else "STABLE".
    classification: str


def leave_one_period_out(trades: pd.DataFrame, group_col: str) -> list[LeaveOneOutResult]:
    """trades: must already have `group_col` assigned (e.g. a "year" or
    "bear_episode_index" column added by the caller) - rows where
    group_col is null are excluded entirely (neither counted in the full
    sample nor removable), since they don't belong to any of the periods
    being tested.
    """
    valid = trades.dropna(subset=[group_col])
    if valid.empty:
        return []

    full_metrics = compute_metrics(valid)
    labels = sorted(valid[group_col].unique(), key=str)

    results = []
    for label in labels:
        mask = valid[group_col] == label
        remaining = valid[~mask]
        remaining_metrics = compute_metrics(remaining)

        classification = "STABLE"
        if (
            full_metrics.profit_factor is not None
            and full_metrics.profit_factor > 1.0
            and (remaining_metrics.profit_factor is None or remaining_metrics.profit_factor < 1.0)
        ):
            classification = "PERIOD_DEPENDENT"

        results.append(
            LeaveOneOutResult(
                period_label=str(label),
                n_trades_removed=int(mask.sum()),
                full_sample_pf=full_metrics.profit_factor,
                leave_one_out_pf=remaining_metrics.profit_factor,
                full_sample_expectancy=full_metrics.expectancy,
                leave_one_out_expectancy=remaining_metrics.expectancy,
                classification=classification,
            )
        )
    return results
