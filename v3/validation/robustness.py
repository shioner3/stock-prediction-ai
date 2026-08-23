"""Bootstrap (Trade-level / Day-Cluster / Block) + Permutation Test (spec
section 19/20), on POOLED OOS predictions (every WFO window's OOS rows
concatenated - non-overlapping by construction, see `v3/validation/
windows.py`). Reuses V1's `backtest.bootstrap.bootstrap_diff_ci()` and
Phase V2-2's `v2/validation/spread_bootstrap.py::day_cluster_spread_bootstrap()`/
`block_spread_bootstrap()` (both unmodified) for the Q5-Q1 spread CI, and
V1's `backtest.day_cluster_bootstrap.day_cluster_bootstrap()`/
`backtest.block_bootstrap.block_bootstrap()`/`backtest.bootstrap.
bootstrap_ci()` (unmodified) for a single group's (e.g. Top-5 daily
return) CI. `backtest.permutation.permutation_test()` (unmodified) tests
Q5 vs population and Q1 vs population separately - the same convention
Phase V2-2/V2-3 already established, not a new joint spread-permutation
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.block_bootstrap import BlockBootstrapResult, block_bootstrap
from backtest.bootstrap import BootstrapDiffResult, BootstrapResult, bootstrap_ci, bootstrap_diff_ci
from backtest.day_cluster_bootstrap import DayClusterBootstrapResult, day_cluster_bootstrap
from backtest.permutation import PermutationResult, permutation_test
from scoring.validation import assign_quantile_buckets
from v2.validation.spread_bootstrap import (
    SpreadBootstrapResult,
    block_spread_bootstrap,
    day_cluster_spread_bootstrap,
)
from v3.validation.wfo_config import (
    BLOCK_BOOTSTRAP_CONFIG,
    DAY_CLUSTER_BOOTSTRAP_CONFIG,
    PERMUTATION_CONFIG,
    TRADE_LEVEL_BOOTSTRAP_CONFIG,
)


@dataclass(frozen=True)
class SpreadBootstrapBattery:
    trade_level: BootstrapDiffResult
    day_cluster: SpreadBootstrapResult
    block: SpreadBootstrapResult


def bootstrap_q5_q1_spread(
    predictions: pd.DataFrame, prediction_col: str = "prediction", actual_col: str = "actual",
    date_col: str = "date",
) -> SpreadBootstrapBattery:
    valid = predictions.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    q5 = valid[valid["_bucket"] == "Q5"]
    q1 = valid[valid["_bucket"] == "Q1"]

    trade_level = bootstrap_diff_ci(
        q5[actual_col].to_numpy(), q1[actual_col].to_numpy(), TRADE_LEVEL_BOOTSTRAP_CONFIG
    )
    q5_for_bootstrap = q5.rename(columns={actual_col: "return"})
    q1_for_bootstrap = q1.rename(columns={actual_col: "return"})
    day_cluster = day_cluster_spread_bootstrap(
        q5_for_bootstrap, q1_for_bootstrap, DAY_CLUSTER_BOOTSTRAP_CONFIG, date_col=date_col
    )
    block = block_spread_bootstrap(
        q5_for_bootstrap, q1_for_bootstrap, BLOCK_BOOTSTRAP_CONFIG, date_col=date_col
    )
    return SpreadBootstrapBattery(trade_level=trade_level, day_cluster=day_cluster, block=block)


@dataclass(frozen=True)
class SingleGroupBootstrapBattery:
    trade_level: BootstrapResult
    day_cluster: DayClusterBootstrapResult
    block: BlockBootstrapResult


def bootstrap_daily_return_series(
    daily_returns: pd.Series, dates: pd.Series
) -> SingleGroupBootstrapBattery:
    """daily_returns/dates: one row per trading day (e.g. a Top-5 daily
    equal-weight return series) - NOT per-ticker rows.
    """
    trades = pd.DataFrame({"return": daily_returns.to_numpy(), "signal_date": dates.to_numpy()})
    trade_level = bootstrap_ci(
        trades["return"].to_numpy(), "mean_return", TRADE_LEVEL_BOOTSTRAP_CONFIG
    )
    day_cluster = day_cluster_bootstrap(trades, "mean_return", DAY_CLUSTER_BOOTSTRAP_CONFIG)
    block = block_bootstrap(trades, "mean_return", BLOCK_BOOTSTRAP_CONFIG)
    return SingleGroupBootstrapBattery(
        trade_level=trade_level, day_cluster=day_cluster, block=block
    )


@dataclass(frozen=True)
class BucketPermutationResult:
    bucket_label: str
    result: PermutationResult


def run_bucket_permutation_tests(
    predictions: pd.DataFrame, prediction_col: str = "prediction", actual_col: str = "actual",
    config=PERMUTATION_CONFIG,
) -> list[BucketPermutationResult]:
    valid = predictions.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    population = valid[actual_col].dropna().to_numpy()
    results = []
    for bucket in ("Q1", "Q5"):
        bucket_returns = valid.loc[valid["_bucket"] == bucket, actual_col].dropna().to_numpy()
        results.append(
            BucketPermutationResult(
                bucket_label=bucket, result=permutation_test(bucket_returns, population, config)
            )
        )
    return results
