"""Permutation Test for the Score-Forward Return relationship (spec
section 17).

Reuses backtest.permutation.permutation_test() DIRECTLY (unmodified V1
code, including its memory-bounded chunking for Full Universe scale -
spec section 17's own explicit instruction: "V2-1/Phase 12/13で修正済
みのchunk処理方式を再利用する"). Null hypothesis, applied per bucket of
interest (Q1, Q5, and each Top-N group): "this bucket's mean Forward
Return is indistinguishable from a same-sized random draw from the full
Universe population" - the exact same signal-vs-population design every
V1 Signal permutation test already uses, just applied to a Score
quantile bucket instead of a discrete Signal trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.permutation import PermutationResult, permutation_test
from config.loader import PermutationConfig


@dataclass(frozen=True)
class BucketPermutationResult:
    bucket_label: str
    window_days: int
    result: PermutationResult


def run_bucket_permutation_tests(
    scored: pd.DataFrame,
    return_col: str,
    window_days: int,
    config: PermutationConfig,
    bucket_col: str = "score_bucket",
    buckets: tuple[str, ...] = ("Q1", "Q5"),
) -> list[BucketPermutationResult]:
    population = scored[return_col].dropna().to_numpy()
    results = []
    for bucket in buckets:
        bucket_returns = scored.loc[scored[bucket_col] == bucket, return_col].dropna().to_numpy()
        results.append(
            BucketPermutationResult(
                bucket_label=bucket, window_days=window_days,
                result=permutation_test(bucket_returns, population, config),
            )
        )
    return results
