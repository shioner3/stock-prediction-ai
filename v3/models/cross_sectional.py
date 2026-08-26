"""Cross-sectional ranking check + Random baseline (spec section 12/13).

Reuses `scoring.validation.assign_quantile_buckets()` (V1, unmodified)
for the Q1-Q5 split and `v2.stats.compute_quantile_bucket_stats()`/
`compute_q5_q1_spread()` (V2, unmodified) for the per-bucket statistics -
the SAME generic, already-tested primitives V2's own Score validation and
Phase V2-2/V2-3 already use, applied here to a MODEL PREDICTION column
instead of V2's rule-based Score. This is a read-only import of pure
functions; neither v1 nor v2 code is modified.

spec section 12's explicit instruction: Q5 > Q1 is NOT a target to
optimize for here - whatever the buckets show is reported as-is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scoring.validation import assign_quantile_buckets
from v2.stats import QuantileBucketStats, compute_q5_q1_spread, compute_quantile_bucket_stats

RANDOM_BASELINE_SEED = 101


@dataclass(frozen=True)
class CrossSectionalResult:
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None


def evaluate_cross_sectional_ranking(
    dataset: pd.DataFrame, prediction_col: str, target_col: str, window_days: int
) -> CrossSectionalResult:
    """dataset must have `date`/`ticker`/prediction_col/target_col for
    every row already scored by a model. Buckets are assigned WITHIN this
    dataset's OWN prediction distribution (not per-day, matching V2-1/
    V2-2's own pooled-quantile convention - see v2/ranking/score.py and
    scoring/validation.py's own docstrings for why a pooled quantile is
    an accepted, documented design choice in this project already).
    """
    valid = dataset.dropna(subset=[prediction_col, target_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    bucket_stats = compute_quantile_bucket_stats(valid, "_bucket", target_col, window_days)
    return CrossSectionalResult(
        bucket_stats=bucket_stats, q5_q1_spread=compute_q5_q1_spread(bucket_stats)
    )


def add_random_baseline_column(
    dataset: pd.DataFrame, seed: int = RANDOM_BASELINE_SEED, column_name: str = "_random_baseline"
) -> pd.DataFrame:
    """A fixed-seed random ranking (spec section 13) - same shape as a
    real prediction column, so evaluate_cross_sectional_ranking() can be
    reused unmodified against it for a side-by-side comparison.
    """
    rng = np.random.default_rng(seed)
    out = dataset.copy()
    out[column_name] = rng.uniform(0.0, 1.0, size=len(out))
    return out
