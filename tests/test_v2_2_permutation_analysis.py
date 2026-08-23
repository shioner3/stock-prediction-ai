from __future__ import annotations

import numpy as np
import pandas as pd

from config.loader import PermutationConfig
from v2.validation.permutation_analysis import run_bucket_permutation_tests


def test_permutation_detects_real_bucket_difference() -> None:
    rng = np.random.default_rng(1)
    n_per_bucket = 200
    rows = []
    for bucket, mean in (("Q1", -0.05), ("Q2", -0.01), ("Q3", 0.0), ("Q4", 0.01), ("Q5", 0.05)):
        rows.extend(
            {"score_bucket": bucket, "ret": v} for v in rng.normal(mean, 0.01, n_per_bucket)
        )
    scored = pd.DataFrame(rows)

    results = run_bucket_permutation_tests(
        scored, "ret", window_days=5, config=PermutationConfig(n_permutations=2000, seed=1)
    )
    by_bucket = {r.bucket_label: r.result for r in results}
    assert by_bucket["Q5"].p_value < 0.05
    assert by_bucket["Q1"].p_value < 0.05


def test_permutation_reproducible_with_same_seed() -> None:
    rng = np.random.default_rng(1)
    rows = [{"score_bucket": "Q1", "ret": v} for v in rng.normal(0, 0.01, 100)]
    rows += [{"score_bucket": "Q5", "ret": v} for v in rng.normal(0, 0.01, 100)]
    scored = pd.DataFrame(rows)
    config = PermutationConfig(n_permutations=500, seed=5)

    a = run_bucket_permutation_tests(scored, "ret", 5, config)
    b = run_bucket_permutation_tests(scored, "ret", 5, config)
    assert [r.result.p_value for r in a] == [r.result.p_value for r in b]


def test_empty_bucket_gives_nan_p_value() -> None:
    scored = pd.DataFrame({"score_bucket": ["Q1"], "ret": [0.01]})
    results = run_bucket_permutation_tests(
        scored, "ret", 5, PermutationConfig(n_permutations=100), buckets=("Q5",)
    )
    assert np.isnan(results[0].result.p_value)
