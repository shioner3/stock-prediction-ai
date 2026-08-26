"""Q1 vs Random Control (spec sections 28/29): is Q1's relative
outperformance just "the market was up that day" (any random same-sized
subset of the Universe would show the same thing), or specific to the
tickers Score actually placed in Q1?

For each date, draws a random subset of the Universe (excluding rows
without a resolved Forward Return) the SAME SIZE as that date's actual
Q1 membership - repeated across several fixed, pre-registered seeds
(chosen before running, arbitrary but fixed) rather than a single draw,
so the comparison isn't sensitive to one unlucky/lucky seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from v2.stats import ReturnStats, compute_return_stats

# Fixed before running - arbitrary values, never chosen to make the
# random control look more or less similar to Q1 after the fact.
RANDOM_CONTROL_SEEDS = (201, 202, 203, 204, 205)


@dataclass(frozen=True)
class RandomControlSeedResult:
    seed: int
    n: int
    stats: ReturnStats


@dataclass(frozen=True)
class RandomControlResult:
    window_days: int
    per_seed: list[RandomControlSeedResult]
    pooled_stats: ReturnStats


def _sample_matching_group_size(
    group: pd.DataFrame, counts: pd.Series, seed: int, date_col_value: object
) -> pd.DataFrame:
    n = int(counts.get(date_col_value, 0))
    n = min(n, len(group))
    if n == 0:
        return group.iloc[0:0]
    return group.sample(n=n, random_state=seed)


def build_random_control_sample(
    clean: pd.DataFrame,
    seed: int,
    bucket_col: str = "score_bucket",
    bucket_label: str = "Q1",
    date_col: str = "date",
) -> pd.DataFrame:
    counts = clean[clean[bucket_col] == bucket_label].groupby(date_col).size()
    frames = [
        _sample_matching_group_size(group, counts, seed, date_value)
        for date_value, group in clean.groupby(date_col)
    ]
    if not frames:
        return clean.iloc[0:0]
    return pd.concat(frames)


def run_random_control(
    clean: pd.DataFrame,
    return_col: str,
    window_days: int,
    seeds: tuple[int, ...] = RANDOM_CONTROL_SEEDS,
    bucket_col: str = "score_bucket",
    bucket_label: str = "Q1",
    date_col: str = "date",
) -> RandomControlResult:
    per_seed = []
    pooled_frames = []
    for seed in seeds:
        sample = build_random_control_sample(clean, seed, bucket_col, bucket_label, date_col)
        per_seed.append(
            RandomControlSeedResult(
                seed=seed, n=len(sample), stats=compute_return_stats(sample[return_col])
            )
        )
        pooled_frames.append(sample)
    pooled = pd.concat(pooled_frames) if pooled_frames else clean.iloc[0:0]
    return RandomControlResult(
        window_days=window_days,
        per_seed=per_seed,
        pooled_stats=compute_return_stats(pooled[return_col]),
    )
