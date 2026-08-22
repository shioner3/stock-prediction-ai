"""Phase 12 section 12/29: naturally-occurring Signal combination
aggregation (2-way/3-way/4+-way, within one direction) plus pairwise
co-occurrence (Jaccard) to check whether "more Signals firing" reflects
independent information or just correlated near-duplicate Signals.

Only combinations that ACTUALLY occurred in the data are enumerated -
this is never an exhaustive search over all C(6,k) hypothetical
same-direction subsets (spec section 12: "組み合わせの組み合わせを
無制限に探索してはならない"). Combinations below MIN_COMBINATION_SAMPLE
are still reported (for transparency) but flagged
sufficient_sample=False, to be treated as INSUFFICIENT_EVIDENCE
downstream (ensemble/decision.py) rather than silently dropped.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import pandas as pd

# Fixed BEFORE running any Phase 12 analysis (spec section 12: "minimum
# sample threshold は既存configの思想に合わせ、事前に固定する") - matches
# config/loader.py::MinSampleConfig.min_oos_trades' existing default of
# 30, the same "at least 30 observations" bar every prior OOS Decision
# gate in this project already uses.
MIN_COMBINATION_SAMPLE = 30


@dataclass(frozen=True)
class CombinationCount:
    direction: str
    signals: tuple[str, ...]
    combo_size: int
    n_occurrences: int
    sufficient_sample: bool


@dataclass(frozen=True)
class PairwiseCooccurrence:
    direction: str
    signal_a: str
    signal_b: str
    n_a: int
    n_b: int
    n_both: int
    jaccard: float


def aggregate_combinations(
    signal_counts_df: pd.DataFrame,
    direction: str,
    min_sample: int = MIN_COMBINATION_SAMPLE,
) -> list[CombinationCount]:
    """signal_counts_df: ensemble.signal_count.aggregate_signal_counts()
    output. Only rows with >=2 triggered same-direction Signals count as
    a "combination" (a single Signal is not a combination).
    """
    col = "long_signals" if direction == "LONG" else "short_signals"
    if col not in signal_counts_df.columns or signal_counts_df.empty:
        return []
    combos = signal_counts_df[col][signal_counts_df[col].apply(len) >= 2]
    counts = Counter(combos)
    results = [
        CombinationCount(
            direction=direction, signals=combo, combo_size=len(combo),
            n_occurrences=n, sufficient_sample=n >= min_sample,
        )
        for combo, n in counts.items()
    ]
    return sorted(results, key=lambda c: (-c.n_occurrences, c.signals))


def compute_pairwise_cooccurrence(
    signal_counts_df: pd.DataFrame, direction: str
) -> list[PairwiseCooccurrence]:
    """Jaccard similarity = n_both / (n_a + n_b - n_both) for every pair
    of Signals that co-occurred at least once - a high Jaccard for a
    pair means those two Signals are near-duplicates of each other
    (firing together almost every time either fires), which would mean
    a high Signal Count driven by that pair is not really "independent
    confirmation" (spec section 29).
    """
    col = "long_signals" if direction == "LONG" else "short_signals"
    if col not in signal_counts_df.columns:
        return []

    single_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for combo in signal_counts_df[col]:
        unique_signals = sorted(set(combo))
        for s in unique_signals:
            single_counts[s] += 1
        for a, b in itertools.combinations(unique_signals, 2):
            pair_counts[(a, b)] += 1

    results = []
    for (a, b), n_both in pair_counts.items():
        n_a, n_b = single_counts[a], single_counts[b]
        denom = n_a + n_b - n_both
        jaccard = n_both / denom if denom > 0 else 0.0
        results.append(PairwiseCooccurrence(direction, a, b, n_a, n_b, n_both, jaccard))
    return sorted(results, key=lambda p: -p.jaccard)
