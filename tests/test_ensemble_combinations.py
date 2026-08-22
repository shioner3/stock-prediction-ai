from __future__ import annotations

import pandas as pd

from ensemble.combinations import aggregate_combinations, compute_pairwise_cooccurrence


def _df(long_signals_list: list[tuple[str, ...]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "long_signals": long_signals_list,
            "short_signals": [() for _ in long_signals_list],
        }
    )


def test_aggregate_combinations_ignores_single_signal_rows() -> None:
    # aggregate_combinations trusts its input tuples are already sorted
    # (the real convention from ensemble.signal_count.aggregate_signal_
    # counts, which builds every combo via tuple(sorted(...))).
    df = _df([("long_pullback",), ("long_ma_rebound", "long_pullback")])
    combos = aggregate_combinations(df, "LONG", min_sample=1)
    assert len(combos) == 1
    assert combos[0].signals == ("long_ma_rebound", "long_pullback")
    assert combos[0].n_occurrences == 1


def test_aggregate_combinations_counts_repeated_combos() -> None:
    combo = ("long_ma_rebound", "long_pullback")
    df = _df([combo, combo, combo, ("long_breakout", "long_volume_breakout")])
    combos = aggregate_combinations(df, "LONG", min_sample=1)
    assert combos[0].signals == combo
    assert combos[0].n_occurrences == 3
    assert combos[0].sufficient_sample is True


def test_aggregate_combinations_flags_insufficient_sample() -> None:
    combo = ("long_ma_rebound", "long_pullback")
    df = _df([combo, combo])
    combos = aggregate_combinations(df, "LONG", min_sample=30)
    assert combos[0].n_occurrences == 2
    assert combos[0].sufficient_sample is False


def test_aggregate_combinations_empty_df() -> None:
    assert aggregate_combinations(pd.DataFrame(), "LONG") == []


def test_aggregate_combinations_sorted_by_frequency_desc() -> None:
    common = ("a", "b")
    rare = ("c", "d")
    df = _df([common, common, common, rare])
    combos = aggregate_combinations(df, "LONG", min_sample=1)
    assert combos[0].signals == common
    assert combos[1].signals == rare


def test_pairwise_cooccurrence_jaccard() -> None:
    # a and b always co-occur (perfect overlap) -> jaccard 1.0
    # a and c never co-occur -> not in results at all
    df = _df([("a", "b"), ("a", "b"), ("a", "b")])
    results = compute_pairwise_cooccurrence(df, "LONG")
    assert len(results) == 1
    pair = results[0]
    assert {pair.signal_a, pair.signal_b} == {"a", "b"}
    assert pair.n_both == 3
    assert pair.jaccard == 1.0


def test_pairwise_cooccurrence_partial_overlap() -> None:
    df = _df([("a", "b"), ("a",), ("b",)])
    results = compute_pairwise_cooccurrence(df, "LONG")
    assert len(results) == 1
    pair = results[0]
    assert pair.n_a == 2  # "a" appears in row 1 and row 2
    assert pair.n_b == 2  # "b" appears in row 1 and row 3
    assert pair.n_both == 1
    assert pair.jaccard == 1 / 3


def test_pairwise_cooccurrence_no_pairs() -> None:
    df = _df([("a",), ("b",)])
    assert compute_pairwise_cooccurrence(df, "LONG") == []
