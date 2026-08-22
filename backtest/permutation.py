"""Permutation Test (Phase 6 section 13): is the mean Forward Return of
Signal-triggered dates different from what you'd get by drawing a
random, equal-sized subset of dates from the full population (every
ticker/date row in the same evaluation window, triggered or not)?

Null hypothesis: "Signal発生とForward Returnの間に特別な関係はない" -
there's nothing special about the days a Signal fires, so its mean
Forward Return should be indistinguishable from a random draw of the
same size. Two-sided p-value: the fraction of null draws whose |mean|
is >= the observed |mean|.

Uses targets/forward_returns.py's Forward Return (Close[t+n]/Close[t]-1,
not the Backtest Engine's Entry=t+1 Open trade return) as the outcome
variable, per the spec's own wording ("Signal発生とForward Returnの間
の関係").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.loader import PermutationConfig

# Module-level (not a local) so tests can monkeypatch it small to force
# multiple chunks without allocating a Full-Universe-sized population -
# see tests/test_backtest_permutation.py's chunk-equivalence test.
MAX_CELLS_PER_CHUNK = 20_000_000  # ~160MB of float64 per chunk


@dataclass(frozen=True)
class PermutationResult:
    observed_mean: float
    p_value: float
    n_permutations: int
    n_signal: int
    n_population: int


def permutation_test(
    signal_returns: npt.ArrayLike, population_returns: npt.ArrayLike, config: PermutationConfig
) -> PermutationResult:
    signal_values = np.asarray(signal_returns, dtype=float)
    signal_values = signal_values[~np.isnan(signal_values)]
    population_values = np.asarray(population_returns, dtype=float)
    population_values = population_values[~np.isnan(population_values)]

    n_signal = len(signal_values)
    n_population = len(population_values)

    if n_signal == 0 or n_population < n_signal:
        return PermutationResult(
            float("nan"), float("nan"), config.n_permutations, n_signal, n_population
        )

    observed_mean = float(signal_values.mean())

    rng = np.random.default_rng(config.seed)
    # Vectorized without-replacement sampling of n_signal items from the
    # population, repeated config.n_permutations times: for each
    # permutation row, assign an i.i.d. random key to every population
    # element and take the n_signal elements with the smallest keys -
    # since the keys are i.i.d., that selection is a uniformly random
    # size-n_signal subset, and doing it this way (argpartition) avoids
    # a slow Python loop over rng.choice(replace=False).
    #
    # Phase 6.5 (Full Universe scale): a single (n_permutations,
    # n_population) call can be enormous - e.g. 10,000 x ~800,000 rows is
    # ~64GB of float64 and simply fails to allocate. Processing
    # permutations in memory-bounded CHUNKS instead is purely a memory
    # optimization, not a statistical or determinism change: a
    # np.random.Generator consumes its underlying bit stream in row-major
    # order regardless of how many rows are requested per call, so
    # drawing chunk_size rows at a time from the SAME rng produces a
    # BIT-IDENTICAL concatenated result to one big (n_permutations,
    # n_population) draw (verified directly - see
    # tests/test_backtest_permutation.py).
    chunk_size = max(1, MAX_CELLS_PER_CHUNK // max(n_population, 1))

    null_means = np.empty(config.n_permutations, dtype=float)
    start = 0
    while start < config.n_permutations:
        end = min(start + chunk_size, config.n_permutations)
        random_keys = rng.random((end - start, n_population))
        smallest_idx = np.argpartition(random_keys, n_signal - 1, axis=1)[:, :n_signal]
        null_samples = population_values[smallest_idx]
        null_means[start:end] = null_samples.mean(axis=1)
        start = end

    p_value = float(np.mean(np.abs(null_means) >= abs(observed_mean)))

    return PermutationResult(
        observed_mean=observed_mean,
        p_value=p_value,
        n_permutations=config.n_permutations,
        n_signal=n_signal,
        n_population=n_population,
    )
