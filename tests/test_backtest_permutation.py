from __future__ import annotations

import numpy as np
import pytest

import backtest.permutation as permutation_module
from backtest.permutation import permutation_test
from config.loader import PermutationConfig


def test_deterministic_same_seed_gives_identical_result() -> None:
    rng = np.random.default_rng(0)
    population = rng.normal(0, 0.02, size=500)
    signal = rng.choice(population, size=30, replace=False)
    config = PermutationConfig(n_permutations=2000, seed=11)
    a = permutation_test(signal, population, config)
    b = permutation_test(signal, population, config)
    assert a == b


def test_strong_signal_gives_low_p_value() -> None:
    rng = np.random.default_rng(1)
    population = rng.normal(0.0, 0.01, size=1000)
    # A signal subset with a clearly elevated mean, far from the population's.
    signal = rng.normal(0.10, 0.01, size=40)
    config = PermutationConfig(n_permutations=5000, seed=12)
    result = permutation_test(signal, population, config)
    assert result.p_value < 0.01


def test_signal_drawn_from_population_gives_high_p_value_on_average() -> None:
    rng = np.random.default_rng(2)
    population = rng.normal(0.0, 0.02, size=1000)
    signal = rng.choice(population, size=50, replace=False)
    config = PermutationConfig(n_permutations=5000, seed=13)
    result = permutation_test(signal, population, config)
    # Not statistically guaranteed for any single draw, but with a
    # random subset of the population itself, a very low p-value would
    # be a red flag for the test's own correctness.
    assert result.p_value > 0.01


def test_p_value_is_two_sided() -> None:
    """A signal subset with a strongly NEGATIVE mean should also be
    flagged (a naive one-sided test would miss this).
    """
    rng = np.random.default_rng(3)
    population = rng.normal(0.0, 0.01, size=1000)
    signal = rng.normal(-0.10, 0.01, size=40)
    config = PermutationConfig(n_permutations=5000, seed=14)
    result = permutation_test(signal, population, config)
    assert result.p_value < 0.01


def test_insufficient_population_gives_nan() -> None:
    signal = np.array([0.01, 0.02, 0.03])
    population = np.array([0.01, 0.02])  # smaller than the signal subset
    result = permutation_test(signal, population, PermutationConfig(n_permutations=100))
    assert np.isnan(result.p_value)


def test_empty_signal_gives_nan() -> None:
    result = permutation_test(
        np.array([]), np.array([0.01, 0.02, 0.03]), PermutationConfig(n_permutations=100)
    )
    assert np.isnan(result.p_value)


def test_records_sample_sizes() -> None:
    signal = np.array([0.01, 0.02, 0.03])
    population = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    result = permutation_test(signal, population, PermutationConfig(n_permutations=500))
    assert result.n_signal == 3
    assert result.n_population == 6


# --- Phase 6.5 (Full Universe scale): chunked processing ---------------------


def test_chunked_processing_gives_identical_result_to_single_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large population/permutation count that forces MANY small
    chunks must give the exact same PermutationResult as one big chunk -
    proving the chunking introduced in Phase 6.5 (to avoid allocating a
    (n_permutations, n_population) array that can reach tens of GB at
    Full Universe scale) does not change the statistical result at all.
    """
    rng = np.random.default_rng(42)
    population = rng.normal(0.0, 0.02, size=300)
    signal = rng.choice(population, size=25, replace=False)
    config = PermutationConfig(n_permutations=400, seed=7)

    monkeypatch.setattr(permutation_module, "MAX_CELLS_PER_CHUNK", 300 * 300)  # 1 chunk
    one_chunk = permutation_test(signal, population, config)

    monkeypatch.setattr(permutation_module, "MAX_CELLS_PER_CHUNK", 300)  # forces many chunks
    many_chunks = permutation_test(signal, population, config)

    assert one_chunk == many_chunks


def test_chunk_size_never_below_one_even_for_huge_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A population far larger than MAX_CELLS_PER_CHUNK must still
    process (one permutation per chunk, not a division-by-zero or a
    zero-sized chunk) rather than crash.
    """
    rng = np.random.default_rng(1)
    population = rng.normal(0.0, 0.01, size=2000)
    signal = rng.choice(population, size=10, replace=False)

    monkeypatch.setattr(permutation_module, "MAX_CELLS_PER_CHUNK", 100)  # << n_population
    result = permutation_test(signal, population, PermutationConfig(n_permutations=50, seed=3))
    assert result.n_permutations == 50
    assert not np.isnan(result.p_value)


def test_chunking_bounds_peak_allocation_at_full_universe_scale() -> None:
    """Reproduces the exact shape that previously crashed with
    numpy._core._exceptions._ArrayMemoryError (10,000 permutations x
    ~800,000 population rows = ~64GB unchunked) - this must now complete
    without attempting anywhere near that allocation.
    """
    rng = np.random.default_rng(5)
    population = rng.normal(0.0, 0.01, size=800_000)
    signal = rng.choice(population, size=200, replace=False)
    config = PermutationConfig(n_permutations=10_000, seed=9)
    result = permutation_test(signal, population, config)
    assert result.n_population == 800_000
    assert not np.isnan(result.p_value)
