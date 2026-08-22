from __future__ import annotations

import numpy as np

from backtest.bootstrap import bootstrap_ci, bootstrap_diff_ci
from config.loader import BootstrapConfig


def test_deterministic_same_seed_gives_identical_result() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.015, -0.01, 0.02, 0.005])
    config = BootstrapConfig(n_resamples=2000, seed=7, confidence_level=0.95)
    a = bootstrap_ci(values, "mean_return", config)
    b = bootstrap_ci(values, "mean_return", config)
    assert a == b


def test_different_seed_can_give_different_result() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.015, -0.01, 0.02, 0.005])
    a = bootstrap_ci(values, "mean_return", BootstrapConfig(n_resamples=2000, seed=1))
    b = bootstrap_ci(values, "mean_return", BootstrapConfig(n_resamples=2000, seed=2))
    assert a.ci_low != b.ci_low or a.ci_high != b.ci_high


def test_point_estimate_matches_plain_mean() -> None:
    values = np.array([0.01, 0.02, 0.03, -0.01])
    result = bootstrap_ci(values, "mean_return", BootstrapConfig(n_resamples=1000, seed=1))
    assert np.isclose(result.point_estimate, np.mean(values))


def test_ci_bounds_the_point_estimate_for_symmetric_data() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.01, scale=0.02, size=500)
    result = bootstrap_ci(values, "mean_return", BootstrapConfig(n_resamples=5000, seed=5))
    assert result.ci_low < result.point_estimate < result.ci_high


def test_narrower_ci_with_more_data_same_variance() -> None:
    rng = np.random.default_rng(1)
    small = rng.normal(0.01, 0.05, size=20)
    large = rng.normal(0.01, 0.05, size=2000)
    config = BootstrapConfig(n_resamples=5000, seed=9)
    small_result = bootstrap_ci(small, "mean_return", config)
    large_result = bootstrap_ci(large, "mean_return", config)
    assert (large_result.ci_high - large_result.ci_low) < (
        small_result.ci_high - small_result.ci_low
    )


def test_empty_input_gives_nan_result() -> None:
    result = bootstrap_ci(np.array([]), "mean_return", BootstrapConfig())
    assert np.isnan(result.point_estimate)
    assert np.isnan(result.ci_low)
    assert result.n_observations == 0


def test_profit_factor_statistic_all_wins_gives_inf_point_estimate() -> None:
    values = np.array([0.01, 0.02, 0.03])
    result = bootstrap_ci(values, "profit_factor", BootstrapConfig(n_resamples=1000, seed=1))
    assert result.point_estimate == float("inf")


def test_expectancy_equals_mean_return_statistic() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.01])
    config = BootstrapConfig(n_resamples=1000, seed=3)
    mean_result = bootstrap_ci(values, "mean_return", config)
    expectancy_result = bootstrap_ci(values, "expectancy", config)
    assert mean_result.point_estimate == expectancy_result.point_estimate
    assert mean_result.ci_low == expectancy_result.ci_low


# --- bootstrap_diff_ci (Phase 6.5 section 21: Q5-Q1 spread CI) --------------


def test_diff_point_estimate_matches_plain_mean_difference() -> None:
    high = np.array([0.05, 0.06, 0.04])
    low = np.array([0.01, -0.01, 0.02])
    result = bootstrap_diff_ci(high, low, BootstrapConfig(n_resamples=1000, seed=1))
    assert np.isclose(result.point_estimate, np.mean(high) - np.mean(low))


def test_diff_deterministic_same_seed_gives_identical_result() -> None:
    high = np.array([0.05, 0.06, 0.04, 0.03])
    low = np.array([0.01, -0.01, 0.02, 0.0])
    config = BootstrapConfig(n_resamples=2000, seed=7, confidence_level=0.95)
    a = bootstrap_diff_ci(high, low, config)
    b = bootstrap_diff_ci(high, low, config)
    assert a == b


def test_diff_ci_bounds_point_estimate_when_samples_clearly_differ() -> None:
    rng = np.random.default_rng(0)
    high = rng.normal(loc=0.05, scale=0.01, size=200)
    low = rng.normal(loc=0.01, scale=0.01, size=200)
    result = bootstrap_diff_ci(high, low, BootstrapConfig(n_resamples=5000, seed=5))
    assert result.ci_low < result.point_estimate < result.ci_high
    assert result.ci_low > 0  # samples are far enough apart that 0 is excluded


def test_diff_empty_sample_gives_nan_result() -> None:
    result = bootstrap_diff_ci(np.array([]), np.array([0.01, 0.02]), BootstrapConfig())
    assert np.isnan(result.point_estimate)
    assert np.isnan(result.ci_low)
    assert result.n_observations_high == 0
    assert result.n_observations_low == 2


def test_diff_nan_values_are_dropped_from_each_sample() -> None:
    high = np.array([0.05, np.nan, 0.06])
    low = np.array([0.01, 0.02, np.nan])
    result = bootstrap_diff_ci(high, low, BootstrapConfig(n_resamples=500, seed=2))
    assert result.n_observations_high == 2
    assert result.n_observations_low == 2


# --- chunked resampling (Phase 11 fix: long_momentum_continuation's very
# large forward-return population OOM'd the old single-shot (n_resamples,
# n) allocation - see backtest/bootstrap.py's _resample_stat_chunked) ----


def test_chunking_gives_bit_identical_result_to_a_forced_smaller_budget(
    monkeypatch,
) -> None:
    """A numpy Generator's sequential integer stream is unaffected by how
    many calls it's split across (same seed, same total count, same
    order) - proving this empirically for a case that legitimately spans
    multiple chunks under a tiny forced budget, and must still match the
    single-chunk (default budget) result exactly.
    """
    import backtest.bootstrap as bootstrap_module

    values = np.random.default_rng(3).normal(0.01, 0.02, size=137)
    config = BootstrapConfig(n_resamples=953, seed=11, confidence_level=0.9)

    baseline = bootstrap_ci(values, "mean_return", config)

    monkeypatch.setattr(bootstrap_module, "_CHUNK_ELEMENT_BUDGET", 137 * 10)  # ~10 rows/chunk
    chunked = bootstrap_ci(values, "mean_return", config)

    assert chunked == baseline


def test_diff_chunking_gives_bit_identical_result_to_a_forced_smaller_budget(
    monkeypatch,
) -> None:
    import backtest.bootstrap as bootstrap_module

    rng = np.random.default_rng(4)
    high = rng.normal(0.02, 0.01, size=89)
    low = rng.normal(0.01, 0.01, size=61)
    config = BootstrapConfig(n_resamples=811, seed=12, confidence_level=0.9)

    baseline = bootstrap_diff_ci(high, low, config)

    monkeypatch.setattr(bootstrap_module, "_CHUNK_ELEMENT_BUDGET", 89 * 10)
    chunked = bootstrap_diff_ci(high, low, config)

    assert chunked == baseline


def test_large_population_does_not_allocate_full_resample_matrix_at_once() -> None:
    """The exact scenario that OOM'd in production: a Signal whose
    per-row forward-return population is very large (hundreds of
    thousands of observations) combined with n_resamples=10_000 must
    stay well under a full (n_resamples, n) float64 allocation (which
    would be tens of GB here) - this only completes quickly/without
    error because of the chunking, not despite it.
    """
    n = 200_000
    values = np.random.default_rng(5).normal(0.001, 0.01, size=n)
    config = BootstrapConfig(n_resamples=1000, seed=13, confidence_level=0.95)
    result = bootstrap_ci(values, "mean_return", config)
    assert result.n_observations == n
    assert result.ci_low < result.point_estimate < result.ci_high
