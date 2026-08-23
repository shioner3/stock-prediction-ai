"""Full Universe re-verification of Phase V3-1's Leakage Framework (spec
section 22). Phase V3-1/V3-2's own leakage tests (`tests/test_v3_leakage.py`,
`tests/test_v3_2_leakage.py`) already proved the underlying Feature
computation is leak-free using small SYNTHETIC ticker subsets; this
module re-runs the same 4 future-shock types (`v3/leakage/shock_tests.py`,
unmodified) against REAL Full Universe data, since the small-subset tests
never touched real tickers' actual price histories.

Scoped to a SAMPLE of tickers (not all ~2,880) for the price/volume
shocks, to bound the I/O cost of rewriting shocked OHLCV Parquet files -
a deliberate, pre-run design decision (any leakage bug would manifest
per-ticker independently through the SAME Feature computation code path
already exhaustively tested in V3-1/V3-2, so a sample is sufficient to
catch a real-data-scale-dependent issue a synthetic fixture might miss,
without paying the cost of rewriting the entire Universe). The Index
(TOPIX) shock is run against the full Universe since it only requires
rewriting ONE file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import numpy as np
import pandas as pd

from storage.parquet_store import load_ohlcv, save_ohlcv
from v3.config.loader import V3Config
from v3.dataset import build_v3_dataset
from v3.features.registry import CORE_FEATURE_NAMES
from v3.leakage.shock_tests import (
    random_perturb_after,
    shock_index_after,
    shock_price_after,
    shock_volume_after,
)

SAMPLE_SIZE = 100
SAMPLE_SEED = 501


@dataclass(frozen=True)
class ShockCheckResult:
    label: str
    n_rows_compared: int
    n_mismatches: int
    passed: bool


def _sample_tickers(tickers: list[str], n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    if len(tickers) <= n:
        return list(tickers)
    return sorted(rng.choice(tickers, size=n, replace=False).tolist())


def _compare_features_before_cutoff(
    baseline: pd.DataFrame, shocked: pd.DataFrame, cutoff: date_type, label: str
) -> ShockCheckResult:
    baseline_early = (
        baseline[baseline["date"] <= cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    shocked_early = (
        shocked[shocked["date"] <= cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    common_tickers = set(baseline_early["ticker"]) & set(shocked_early["ticker"])
    baseline_early = baseline_early[baseline_early["ticker"].isin(common_tickers)]
    shocked_early = shocked_early[shocked_early["ticker"].isin(common_tickers)]

    n_mismatches = 0
    n_compared = 0
    for col in CORE_FEATURE_NAMES:
        b = baseline_early[col].to_numpy(dtype=float)
        s = shocked_early[col].to_numpy(dtype=float)
        mismatch = ~((b == s) | (np.isnan(b) & np.isnan(s)))
        n_mismatches += int(mismatch.sum())
        n_compared += len(b)
    return ShockCheckResult(
        label=label, n_rows_compared=n_compared, n_mismatches=n_mismatches, passed=n_mismatches == 0
    )


def run_full_universe_shock_checks(
    tickers: list[str], config: V3Config, cutoff: date_type, work_dir: Path, baseline: pd.DataFrame
) -> list[ShockCheckResult]:
    """baseline: the ALREADY-BUILT Full Universe dataset (same one used
    for WFO training) - never rebuilt here, matching spec section 35's
    "同じFeatureについて何度も再計算しない" discipline established since
    Phase V2-2.
    """
    sampled = _sample_tickers(tickers)

    results = []

    # A. Future price shock (sampled tickers)
    price_dir = work_dir / "shocked_price"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, config.source_processed_dir)
        shocked = shock_price_after(ohlcv, cutoff) if ticker in sampled else ohlcv
        save_ohlcv(shocked, ticker, price_dir)
    price_config = config.model_copy(update={"source_processed_dir": price_dir})
    price_shocked = build_v3_dataset(tickers, price_config)
    results.append(
        _compare_features_before_cutoff(baseline, price_shocked, cutoff, "A_price_shock")
    )

    # B. Future index (TOPIX) shock (whole Universe affected via RS/market context)
    index_dir = work_dir / "shocked_index"
    for ticker in tickers:
        save_ohlcv(load_ohlcv(ticker, config.source_processed_dir), ticker, index_dir)
    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    save_ohlcv(shock_index_after(topix, cutoff), "TOPIX", index_dir)
    index_config = config.model_copy(update={"source_processed_dir": index_dir})
    index_shocked = build_v3_dataset(tickers, index_config)
    results.append(
        _compare_features_before_cutoff(baseline, index_shocked, cutoff, "B_index_shock")
    )

    # C. Future volume shock (sampled tickers)
    volume_dir = work_dir / "shocked_volume"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, config.source_processed_dir)
        shocked = shock_volume_after(ohlcv, cutoff) if ticker in sampled else ohlcv
        save_ohlcv(shocked, ticker, volume_dir)
    volume_config = config.model_copy(update={"source_processed_dir": volume_dir})
    volume_shocked = build_v3_dataset(tickers, volume_config)
    results.append(
        _compare_features_before_cutoff(baseline, volume_shocked, cutoff, "C_volume_shock")
    )

    # D. Random future perturbation (sampled tickers)
    random_dir = work_dir / "shocked_random"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, config.source_processed_dir)
        shocked = random_perturb_after(ohlcv, cutoff, seed=17) if ticker in sampled else ohlcv
        save_ohlcv(shocked, ticker, random_dir)
    random_config = config.model_copy(update={"source_processed_dir": random_dir})
    random_shocked = build_v3_dataset(tickers, random_config)
    results.append(
        _compare_features_before_cutoff(baseline, random_shocked, cutoff, "D_random_perturbation")
    )

    return results
