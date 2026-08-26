"""Spec section 28, with explicit emphasis on beta estimation:
"特にBeta-adjusted Targetについて、beta推定期間に未来情報が混入して
いないことを重点確認". Re-runs the SAME 4 future-shock types
(`v3/leakage/shock_tests.py`, unmodified) against real Full Universe
data, exactly like `v3.validation.leakage_check.run_full_universe_
shock_checks()` already does for the underlying Feature panel - but
additionally rebuilds `compute_rolling_beta()`/`compute_residual_
targets()` (this Phase's own new code) from each shocked dataset and
compares BOTH `beta` and every `RESIDUAL_TARGET_COLUMNS` entry for rows
dated <= cutoff. If beta's rolling window ever read a future row, or a
residual Target's `market_forward`/sector-mean term ever picked up a
shocked future value, this would catch it - a targeted extension of the
already-proven Feature-level guarantee, not a re-derivation of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import numpy as np
import pandas as pd

from storage.parquet_store import load_ohlcv, save_ohlcv
from v3.config.loader import V3Config
from v3.dataset import build_v3_dataset
from v3.leakage.shock_tests import (
    random_perturb_after,
    shock_index_after,
    shock_price_after,
    shock_volume_after,
)
from v3.residual.targets import RESIDUAL_TARGET_COLUMNS, compute_residual_targets
from v3.robustness.aux_panel import attach_sector_and_scale
from v3.robustness.beta import compute_rolling_beta
from v3.targets.registry import HORIZONS

SAMPLE_SIZE = 100
SAMPLE_SEED = 502  # deliberately different from v3/validation/leakage_check.py's 501


@dataclass(frozen=True)
class ResidualShockCheckResult:
    label: str
    n_rows_compared: int
    n_mismatches: int
    passed: bool


def _sample_tickers(tickers: list[str], n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    if len(tickers) <= n:
        return list(tickers)
    return sorted(rng.choice(tickers, size=n, replace=False).tolist())


def _build_beta_and_residuals(dataset: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    beta_panel = compute_rolling_beta(dataset)
    return compute_residual_targets(dataset, beta_panel, sector_map)


def _compare_at_cutoff(
    baseline: pd.DataFrame, shocked: pd.DataFrame, cols: list[str], cutoff: date_type, label: str,
) -> tuple[int, int]:
    baseline_early = (
        baseline[baseline["date"] <= cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    shocked_early = (
        shocked[shocked["date"] <= cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    common = set(baseline_early["ticker"]) & set(shocked_early["ticker"])
    baseline_early = baseline_early[baseline_early["ticker"].isin(common)]
    shocked_early = shocked_early[shocked_early["ticker"].isin(common)]

    n_mismatches = 0
    n_compared = 0
    for col in cols:
        b = baseline_early[col].to_numpy(dtype=float)
        s = shocked_early[col].to_numpy(dtype=float)
        mismatch = ~((b == s) | (np.isnan(b) & np.isnan(s)))
        n_mismatches += int(mismatch.sum())
        n_compared += len(b)
    return n_mismatches, n_compared


def _compare_before_cutoff(
    baseline: pd.DataFrame, shocked: pd.DataFrame, cutoff: date_type, target_safe_cutoff: date_type,
    label: str,
) -> ResidualShockCheckResult:
    """`beta` is purely backward-looking - compared up to `cutoff` itself.
    The residual Target columns read FORWARD data (Close[t+h]), so a row
    dated close to `cutoff` legitimately changes once dates > cutoff are
    shocked - NOT leakage. Those are instead compared only up to
    `target_safe_cutoff` (`cutoff` minus the largest registered Horizon's
    worth of TRADING rows - an embargo, the exact same concept the WFO
    train/test split already uses elsewhere in this project).
    """
    beta_mismatches, beta_compared = _compare_at_cutoff(baseline, shocked, ["beta"], cutoff, label)
    target_mismatches, target_compared = _compare_at_cutoff(
        baseline, shocked, RESIDUAL_TARGET_COLUMNS, target_safe_cutoff, label
    )
    n_mismatches = beta_mismatches + target_mismatches
    n_compared = beta_compared + target_compared
    return ResidualShockCheckResult(
        label=label, n_rows_compared=n_compared, n_mismatches=n_mismatches, passed=n_mismatches == 0
    )


def run_residual_shock_checks(
    tickers: list[str], config: V3Config, cutoff: date_type, work_dir: Path,
    baseline_with_residuals: pd.DataFrame,
) -> list[ResidualShockCheckResult]:
    """baseline_with_residuals: `compute_residual_targets(baseline_dataset,
    compute_rolling_beta(baseline_dataset), sector_map)` - the ALREADY-
    BUILT baseline, never rebuilt here (same "don't recompute what's
    already built" discipline `v3.validation.leakage_check` established).
    """
    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]]
    sampled = _sample_tickers(tickers)
    results: list[ResidualShockCheckResult] = []

    trading_dates = sorted(baseline_with_residuals["date"].unique())
    cutoff_idx = max(i for i, d in enumerate(trading_dates) if d <= cutoff)
    safe_idx = max(0, cutoff_idx - max(HORIZONS))
    target_safe_cutoff = trading_dates[safe_idx]

    def _run_sampled_shock(
        label: str, shock_fn: Callable[[pd.DataFrame, date_type], pd.DataFrame],
    ) -> None:
        shock_dir = work_dir / label
        for ticker in [*tickers, "TOPIX"]:
            ohlcv = load_ohlcv(ticker, config.source_processed_dir)
            shocked = shock_fn(ohlcv, cutoff) if ticker in sampled else ohlcv
            save_ohlcv(shocked, ticker, shock_dir)
        shock_config = config.model_copy(update={"source_processed_dir": shock_dir})
        shocked_dataset = build_v3_dataset(tickers, shock_config)
        shocked_with_residuals = _build_beta_and_residuals(shocked_dataset, sector_map)
        results.append(
            _compare_before_cutoff(
                baseline_with_residuals, shocked_with_residuals, cutoff, target_safe_cutoff, label
            )
        )

    _run_sampled_shock("A_price_shock", shock_price_after)

    index_dir = work_dir / "B_index_shock"
    for ticker in tickers:
        save_ohlcv(load_ohlcv(ticker, config.source_processed_dir), ticker, index_dir)
    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    save_ohlcv(shock_index_after(topix, cutoff), "TOPIX", index_dir)
    index_config = config.model_copy(update={"source_processed_dir": index_dir})
    index_shocked_dataset = build_v3_dataset(tickers, index_config)
    index_shocked_with_residuals = _build_beta_and_residuals(index_shocked_dataset, sector_map)
    results.append(
        _compare_before_cutoff(
            baseline_with_residuals, index_shocked_with_residuals, cutoff, target_safe_cutoff,
            "B_index_shock",
        )
    )

    _run_sampled_shock("C_volume_shock", shock_volume_after)
    _run_sampled_shock("D_random_perturbation", lambda o, c: random_perturb_after(o, c, seed=19))

    return results
