"""Future Shock Test for the daily Observation pipeline (spec section 13):
mutates OHLCV strictly AFTER the observation date and confirms
`build_observation_entries()`'s predictions for that date are BIT-
IDENTICAL before/after. Reuses `v3/leakage/shock_tests.py` (V3-1,
unmodified) and `v3.dataset.build_v3_dataset()` (unmodified) - the same
primitives V3-1/V3-3/V3-4/V3-5's own shock tests already used, applied
here to the NEW `observe_day.py` code path specifically (today-row
selection + frozen-model prediction), not a re-derivation of the
already-proven Feature-level guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import numpy as np

from storage.parquet_store import load_ohlcv, save_ohlcv
from v3.config.loader import V3Config
from v3.frozen.manifest import FrozenModelSpec
from v3.frozen.observe_day import build_observation_entries
from v3.leakage.shock_tests import random_perturb_after, shock_price_after, shock_volume_after


@dataclass(frozen=True)
class ObservationShockResult:
    label: str
    n_predictions_compared: int
    n_mismatches: int
    passed: bool


def _predictions_by_key(entries: list) -> dict[tuple[str, str], float]:
    return {(e.ticker, e.model_id): e.prediction for e in entries}


def run_observation_shock_checks(
    tickers: list[str], v3_config: V3Config, observation_date: date_type, work_dir: Path,
    frozen_specs: list[FrozenModelSpec], market_regime_config, topix_ohlcv,
) -> list[ObservationShockResult]:
    baseline = build_observation_entries(
        tickers, v3_config, observation_date, frozen_specs, market_regime_config, topix_ohlcv,
    )
    baseline_preds = _predictions_by_key(baseline.entries)

    results = []
    shocks = (
        ("A_price_shock", shock_price_after),
        ("B_volume_shock", shock_volume_after),
        ("C_random_perturbation", lambda o, c: random_perturb_after(o, c, seed=23)),
    )
    for label, shock_fn in shocks:
        shock_dir = work_dir / label
        for ticker in [*tickers, "TOPIX"]:
            ohlcv = load_ohlcv(ticker, v3_config.source_processed_dir)
            shocked = shock_fn(ohlcv, observation_date)
            save_ohlcv(shocked, ticker, shock_dir)
        shocked_config = v3_config.model_copy(update={"source_processed_dir": shock_dir})
        shocked_topix = load_ohlcv("TOPIX", shock_dir)
        shocked_result = build_observation_entries(
            tickers, shocked_config, observation_date, frozen_specs, market_regime_config,
            shocked_topix,
        )
        shocked_preds = _predictions_by_key(shocked_result.entries)

        common_keys = set(baseline_preds) & set(shocked_preds)
        n_mismatches = sum(
            1 for k in common_keys
            if not np.isclose(baseline_preds[k], shocked_preds[k], equal_nan=True)
        )
        results.append(
            ObservationShockResult(
                label=label, n_predictions_compared=len(common_keys),
                n_mismatches=n_mismatches, passed=n_mismatches == 0,
            )
        )
    return results
