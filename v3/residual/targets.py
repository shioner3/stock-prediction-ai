"""Spec sections 5/6: Target C (Beta-adjusted Residual Return) and
Target D (Sector-relative Return), for all 4 Horizons (5/10/15/20d).
Target A (Raw) and Target B (TOPIX-relative) need NO new computation -
both are already-computed, frozen V3-1 Target Registry columns
(`target_raw_*d`/`target_topix_relative_*d`, all 4 horizons already
exist - `v3/targets/registry.py`).

Beta is `v3.robustness.beta.compute_rolling_beta()`, REUSED UNCHANGED
(spec section 5's explicit instruction: "V3-4で使用したBeta定義がある
場合は完全再利用") - a single 60-day trailing estimate, independent of
Horizon (a stock's estimated market sensitivity does not change with
how far forward the Target looks). `market_forward_h` is recovered
algebraically as `raw_h - topix_relative_h` (identical for every ticker
on a date, by `target_topix_relative_h`'s own definition), exactly the
trick `v3/robustness/market_decomposition.py` already established for
Phase V3-4's 5d-only case, generalized here to all 4 horizons.

Both raw inputs (`target_raw_h`, `target_topix_relative_h`) and the
day/sector MEAN used for Sector-relative are masked to NaN via the SAME
already-established `MAX_PLAUSIBLE_FORWARD_RETURN` bound BEFORE any
arithmetic - applied proactively from the start this Phase (not
discovered via a real-data bug this time), learning directly from Phase
V3-4's own "Bugs Discovered" section (day/sector-mean pollution, beta
explosion, and TOPIX-Proxy-artifact-date corruption all traced back to
exactly this omission).
"""

from __future__ import annotations

import pandas as pd

from v2.stats import MAX_PLAUSIBLE_FORWARD_RETURN
from v3.robustness.beta import MAX_PLAUSIBLE_BETA
from v3.targets.registry import HORIZONS, VARIANT_RAW, VARIANT_TOPIX_RELATIVE, target_column_name

VARIANT_BETA_RESIDUAL = "beta_residual"
VARIANT_SECTOR_RELATIVE = "sector_relative"


def residual_target_column_name(variant: str, horizon: int) -> str:
    return f"target_{variant}_{horizon}d"


def _plausible(series: pd.Series) -> pd.Series:
    return series.where(series.abs() <= MAX_PLAUSIBLE_FORWARD_RETURN)


def compute_residual_targets(
    dataset: pd.DataFrame, beta_panel: pd.DataFrame, sector_map: pd.DataFrame,
) -> pd.DataFrame:
    """Returns a COPY of `dataset` with 8 new columns appended
    (`target_beta_residual_{5,10,15,20}d`, `target_sector_relative_
    {5,10,15,20}d`) - never mutates or persists back into `v3/dataset.py`
    /`v3/targets/`'s own frozen output. `beta_panel`: `v3.robustness.
    beta.compute_rolling_beta(dataset)`'s own output (date/ticker/beta).
    `sector_map`: ticker/sector33 (`v2.causal.segment`, unmodified).
    """
    out = dataset.merge(beta_panel, on=["date", "ticker"], how="left")
    out["beta"] = out["beta"].where(out["beta"].abs() <= MAX_PLAUSIBLE_BETA)

    sector_lookup = sector_map[["ticker", "sector33"]].drop_duplicates(subset=["ticker"])

    for horizon in HORIZONS:
        raw_col = target_column_name(VARIANT_RAW, horizon)
        topix_rel_col = target_column_name(VARIANT_TOPIX_RELATIVE, horizon)

        plausible_raw = _plausible(out[raw_col])
        plausible_topix_rel = _plausible(out[topix_rel_col])
        market_forward = plausible_raw - plausible_topix_rel

        beta_residual_col = residual_target_column_name(VARIANT_BETA_RESIDUAL, horizon)
        out[beta_residual_col] = plausible_raw - out["beta"] * market_forward

        full_with_sector = dataset[["date", "ticker", raw_col]].merge(
            sector_lookup, on="ticker", how="left"
        )
        full_with_sector[raw_col] = _plausible(full_with_sector[raw_col])
        sector_day_mean = full_with_sector.groupby(["date", "sector33"])[raw_col].transform("mean")
        full_with_sector = full_with_sector.assign(_sector_day_mean=sector_day_mean)
        out = out.merge(
            full_with_sector[["date", "ticker", "_sector_day_mean"]], on=["date", "ticker"],
            how="left",
        )

        sector_relative_col = residual_target_column_name(VARIANT_SECTOR_RELATIVE, horizon)
        out[sector_relative_col] = plausible_raw - out["_sector_day_mean"]
        out = out.drop(columns=["_sector_day_mean"])

    return out


RESIDUAL_TARGET_COLUMNS: list[str] = [
    residual_target_column_name(variant, horizon)
    for variant in (VARIANT_BETA_RESIDUAL, VARIANT_SECTOR_RELATIVE)
    for horizon in HORIZONS
]
