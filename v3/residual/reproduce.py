"""Spec section 2: V3-4 reproduction verification, then Model A training
on the 3 NEW Target definitions (B/C/D) across all 4 Horizons. Target A
(Raw) needs NO retraining - it is reused verbatim from Phase V3-4's own
saved predictions (`data/v3/robustness/predictions/target_raw_*d.parquet`,
themselves a bit-identical reproduction of Phase V3-3's training).

V3-4 introduced no dataset/config/Feature change relative to V3-3 (its
own report's `config_hash`/`feature_hash`/`dataset_hash` all matched
V3-3's exactly) - verifying THIS Phase reproduces V3-3's hashes therefore
IS the "V3-4 reproducibility" check spec section 2 asks for. Reuses
`v3.robustness.reproduce.verify_against_v3_3()`/`check_primary_spread_
reproduction()` UNCHANGED rather than re-deriving the identical check a
second way.
"""

from __future__ import annotations

import pandas as pd

from backtest.walk_forward import WalkForwardWindow
from v3.residual.targets import compute_residual_targets
from v3.robustness.beta import compute_rolling_beta
from v3.robustness.reproduce import (  # noqa: F401 (re-exported for CLI convenience)
    HashVerification,
    V3_3ReferenceHashes,
    load_v3_3_reference_hashes,
    verify_against_v3_3,
)
from v3.robustness.v3_3_reference import check_primary_spread_reproduction
from v3.targets.registry import VARIANT_TOPIX_RELATIVE, target_column_name
from v3.validation.ranking_metrics import evaluate_ranking
from v3.validation.train_predict import run_model_a_on_window
from v3.validation.windows import get_v3_3_windows

TARGET_A_RAW = "raw"
TARGET_B_TOPIX_RELATIVE = "topix_relative"
TARGET_C_BETA_RESIDUAL = "beta_residual"
TARGET_D_SECTOR_RELATIVE = "sector_relative"
TARGET_DEFINITIONS = (
    TARGET_A_RAW, TARGET_B_TOPIX_RELATIVE, TARGET_C_BETA_RESIDUAL, TARGET_D_SECTOR_RELATIVE,
)


def verify_primary_reproduction(reproduced_raw_5d: pd.DataFrame) -> bool:
    ranking = evaluate_ranking(reproduced_raw_5d, 5)
    return check_primary_spread_reproduction(ranking.q5_q1_spread)


def build_augmented_dataset(dataset: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    beta_panel = compute_rolling_beta(dataset)
    return compute_residual_targets(dataset, beta_panel, sector_map)


def target_column_for(definition: str, horizon: int) -> str:
    if definition == TARGET_A_RAW:
        return target_column_name("raw", horizon)
    if definition == TARGET_B_TOPIX_RELATIVE:
        return target_column_name(VARIANT_TOPIX_RELATIVE, horizon)
    if definition == TARGET_C_BETA_RESIDUAL:
        return f"target_beta_residual_{horizon}d"
    if definition == TARGET_D_SECTOR_RELATIVE:
        return f"target_sector_relative_{horizon}d"
    raise ValueError(f"unknown target definition: {definition}")


def reproduce_residual_predictions(
    augmented_dataset: pd.DataFrame, windows: list[WalkForwardWindow] | None,
    horizons: tuple[int, ...], definitions: tuple[str, ...] = (
        TARGET_B_TOPIX_RELATIVE, TARGET_C_BETA_RESIDUAL, TARGET_D_SECTOR_RELATIVE,
    ),
) -> dict[tuple[str, int], pd.DataFrame]:
    """Trains Model A (frozen V3-2/V3-3 code, unmodified) on every
    (definition, horizon) combination NOT already covered by V3-4's own
    saved Raw-target predictions. Returns {(definition, horizon): pooled
    OOS predictions}.
    """
    if windows is None:
        windows = get_v3_3_windows()
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for definition in definitions:
        for horizon in horizons:
            target_col = target_column_for(definition, horizon)
            pooled = []
            for window in windows:
                wp = run_model_a_on_window(augmented_dataset, window, target_col)
                tagged = wp.predictions.copy()
                tagged["window_index"] = window.index
                pooled.append(tagged)
            out[(definition, horizon)] = pd.concat(pooled, ignore_index=True)
    return out
