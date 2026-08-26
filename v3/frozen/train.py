"""Trains each of the 16 Frozen Models exactly once, on ALL available
data through T0 - no Walk-Forward windowing, no held-out OOS slice (this
is deployment training, not research validation; V3-3/V3-4/V3-5 already
did the OOS validation this recipe is frozen from). Reuses `v3.models.
regression.fit_regression_model()`/`v3.models.data_prep.
prepare_training_set()` UNCHANGED - the exact same frozen Hyperparameters/
seed/target-implausibility-filtering every prior V3 training run used.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

from v3.features.registry import CORE_FEATURE_NAMES
from v3.frozen.manifest import (
    FROZEN_HORIZONS,
    FrozenModelSpec,
    build_frozen_model_spec,
    save_model_artifact,
)
from v3.models.data_prep import TrainingSet, prepare_training_set
from v3.models.regression import fit_regression_model
from v3.residual.reproduce import TARGET_DEFINITIONS, target_column_for


@dataclass(frozen=True)
class TrainedFrozenModel:
    spec: FrozenModelSpec
    model: LGBMRegressor
    training_set: TrainingSet


def train_one_frozen_model(
    augmented_dataset: pd.DataFrame, definition: str, horizon: int, t0: date_type,
    dataset_hash: str, artifact_dir: Path,
) -> TrainedFrozenModel:
    target_col = target_column_for(definition, horizon)
    # Narrow to only the columns this ONE model needs before handing the
    # frame to prepare_training_set() (unchanged, V3-2 code) - the full
    # augmented_dataset carries all 16 Target columns + auxiliary columns
    # (sector33, beta, ...) at once, and prepare_training_set()'s own
    # dropna/filter/consolidate steps on a ~3M-row frame that wide can
    # blow past available contiguous memory (observed in the real Full
    # Universe run). Narrowing is purely a memory-footprint change - same
    # rows, same values, same filtering logic, nothing about the Feature/
    # Target/Hyperparameter recipe changes.
    needed_cols = ["date", "ticker", target_col, *CORE_FEATURE_NAMES]
    narrowed = augmented_dataset[needed_cols]
    training_set = prepare_training_set(narrowed, target_col)
    del narrowed
    model = fit_regression_model(training_set.X, training_set.y)

    training_start = pd.to_datetime(training_set.dates).min().date()
    artifact_path = artifact_dir / f"{target_col}__{definition}.txt"
    save_model_artifact(model, artifact_path)

    spec = build_frozen_model_spec(
        definition=definition, horizon=horizon, target_col=target_col, model=model,
        training_start=training_start, training_end=t0, training_row_count=len(training_set.X),
        artifact_path=artifact_path, dataset_hash=dataset_hash,
    )
    # Keep only column identity (not the ~3M actual training rows) in the
    # TrainedFrozenModel returned to the caller - train_all_frozen_models()
    # accumulates 16 of these, and callers (e.g. the reproducibility check
    # in tests/test_v3_frozen_integration.py) only ever need
    # training_set.X.columns, never the row values, once the model+spec
    # already exist. Retaining full-size X/y 16x over would re-introduce
    # the same memory pressure this function just avoided per-call.
    light_training_set = TrainingSet(
        X=training_set.X.iloc[:0], y=training_set.y.iloc[:0],
        dates=training_set.dates.iloc[:0], tickers=training_set.tickers.iloc[:0],
    )
    del training_set
    gc.collect()
    return TrainedFrozenModel(spec=spec, model=model, training_set=light_training_set)


def train_all_frozen_models(
    augmented_dataset: pd.DataFrame, t0: date_type, dataset_hash: str, artifact_dir: Path,
) -> list[TrainedFrozenModel]:
    results = []
    for definition in TARGET_DEFINITIONS:
        for horizon in FROZEN_HORIZONS:
            results.append(
                train_one_frozen_model(
                    augmented_dataset, definition, horizon, t0, dataset_hash, artifact_dir
                )
            )
    return results
