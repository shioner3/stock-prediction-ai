"""Frozen Model identity, hashing, and manifest persistence.

`current_v3_residual_target_hash()` extends V3-2's existing `feature_hash`
concept (which already covers `v3/features/`+`v3/targets/` - the Raw/
TOPIX-relative definitions) with the 2 NEW target-defining modules V3-5
introduced (`v3/residual/targets.py`, and `v3/robustness/beta.py` which
it depends on for the Beta-adjusted Residual definition) - applied
uniformly to all 16 models' manifests (a harmless superset for Raw/
TOPIX-relative, which don't actually depend on this code, but keeping one
consistent field across all 16 is simpler than a per-definition hash and
still catches any relevant drift).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
from lightgbm import LGBMRegressor

from common.hashing import hash_files
from v3.hash import current_v3_code_hash, current_v3_config_hash, current_v3_feature_hash
from v3.models.config import BASELINE_PARAMS, RANDOM_SEED
from v3.models.model_manifest import compute_model_hash
from v3.residual.reproduce import TARGET_DEFINITIONS

FROZEN_MODEL_VERSION = "v3-frozen-v1"
FROZEN_HORIZONS: tuple[int, ...] = (5, 10, 15, 20)

_DEFINITION_LABEL = {
    "raw": "RAW", "topix_relative": "TOPIX", "beta_residual": "BETA", "sector_relative": "SECTOR",
}


def model_id_for(definition: str, horizon: int) -> str:
    return f"V3-FROZEN-{_DEFINITION_LABEL[definition]}-{horizon}D"


ALL_MODEL_IDS: list[str] = [
    model_id_for(d, h) for d in TARGET_DEFINITIONS for h in FROZEN_HORIZONS
]


def current_v3_residual_target_hash() -> str:
    paths = [Path("v3/residual/targets.py"), Path("v3/robustness/beta.py")]
    return hash_files(paths)


@dataclass(frozen=True)
class FrozenModelSpec:
    model_id: str
    target_definition: str
    horizon: int
    target_col: str
    training_start: str
    training_end: str  # T0
    training_row_count: int
    artifact_path: str  # relative to repo root
    model_hash: str
    feature_hash: str
    residual_target_hash: str
    config_hash: str
    code_hash: str
    dataset_hash: str
    hyperparameters: dict[str, Any]
    random_seed: int


@dataclass(frozen=True)
class FrozenModelsManifest:
    version: str
    t0: str
    created_at_utc: str
    models: list[FrozenModelSpec]


def save_model_artifact(model: LGBMRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_model_artifact(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def build_frozen_model_spec(
    definition: str, horizon: int, target_col: str, model: LGBMRegressor,
    training_start: date_type, training_end: date_type, training_row_count: int,
    artifact_path: Path, dataset_hash: str,
) -> FrozenModelSpec:
    return FrozenModelSpec(
        model_id=model_id_for(definition, horizon), target_definition=definition, horizon=horizon,
        target_col=target_col, training_start=training_start.isoformat(),
        training_end=training_end.isoformat(), training_row_count=training_row_count,
        artifact_path=artifact_path.as_posix(), model_hash=compute_model_hash(model),
        feature_hash=current_v3_feature_hash(),
        residual_target_hash=current_v3_residual_target_hash(),
        config_hash=current_v3_config_hash(), code_hash=current_v3_code_hash(),
        dataset_hash=dataset_hash, hyperparameters=BASELINE_PARAMS.as_kwargs(),
        random_seed=RANDOM_SEED,
    )


def build_manifest(t0: date_type, models: list[FrozenModelSpec]) -> FrozenModelsManifest:
    return FrozenModelsManifest(
        version=FROZEN_MODEL_VERSION, t0=t0.isoformat(),
        created_at_utc=datetime.now(timezone.utc).isoformat(), models=models,
    )


def save_manifest(manifest: FrozenModelsManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataclasses.asdict(manifest), indent=2, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def load_manifest_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenModelHashMismatchError(RuntimeError):
    """Raised when ANY of the 16 frozen models' code/config hashes no
    longer match what the manifest recorded at training time - spec's
    own explicit FROZEN_MODEL_HASH_MISMATCH SAFE_ABORT condition.
    """


def verify_frozen_models_unchanged(saved_manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """Recomputes the CODE-level hashes now (feature_hash/residual_target_hash/
    config_hash/code_hash - the same for all 16 models) and compares
    against the manifest. Does NOT retrain or recompute model_hash (that
    would require the training data again) - a code/config-level mismatch
    is already sufficient evidence the frozen recipe has drifted.
    """
    current = {
        "feature_hash": current_v3_feature_hash(),
        "residual_target_hash": current_v3_residual_target_hash(),
        "config_hash": current_v3_config_hash(),
        "code_hash": current_v3_code_hash(),
    }
    mismatches = []
    for model in saved_manifest["models"]:
        for field_name, current_value in current.items():
            if model[field_name] != current_value:
                mismatches.append(f"{model['model_id']}.{field_name}")
    return len(mismatches) == 0, mismatches
