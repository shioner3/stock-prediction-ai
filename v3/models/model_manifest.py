"""V3-2 Model Manifest (spec section 17/20): model_hash + Hyperparameters,
alongside the existing dataset/feature/config/code hashes from
`v3/hash.py` (reused unmodified) - the same independent hash namespace,
extended with one model-specific value.

`compute_model_hash()` hashes the trained LightGBM Booster's own
deterministic text dump (`Booster.model_to_string()`) - identical
(dataset, config, hyperparameters, random seed) training runs produce a
byte-identical dump, so a matching model_hash IS the reproducibility
proof spec section 17 asks for; a mismatch is a real, actionable signal.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from v3.hash import current_v3_code_hash, current_v3_config_hash, current_v3_feature_hash
from v3.models.config import BASELINE_PARAMS, RANDOM_SEED

V3_MODEL_VERSION = "v3-phase-2"


def compute_model_hash(model: LGBMRegressor | LGBMClassifier) -> str:
    dump = model.booster_.model_to_string()
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V3ModelManifest:
    version: str
    generated_at_utc: str
    target_col: str
    model_type: str  # "regression" | "classification" | "quantile"
    hyperparameters: dict[str, Any]
    random_seed: int
    model_hash: str
    dataset_hash: str | None
    feature_hash: str
    config_hash: str
    code_hash: str
    train_rows: int
    test_rows: int
    train_date_range: tuple[str, str]
    test_date_range: tuple[str, str]


def build_model_manifest(
    model_type: str,
    target_col: str,
    model: LGBMRegressor | LGBMClassifier,
    dataset_hash: str | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> V3ModelManifest:
    return V3ModelManifest(
        version=V3_MODEL_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        target_col=target_col,
        model_type=model_type,
        hyperparameters=BASELINE_PARAMS.as_kwargs(),
        random_seed=RANDOM_SEED,
        model_hash=compute_model_hash(model),
        dataset_hash=dataset_hash,
        feature_hash=current_v3_feature_hash(),
        config_hash=current_v3_config_hash(),
        code_hash=current_v3_code_hash(),
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_date_range=(str(train_df["date"].min()), str(train_df["date"].max())),
        test_date_range=(str(test_df["date"].min()), str(test_df["date"].max())),
    )


def save_model_manifest(manifest: V3ModelManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataclasses.asdict(manifest), indent=2, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path
