"""V3 integrity manifest (spec section 32-33): a completely separate hash
namespace from V1's Strategy Hash (`forward_test/manifest.py`) and V2's
manifest (`v2/manifest.py`) - V3's hashes are computed over V3's own
config/code/data only, so nothing here can collide with or be mistaken
for a V1 or V2 integrity value.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.hashing import hash_files

V3_CONFIG_FILE = Path("v3/config/v3_settings.yaml")
V3_CODE_ROOT = Path("v3")
V3_FEATURE_CODE_ROOT = Path("v3/features")
V3_TARGET_CODE_ROOT = Path("v3/targets")
V3_VERSION = "v3-phase-1"


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def current_v3_code_hash() -> str:
    return hash_files(_py_files(V3_CODE_ROOT))


def current_v3_config_hash() -> str:
    return hash_files([V3_CONFIG_FILE])


def current_v3_feature_hash() -> str:
    """Hash of ONLY the dataset-defining code (v3/features/ + v3/targets/)
    - narrower than code_hash (which also covers v3/leakage/, v3/hash.py
    itself, v3/config/), so a Feature/Target formula change is
    detectable independently of unrelated V3-internal edits.
    """
    return hash_files(_py_files(V3_FEATURE_CODE_ROOT) + _py_files(V3_TARGET_CODE_ROOT))


def hash_dataframe(df: pd.DataFrame) -> str:
    """A dataset hash: sha256 over a canonical CSV rendering of df, sorted
    by column name and (if present) by ticker/date row order - so
    re-running the SAME dataset build from the SAME inputs reproduces the
    SAME hash regardless of incidental row/column ordering differences
    (spec section 33's reproducibility requirement).
    """
    ordered = df.reindex(sorted(df.columns), axis=1)
    sort_cols = [c for c in ("ticker", "date") if c in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    csv_bytes = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


@dataclass(frozen=True)
class V3Manifest:
    version: str
    generated_at_utc: str
    git_commit: str | None
    config_hash: str
    code_hash: str
    feature_hash: str
    dataset_hash: str | None
    model_hash: str | None  # always None in Phase V3-1 - no model exists yet
    universe_size: int
    date_range_start: date_type | None
    date_range_end: date_type | None
    feature_list: list[str]
    target_list: list[str]
    horizons: list[int]
    random_seed: int | None


def build_v3_manifest(
    universe_size: int,
    date_range_start: date_type | None,
    date_range_end: date_type | None,
    feature_list: list[str],
    target_list: list[str],
    horizons: list[int],
    dataset: pd.DataFrame | None = None,
    random_seed: int | None = None,
) -> V3Manifest:
    return V3Manifest(
        version=V3_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        config_hash=current_v3_config_hash(),
        code_hash=current_v3_code_hash(),
        feature_hash=current_v3_feature_hash(),
        dataset_hash=hash_dataframe(dataset) if dataset is not None else None,
        model_hash=None,
        universe_size=universe_size,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        feature_list=feature_list,
        target_list=target_list,
        horizons=horizons,
        random_seed=random_seed,
    )


def _json_default(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, date_type):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_v3_manifest(manifest: V3Manifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dataclasses.asdict(manifest), default=_json_default, indent=2, ensure_ascii=False
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path
