"""V2 integrity manifest (spec section 20).

Completely separate hash namespace from V1's Strategy Hash
(forward_test/manifest.py::compute_strategy_hashes) and V1's config_hash
(common/hashing.py::hash_files() over config/settings.yaml +
config/universe_filters.yaml) - V2's hashes are computed over V2's own
config/code files only, so nothing here can ever collide with or be
mistaken for a V1 integrity value.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path

from common.hashing import hash_files

V2_CONFIG_FILE = Path("v2/config/v2_settings.yaml")
V2_CODE_ROOT = Path("v2")
V2_VERSION = "v2-phase-1"


def _v2_code_files() -> list[Path]:
    return sorted(p for p in V2_CODE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


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


@dataclass(frozen=True)
class V2Manifest:
    version: str
    generated_at_utc: str
    git_commit: str | None
    config_hash: str
    code_hash: str
    data_hash: str | None
    universe_size: int
    date_range_start: date_type | None
    date_range_end: date_type | None
    feature_list: list[str]
    score_weights: dict[str, float]
    forward_windows: list[int]


def build_v2_manifest(
    universe_size: int,
    date_range_start: date_type | None,
    date_range_end: date_type | None,
    feature_list: list[str],
    score_weights: dict[str, float],
    forward_windows: list[int],
    data_files: list[Path] | None = None,
) -> V2Manifest:
    data_hash = hash_files(data_files) if data_files else None
    return V2Manifest(
        version=V2_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        config_hash=hash_files([V2_CONFIG_FILE]),
        code_hash=hash_files(_v2_code_files()),
        data_hash=data_hash,
        universe_size=universe_size,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        feature_list=feature_list,
        score_weights=score_weights,
        forward_windows=forward_windows,
    )


def _json_default(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, date_type):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_v2_manifest(manifest: V2Manifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dataclasses.asdict(manifest), default=_json_default, indent=2, ensure_ascii=False
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path
