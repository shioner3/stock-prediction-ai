"""Verifies the frozen Phase V2-1 engine's code/config has not changed
since V2-1 shipped (spec section 27, STOP condition #3).

Deliberately NOT reusing v2/manifest.py::_v2_code_files() (which globs
every v2/**/*.py file) - that glob would immediately "detect a change"
the moment ANY new Phase V2-2 file is added under v2/validation/, which
is not what this check means to catch. FROZEN_V2_1_FILES below is the
explicit file list Phase V2-1 shipped; this module hashes exactly those
files, so adding new V2-2 modules never perturbs the check.
"""

from __future__ import annotations

from pathlib import Path

from common.hashing import hash_files
from v2.manifest import V2_CONFIG_FILE

FROZEN_V2_1_FILES: list[Path] = [
    Path("v2/__init__.py"),
    Path("v2/candidate.py"),
    Path("v2/config/__init__.py"),
    Path("v2/config/loader.py"),
    Path("v2/features_adapter.py"),
    Path("v2/manifest.py"),
    Path("v2/pipeline.py"),
    Path("v2/ranking/__init__.py"),
    Path("v2/ranking/cross_sectional.py"),
    Path("v2/ranking/score.py"),
    Path("v2/stats.py"),
    Path("v2/targets_adapter.py"),
]


def current_v2_1_code_hash() -> str:
    return hash_files(FROZEN_V2_1_FILES)


def current_v2_1_config_hash() -> str:
    return hash_files([V2_CONFIG_FILE])


def verify_v2_1_unchanged(
    expected_code_hash: str, expected_config_hash: str
) -> tuple[bool, list[str]]:
    """Compares TODAY's V2-1 code/config hashes against hashes recorded
    at some earlier point (e.g. Phase V2-1's own saved manifest, or a
    value pinned at the start of this Phase's own preflight). Returns
    (unchanged, mismatched_field_names) - mirrors
    forward_test/manifest.py::verify_strategy_hashes_unchanged()'s
    return shape, the established pattern for this kind of check.
    """
    mismatches = []
    if current_v2_1_code_hash() != expected_code_hash:
        mismatches.append("code_hash")
    if current_v2_1_config_hash() != expected_config_hash:
        mismatches.append("config_hash")
    return len(mismatches) == 0, mismatches
