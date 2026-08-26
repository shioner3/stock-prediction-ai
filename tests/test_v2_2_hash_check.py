from __future__ import annotations

from v2.validation.hash_check import (
    FROZEN_V2_1_FILES,
    current_v2_1_code_hash,
    current_v2_1_config_hash,
    verify_v2_1_unchanged,
)


def test_frozen_file_list_covers_every_v2_1_module() -> None:
    expected = {
        "v2/__init__.py", "v2/candidate.py", "v2/config/__init__.py",
        "v2/config/loader.py", "v2/features_adapter.py", "v2/manifest.py",
        "v2/pipeline.py", "v2/ranking/__init__.py", "v2/ranking/cross_sectional.py",
        "v2/ranking/score.py", "v2/stats.py", "v2/targets_adapter.py",
    }
    actual = {p.as_posix() for p in FROZEN_V2_1_FILES}
    assert actual == expected


def test_frozen_file_list_excludes_v2_validation_package() -> None:
    """The hash check must never include Phase V2-2's own new files -
    otherwise adding new validation modules would falsely "detect" a
    V2-1 change.
    """
    for path in FROZEN_V2_1_FILES:
        assert "validation" not in path.parts


def test_code_hash_is_deterministic() -> None:
    assert current_v2_1_code_hash() == current_v2_1_code_hash()


def test_config_hash_is_deterministic() -> None:
    assert current_v2_1_config_hash() == current_v2_1_config_hash()


def test_verify_v2_1_unchanged_matches_current_state() -> None:
    unchanged, mismatches = verify_v2_1_unchanged(
        current_v2_1_code_hash(), current_v2_1_config_hash()
    )
    assert unchanged is True
    assert mismatches == []


def test_verify_v2_1_unchanged_detects_code_mismatch() -> None:
    unchanged, mismatches = verify_v2_1_unchanged("wrong_hash", current_v2_1_config_hash())
    assert unchanged is False
    assert mismatches == ["code_hash"]


def test_verify_v2_1_unchanged_detects_config_mismatch() -> None:
    unchanged, mismatches = verify_v2_1_unchanged(current_v2_1_code_hash(), "wrong_hash")
    assert unchanged is False
    assert mismatches == ["config_hash"]


def test_verify_v2_1_unchanged_detects_both_mismatch() -> None:
    unchanged, mismatches = verify_v2_1_unchanged("wrong", "wrong")
    assert unchanged is False
    assert set(mismatches) == {"code_hash", "config_hash"}
