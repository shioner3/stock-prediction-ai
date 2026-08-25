"""v3/frozen/manifest.py - model_id naming, hash computation, and
FROZEN_MODEL_HASH_MISMATCH detection.
"""

from __future__ import annotations

from v3.frozen.manifest import (
    ALL_MODEL_IDS,
    FROZEN_HORIZONS,
    current_v3_residual_target_hash,
    model_id_for,
    verify_frozen_models_unchanged,
)
from v3.residual.reproduce import TARGET_DEFINITIONS


def test_model_id_naming() -> None:
    assert model_id_for("raw", 5) == "V3-FROZEN-RAW-5D"
    assert model_id_for("topix_relative", 10) == "V3-FROZEN-TOPIX-10D"
    assert model_id_for("beta_residual", 15) == "V3-FROZEN-BETA-15D"
    assert model_id_for("sector_relative", 20) == "V3-FROZEN-SECTOR-20D"


def test_all_model_ids_covers_16_combinations() -> None:
    assert len(ALL_MODEL_IDS) == len(TARGET_DEFINITIONS) * len(FROZEN_HORIZONS) == 16
    assert len(set(ALL_MODEL_IDS)) == 16  # all unique


def test_residual_target_hash_deterministic() -> None:
    a = current_v3_residual_target_hash()
    b = current_v3_residual_target_hash()
    assert a == b
    assert len(a) == 64  # sha256 hex digest


def test_verify_unchanged_passes_when_hashes_match_current(monkeypatch) -> None:
    import v3.frozen.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "current_v3_feature_hash", lambda: "x")
    monkeypatch.setattr(manifest_module, "current_v3_residual_target_hash", lambda: "y")
    monkeypatch.setattr(manifest_module, "current_v3_config_hash", lambda: "z")
    monkeypatch.setattr(manifest_module, "current_v3_code_hash", lambda: "w")

    current = {
        "feature_hash": "x", "residual_target_hash": "y", "config_hash": "z", "code_hash": "w",
    }
    saved_manifest = {
        "models": [
            {"model_id": "V3-FROZEN-RAW-5D", **current},
            {"model_id": "V3-FROZEN-BETA-5D", **current},
        ]
    }
    unchanged, mismatches = verify_frozen_models_unchanged(saved_manifest)
    assert unchanged is True
    assert mismatches == []


def test_verify_unchanged_detects_mismatch() -> None:
    saved_manifest = {
        "models": [
            {
                "model_id": "V3-FROZEN-RAW-5D", "feature_hash": "OLD",
                "residual_target_hash": "y", "config_hash": "z", "code_hash": "w",
            },
        ]
    }
    unchanged, mismatches = verify_frozen_models_unchanged(saved_manifest)
    assert unchanged is False
    assert "V3-FROZEN-RAW-5D.feature_hash" in mismatches
