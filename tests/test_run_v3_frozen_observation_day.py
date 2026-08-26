"""scripts/run_v3_frozen_observation_day.py - Phase V3-9's Artifact-level
model_hash re-verification (`_verify_artifacts_unchanged`). Kept as a
CLI-script-level check (not inside v3/frozen/manifest.py) deliberately -
see that function's own docstring for why: v3.hash.current_v3_code_hash()
hashes the whole v3/ tree, so adding this check inside v3/ would itself
shift code_hash and break verification against the ALREADY-TRAINED
manifest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
from v3_3_test_helpers import build_v3_3_config_and_tickers

from v3.dataset import build_v3_dataset
from v3.frozen.train import train_one_frozen_model
from v3.residual.reproduce import build_augmented_dataset
from v3.robustness.aux_panel import attach_sector_and_scale

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_v3_frozen_observation_day.py"
)
_spec = importlib.util.spec_from_file_location("run_v3_frozen_observation_day", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
observation_day = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = observation_day
_spec.loader.exec_module(observation_day)


def _train_two_tiny_models(tmp_path: Path):
    config, tickers = build_v3_3_config_and_tickers(tmp_path, n_tickers=8, n_days=400)
    dataset = build_v3_dataset(tickers, config)
    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]].drop_duplicates(
        subset=["ticker"]
    )
    augmented = build_augmented_dataset(dataset, sector_map)
    t0 = dataset["date"].max()
    trained_a = train_one_frozen_model(
        augmented, "raw", 5, t0, "irrelevant-for-this-test", tmp_path / "artifacts"
    )
    trained_b = train_one_frozen_model(
        augmented, "raw", 10, t0, "irrelevant-for-this-test", tmp_path / "artifacts"
    )
    return trained_a, trained_b


def test_verify_artifacts_unchanged_passes_for_untouched_file(tmp_path: Path) -> None:
    trained_a, _ = _train_two_tiny_models(tmp_path)
    saved_manifest = {
        "models": [
            {
                "model_id": trained_a.spec.model_id,
                "artifact_path": trained_a.spec.artifact_path,
                "model_hash": trained_a.spec.model_hash,
            }
        ]
    }
    mismatches = observation_day._verify_artifacts_unchanged(saved_manifest)
    assert mismatches == []


def test_verify_artifacts_unchanged_detects_tampered_file(tmp_path: Path) -> None:
    trained_a, trained_b = _train_two_tiny_models(tmp_path)
    assert trained_a.spec.model_hash != trained_b.spec.model_hash

    # Simulate corruption/tampering: model A's artifact file on disk is
    # silently replaced by a DIFFERENT (but still validly-formatted)
    # trained model's bytes, WITHOUT any code/config change - the exact
    # gap verify_frozen_models_unchanged() (code-level only) cannot see.
    # A "swap with another valid model" is used (rather than corrupting
    # bytes) because LightGBM's own parser hard-fails on malformed text,
    # which would raise instead of exercising the hash-mismatch path.
    artifact_path_a = Path(trained_a.spec.artifact_path)
    artifact_path_a.write_bytes(Path(trained_b.spec.artifact_path).read_bytes())

    saved_manifest = {
        "models": [
            {
                "model_id": trained_a.spec.model_id,
                "artifact_path": trained_a.spec.artifact_path,
                "model_hash": trained_a.spec.model_hash,
            }
        ]
    }
    mismatches = observation_day._verify_artifacts_unchanged(saved_manifest)
    assert mismatches == [f"{trained_a.spec.model_id}.model_hash"]
