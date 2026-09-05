"""scripts/run_v3_frozen_observation_day.py - Phase V3-9's Artifact-level
model_hash re-verification (`_verify_artifacts_unchanged`), and Phase
V3-9's per-observation-date log storage (`_predictions_path`,
`_realized_returns_path`, `_count_lines`, `_is_settled`).

The model_hash check is kept as a CLI-script-level check (not inside
v3/frozen/manifest.py) deliberately - see that function's own docstring
for why: v3.hash.current_v3_code_hash() hashes the whole v3/ tree, so
adding this check inside v3/ would itself shift code_hash and break
verification against the ALREADY-TRAINED manifest.

The per-date log storage replaces a single ever-growing predictions_
log.jsonl (which hit GitHub's 100MB per-file limit after 7 trading
days) with one bounded-size file per observation_date - see this
script's own module docstring for the full rationale.
"""

from __future__ import annotations

import datetime
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from v3_3_test_helpers import build_v3_3_config_and_tickers

from v3.dataset import build_v3_dataset
from v3.frozen.observation_log import (
    PredictionLogEntry,
    RealizedReturnLogEntry,
    append_prediction_entries,
    append_realized_return_entries,
)
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


def test_predictions_path_is_one_file_per_observation_date(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(observation_day, "PREDICTIONS_DIR", tmp_path / "predictions")
    d1 = datetime.date(2026, 8, 20)
    d2 = datetime.date(2026, 8, 21)
    assert observation_day._predictions_path(d1) != observation_day._predictions_path(d2)
    assert observation_day._predictions_path(d1).name == "2026-08-20.jsonl"


def test_count_lines_missing_file_is_zero(tmp_path: Path) -> None:
    assert observation_day._count_lines(tmp_path / "does_not_exist.jsonl") == 0


def _dummy_prediction(observation_date: str, ticker: str) -> PredictionLogEntry:
    return PredictionLogEntry(
        observation_date=observation_date, ticker=ticker, model_id="V3-FROZEN-RAW-5D",
        target_definition="raw", horizon=5, prediction=0.01, rank=1, percentile=0.9,
        bucket="Q5", regime="NEUTRAL", data_quality="OK", logged_at="2026-08-20T00:00:00",
    )


def _dummy_realized(observation_date: str, ticker: str) -> RealizedReturnLogEntry:
    return RealizedReturnLogEntry(
        observation_date=observation_date, ticker=ticker, model_id="V3-FROZEN-RAW-5D",
        target_definition="raw", horizon=5, realized_return=0.02,
        realized_date="2026-08-27", logged_at="2026-08-27T00:00:00",
    )


def test_is_settled_false_when_realized_count_below_predictions(tmp_path: Path) -> None:
    pred_path = tmp_path / "2026-08-20.jsonl"
    real_path = tmp_path / "realized" / "2026-08-20.jsonl"
    append_prediction_entries(
        pred_path,
        [_dummy_prediction("2026-08-20", "1301"), _dummy_prediction("2026-08-20", "1302")],
    )
    append_realized_return_entries(real_path, [_dummy_realized("2026-08-20", "1301")])
    assert observation_day._is_settled(pred_path, real_path) is False


def test_is_settled_true_when_every_prediction_realized(tmp_path: Path) -> None:
    pred_path = tmp_path / "2026-08-20.jsonl"
    real_path = tmp_path / "realized" / "2026-08-20.jsonl"
    append_prediction_entries(pred_path, [_dummy_prediction("2026-08-20", "1301")])
    append_realized_return_entries(real_path, [_dummy_realized("2026-08-20", "1301")])
    assert observation_day._is_settled(pred_path, real_path) is True


def test_is_settled_true_when_neither_file_exists(tmp_path: Path) -> None:
    # Not a real call-site scenario (the real loop only ever calls
    # _is_settled() for a predictions file that was just found to exist
    # via PREDICTIONS_DIR.glob()) - documented here as the trivial
    # "nothing logged, nothing pending" edge case: 0 predictions means
    # 0 realized returns are needed, so it is vacuously settled.
    pred_path = tmp_path / "2026-08-20.jsonl"
    real_path = tmp_path / "realized" / "2026-08-20.jsonl"
    assert observation_day._is_settled(pred_path, real_path) is True
