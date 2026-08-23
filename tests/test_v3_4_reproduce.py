"""v3/robustness/reproduce.py - tail-based hash extraction (verified
against a small synthetic stand-in for V3-3's real (multi-GB) report
JSON, same key order/format `scripts/run_v3_3_full_universe_oos.py`
actually writes) and the hash-mismatch verification logic.
"""

from __future__ import annotations

import json
from pathlib import Path

from v3.hash import current_v3_code_hash, current_v3_config_hash, current_v3_feature_hash
from v3.robustness.reproduce import (
    V3_3ReferenceHashes,
    load_v3_3_reference_hashes,
    verify_against_v3_3,
)


def _write_fake_v3_3_report(path: Path) -> V3_3ReferenceHashes:
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = V3_3ReferenceHashes(
        code_hash="a" * 64, config_hash="b" * 64, feature_hash="c" * 64, dataset_hash="d" * 64,
    )
    payload = {
        "report": {"huge": "x" * 20_000},  # stand-in for the real multi-GB payload
        "decision_inputs": {}, "decision": {},
        "code_hash": hashes.code_hash, "config_hash": hashes.config_hash,
        "feature_hash": hashes.feature_hash, "dataset_hash": hashes.dataset_hash,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return hashes


def test_load_v3_3_reference_hashes_from_tail(tmp_path: Path) -> None:
    report_path = tmp_path / "fake_report.json"
    expected = _write_fake_v3_3_report(report_path)
    loaded = load_v3_3_reference_hashes(report_path)
    assert loaded == expected


def test_load_v3_3_reference_hashes_missing_field_raises(tmp_path: Path) -> None:
    report_path = tmp_path / "incomplete.json"
    report_path.write_text(json.dumps({"report": {}, "code_hash": "a" * 64}), encoding="utf-8")
    try:
        load_v3_3_reference_hashes(report_path, tail_bytes=4096)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_verify_against_v3_3_detects_mismatch() -> None:
    import pandas as pd

    dataset = pd.DataFrame({"date": ["2023-01-01"], "ticker": ["T0"], "x": [1.0]})
    reference = V3_3ReferenceHashes(
        code_hash="mismatch", config_hash="mismatch",
        feature_hash="mismatch", dataset_hash="mismatch",
    )
    result = verify_against_v3_3(dataset, reference=reference)
    assert result.all_match is False
    assert result.code_hash_match is False


def test_verify_against_v3_3_matches_current_code() -> None:
    import pandas as pd

    dataset = pd.DataFrame({"date": ["2023-01-01"], "ticker": ["T0"], "x": [1.0]})
    reference = V3_3ReferenceHashes(
        code_hash=current_v3_code_hash(), config_hash=current_v3_config_hash(),
        feature_hash=current_v3_feature_hash(), dataset_hash="irrelevant-for-this-check",
    )
    result = verify_against_v3_3(dataset, reference=reference)
    assert result.code_hash_match is True
    assert result.config_hash_match is True
    assert result.feature_hash_match is True
    assert result.dataset_hash_match is False
