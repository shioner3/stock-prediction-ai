"""scripts/migrate_v3_observation_logs_to_daily_files.py - the one-time
Phase V3-9 migration from a single ever-growing predictions_log.jsonl /
realized_returns_log.jsonl into one file per observation_date. Verifies
the round-trip guarantee (row count AND exact key set preserved) that
gates deletion of the original files, on synthetic data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "migrate_v3_observation_logs_to_daily_files.py"
)
_spec = importlib.util.spec_from_file_location(
    "migrate_v3_observation_logs_to_daily_files", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
migrate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = migrate
_spec.loader.exec_module(migrate)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(observation_date: str, ticker: str, model_id: str = "V3-FROZEN-RAW-5D") -> dict:
    return {
        "observation_date": observation_date, "ticker": ticker, "model_id": model_id,
        "target_definition": "raw", "horizon": 5, "prediction": 0.01, "rank": 1,
        "percentile": 0.9, "bucket": "Q5", "regime": "NEUTRAL", "data_quality": "OK",
        "logged_at": "2026-08-20T00:00:00",
    }


def test_split_preserves_all_rows_and_removes_original(tmp_path: Path, monkeypatch) -> None:
    old_path = tmp_path / "predictions_log.jsonl"
    out_dir = tmp_path / "predictions"
    rows = [
        _row("2026-08-20", "1301"), _row("2026-08-20", "1302"),
        _row("2026-08-21", "1301"), _row("2026-08-21", "1303"),
        _row("2026-08-21", "1304"),
    ]
    _write_jsonl(old_path, rows)

    total_rows, total_keys = migrate._split_by_observation_date(old_path, out_dir)
    assert total_rows == 5
    assert len(total_keys) == 5
    migrate._verify_round_trip(old_path, total_rows, total_keys)  # must not raise

    assert (out_dir / "2026-08-20.jsonl").exists()
    assert (out_dir / "2026-08-21.jsonl").exists()
    assert sum(1 for _ in (out_dir / "2026-08-20.jsonl").open(encoding="utf-8")) == 2
    assert sum(1 for _ in (out_dir / "2026-08-21.jsonl").open(encoding="utf-8")) == 3


def test_verify_round_trip_raises_on_row_count_mismatch(tmp_path: Path) -> None:
    old_path = tmp_path / "predictions_log.jsonl"
    _write_jsonl(old_path, [_row("2026-08-20", "1301"), _row("2026-08-20", "1302")])

    import pytest

    with pytest.raises(RuntimeError, match="row count mismatch"):
        migrate._verify_round_trip(old_path, new_rows=1, new_keys={("2026-08-20", "1301", "x")})


def test_verify_round_trip_raises_on_key_mismatch(tmp_path: Path) -> None:
    old_path = tmp_path / "predictions_log.jsonl"
    _write_jsonl(old_path, [_row("2026-08-20", "1301")])

    import pytest

    wrong_key = {("2026-08-20", "9999", "V3-FROZEN-RAW-5D")}
    with pytest.raises(RuntimeError, match="key mismatch"):
        migrate._verify_round_trip(old_path, new_rows=1, new_keys=wrong_key)


def test_split_refuses_to_overwrite_existing_output_file(tmp_path: Path) -> None:
    old_path = tmp_path / "predictions_log.jsonl"
    out_dir = tmp_path / "predictions"
    _write_jsonl(old_path, [_row("2026-08-20", "1301")])
    out_dir.mkdir(parents=True)
    (out_dir / "2026-08-20.jsonl").write_text("pre-existing content\n", encoding="utf-8")

    import pytest

    with pytest.raises(RuntimeError, match="already exists"):
        migrate._split_by_observation_date(old_path, out_dir)


def test_split_missing_old_file_is_a_noop(tmp_path: Path) -> None:
    old_path = tmp_path / "does_not_exist.jsonl"
    out_dir = tmp_path / "predictions"
    total_rows, total_keys = migrate._split_by_observation_date(old_path, out_dir)
    assert total_rows == 0
    assert total_keys == set()
