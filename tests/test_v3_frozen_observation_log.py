"""v3/frozen/observation_log.py - append-only + idempotency guarantees."""

from __future__ import annotations

from pathlib import Path

from v3.frozen.observation_log import (
    PredictionLogEntry,
    append_prediction_entries,
    append_realized_return_entries,
    load_all_entries,
    load_existing_keys,
)


def _entry(date: str, ticker: str, model_id: str, prediction: float = 0.01) -> PredictionLogEntry:
    return PredictionLogEntry(
        observation_date=date, ticker=ticker, model_id=model_id, target_definition="raw",
        horizon=5, prediction=prediction, rank=1, percentile=0.9, bucket="Q5", regime="BULL",
        data_quality="OK", logged_at="2026-08-21T00:00:00",
    )


def test_append_writes_new_entries(tmp_path: Path) -> None:
    path = tmp_path / "predictions_log.jsonl"
    entries = [_entry("2026-08-21", "7203", "V3-FROZEN-RAW-5D")]
    written = append_prediction_entries(path, entries)
    assert written == 1
    assert len(load_all_entries(path)) == 1


def test_append_is_idempotent_on_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "predictions_log.jsonl"
    entries = [_entry("2026-08-21", "7203", "V3-FROZEN-RAW-5D")]
    append_prediction_entries(path, entries)
    # Same day re-run with the SAME (date, ticker, model_id) key - even if
    # the prediction value were somehow different, the existing row must
    # never be duplicated or overwritten.
    duplicate = _entry("2026-08-21", "7203", "V3-FROZEN-RAW-5D", prediction=0.99)
    written_again = append_prediction_entries(path, [duplicate])
    assert written_again == 0
    all_entries = load_all_entries(path)
    assert len(all_entries) == 1
    assert all_entries[0]["prediction"] == 0.01  # original value preserved, never overwritten


def test_different_model_id_same_day_ticker_is_a_new_row(tmp_path: Path) -> None:
    path = tmp_path / "predictions_log.jsonl"
    append_prediction_entries(path, [_entry("2026-08-21", "7203", "V3-FROZEN-RAW-5D")])
    written = append_prediction_entries(path, [_entry("2026-08-21", "7203", "V3-FROZEN-BETA-5D")])
    assert written == 1
    assert len(load_all_entries(path)) == 2


def test_load_existing_keys_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_existing_keys(tmp_path / "does_not_exist.jsonl") == set()


def test_realized_return_log_append_only(tmp_path: Path) -> None:
    from v3.frozen.observation_log import RealizedReturnLogEntry

    path = tmp_path / "realized_returns_log.jsonl"
    entry = RealizedReturnLogEntry(
        observation_date="2026-08-21", ticker="7203", model_id="V3-FROZEN-RAW-5D",
        target_definition="raw", horizon=5, realized_return=0.03, realized_date="2026-08-28",
        logged_at="2026-08-28T00:00:00",
    )
    written = append_realized_return_entries(path, [entry])
    assert written == 1
    written_again = append_realized_return_entries(path, [entry])
    assert written_again == 0
    assert len(load_all_entries(path)) == 1
