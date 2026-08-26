"""Append-only Forward Observation logs. TWO separate immutable JSONL
files, joined by key at analysis time, rather than one log with a
"fillable later" field - this satisfies BOTH of the spec's append-only
requirements (a Prediction entry can never be overwritten; a Realized
Return only becomes writable once its Horizon has matured) with the
SAME simple write-once-per-file mechanism for both, instead of needing
in-place JSONL row mutation (which JSONL doesn't support cleanly).

`predictions_log.jsonl`: one line per (observation_date, ticker,
model_id) - written exactly once, the day the prediction was made.
`realized_returns_log.jsonl`: one line per (observation_date, ticker,
model_id) - appended ONLY once that Horizon's forward Close is available
in the (separately, already-fetched) OHLCV history; a key already
present is never re-appended (checked before writing, matching V1's own
`_load_existing_signal_log_keys()` dedup pattern in `pipeline/
run_forward_test.py`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionLogEntry:
    observation_date: str
    ticker: str
    model_id: str
    target_definition: str
    horizon: int
    prediction: float
    rank: int
    percentile: float
    bucket: str  # "Q1".."Q5"
    regime: str | None
    data_quality: str  # "OK" or a short issue label
    logged_at: str


@dataclass(frozen=True)
class RealizedReturnLogEntry:
    observation_date: str
    ticker: str
    model_id: str
    target_definition: str
    horizon: int
    realized_return: float
    realized_date: str
    logged_at: str


def _key(observation_date: str, ticker: str, model_id: str) -> tuple[str, str, str]:
    return (observation_date, ticker, model_id)


def load_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            keys.add(_key(entry["observation_date"], entry["ticker"], entry["model_id"]))
    return keys


def append_prediction_entries(path: Path, entries: list[PredictionLogEntry]) -> int:
    """Skips any entry whose (observation_date, ticker, model_id) key
    already exists in the file - the idempotency guarantee re-running
    the same day twice relies on. Returns the number of NEW rows written.
    """
    existing = load_existing_keys(path)
    new_entries = [
        e for e in entries if _key(e.observation_date, e.ticker, e.model_id) not in existing
    ]
    if not new_entries:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return len(new_entries)


def append_realized_return_entries(path: Path, entries: list[RealizedReturnLogEntry]) -> int:
    existing = load_existing_keys(path)
    new_entries = [
        e for e in entries if _key(e.observation_date, e.ticker, e.model_id) not in existing
    ]
    if not new_entries:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return len(new_entries)


def load_all_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
