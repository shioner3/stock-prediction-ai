"""ONE-TIME migration (Phase V3-9): splits the existing monolithic
`data/forward_test/v3/predictions_log.jsonl` and `realized_returns_log.
jsonl` into one file per observation_date under `predictions/` and
`realized_returns/` - see scripts/run_v3_frozen_observation_day.py's own
module docstring for why (the single-file design hit GitHub's 100MB
per-file limit).

Not part of the daily automation - run exactly once, by hand, to
reorganize already-logged data. Verifies every row is preserved
(row count AND the exact set of (observation_date, ticker, model_id)
keys) before removing the old files - refuses to delete anything if the
split does not round-trip perfectly. Old files remain fully recoverable
via git history regardless.

Usage:
    python scripts/migrate_v3_observation_logs_to_daily_files.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STATE_ROOT = Path("data/forward_test/v3")
OLD_PREDICTIONS_PATH = STATE_ROOT / "predictions_log.jsonl"
OLD_REALIZED_RETURNS_PATH = STATE_ROOT / "realized_returns_log.jsonl"
PREDICTIONS_DIR = STATE_ROOT / "predictions"
REALIZED_RETURNS_DIR = STATE_ROOT / "realized_returns"


def _key(entry: dict) -> tuple[str, str, str]:
    return (entry["observation_date"], entry["ticker"], entry["model_id"])


def _split_by_observation_date(old_path: Path, out_dir: Path) -> tuple[int, set]:
    """Returns (rows_written, keys_written) for verification."""
    if not old_path.exists():
        print(f"  {old_path} does not exist - nothing to migrate")
        return 0, set()

    by_date: dict[str, list[str]] = {}
    with old_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_date.setdefault(entry["observation_date"], []).append(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    all_keys: set = set()
    for date_str, lines in sorted(by_date.items()):
        out_path = out_dir / f"{date_str}.jsonl"
        if out_path.exists():
            raise RuntimeError(f"STOP: {out_path} already exists - refusing to overwrite")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total_rows += len(lines)
        for line in lines:
            all_keys.add(_key(json.loads(line)))
    return total_rows, all_keys


def _verify_round_trip(old_path: Path, new_rows: int, new_keys: set) -> None:
    if not old_path.exists():
        return
    old_rows = 0
    old_keys: set = set()
    with old_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            old_rows += 1
            old_keys.add(_key(json.loads(line)))
    if old_rows != new_rows:
        raise RuntimeError(
            f"STOP: row count mismatch for {old_path} - old={old_rows} new={new_rows}. "
            "Refusing to delete the original file."
        )
    if old_keys != new_keys:
        missing = old_keys - new_keys
        extra = new_keys - old_keys
        raise RuntimeError(
            f"STOP: key mismatch for {old_path} - missing={len(missing)} extra={len(extra)}. "
            "Refusing to delete the original file."
        )
    print(f"  verified: {old_rows} rows / {len(old_keys)} keys preserved exactly")


def main() -> None:
    print("STEP 1: splitting predictions_log.jsonl by observation_date")
    pred_rows, pred_keys = _split_by_observation_date(OLD_PREDICTIONS_PATH, PREDICTIONS_DIR)
    print(f"  wrote {pred_rows} rows across {len(list(PREDICTIONS_DIR.glob('*.jsonl')))} files")

    print("\nSTEP 2: splitting realized_returns_log.jsonl by observation_date")
    real_rows, real_keys = _split_by_observation_date(
        OLD_REALIZED_RETURNS_PATH, REALIZED_RETURNS_DIR
    )
    real_file_count = len(list(REALIZED_RETURNS_DIR.glob("*.jsonl")))
    print(f"  wrote {real_rows} rows across {real_file_count} files")

    print("\nSTEP 3: verifying round-trip (row count + key set) before touching originals")
    _verify_round_trip(OLD_PREDICTIONS_PATH, pred_rows, pred_keys)
    _verify_round_trip(OLD_REALIZED_RETURNS_PATH, real_rows, real_keys)

    print("\nSTEP 4: removing the now-redundant monolithic files (content fully preserved above; "
          "also permanently recoverable via git history)")
    if OLD_PREDICTIONS_PATH.exists():
        OLD_PREDICTIONS_PATH.unlink()
        print(f"  removed {OLD_PREDICTIONS_PATH}")
    if OLD_REALIZED_RETURNS_PATH.exists():
        OLD_REALIZED_RETURNS_PATH.unlink()
        print(f"  removed {OLD_REALIZED_RETURNS_PATH}")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
