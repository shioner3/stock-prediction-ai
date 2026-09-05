"""Phase V3-7 CLI entry point: one day's V3 Forward Observation. Runs
AFTER V1's own `scripts/run_forward_test_day.py` in the same workflow -
reuses the SAME already-fetched OHLCV cache (`data/forward_test/raw`/
`processed`) and the SAME frozen Universe filter's fetch manifest
(`data/forward_test/_fetch_manifest.json`), rather than re-fetching
~2,880 tickers a second time. V1 and V3 state are otherwise completely
separate (`data/forward_test/v3/` vs. V1's own top-level files/dirs).

Safe to re-run on the same day (idempotent - see `v3.frozen.
observation_log`'s append-only+key-dedup design).

Phase V3-9 (log storage fix): Predictions and Realized Returns are each
stored ONE FILE PER OBSERVATION DATE (`predictions/{date}.jsonl`,
`realized_returns/{date}.jsonl`) rather than in a single
ever-growing log - the single-file version hit GitHub's 100MB per-file
limit after only 7 trading days (~44,000 rows/day). `v3.frozen.
observation_log`'s append/load functions are unchanged and still fully
generic over whatever path they're given; only the PATH this script
picks changed. A day's own file is bounded in size forever (one day's
Universe x 16 models, never more), so no file can ever approach the
limit again, and idempotency dedup only needs that one day's own file -
strictly cheaper than before, not just smaller. Realized-return
maturity checking used to reload the ENTIRE historical Predictions Log
every run (O(total rows ever logged), the same unbounded-growth
problem); it now only revisits observation-date files that are not yet
"settled" (a day is settled once its Realized Returns file has as many
rows as its Predictions file - see `_is_settled()` - at which point it
is never opened again). This changes WHEN maturity is (re-)checked, not
WHAT is computed - the same `compute_realized_return_entries()` formula
against the same augmented dataset, still no future data ever reaching
a training path.

Usage:
    python scripts/run_v3_frozen_observation_day.py                     # today, JST
    python scripts/run_v3_frozen_observation_day.py --run-date 2026-08-21
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backtest.market_regime import compute_market_regime  # noqa: E402
from forward_test.integrity import check_data_integrity  # noqa: E402
from storage.parquet_store import load_ohlcv  # noqa: E402
from v3.config.loader import load_v3_config  # noqa: E402
from v3.dataset import build_v3_dataset  # noqa: E402
from v3.frozen.manifest import (  # noqa: E402
    FrozenModelHashMismatchError,
    FrozenModelSpec,  # noqa: E402
    load_manifest_raw,
    load_model_artifact,
    verify_frozen_models_unchanged,
)
from v3.frozen.observation_log import (  # noqa: E402
    append_prediction_entries,
    append_realized_return_entries,
    load_all_entries,
    load_existing_keys,
)
from v3.frozen.observe_day import build_observation_entries  # noqa: E402
from v3.frozen.realize_returns import compute_realized_return_entries  # noqa: E402
from v3.residual.reproduce import build_augmented_dataset  # noqa: E402
from v3.robustness.aux_panel import attach_sector_and_scale  # noqa: E402
from v3.validation.wfo_config import MARKET_REGIME_CONFIG  # noqa: E402

V1_FETCH_MANIFEST_PATH = Path("data/forward_test/_fetch_manifest.json")
V1_RAW_DIR = Path("data/forward_test/raw")
V1_PROCESSED_DIR = Path("data/forward_test/processed")
STATE_ROOT = Path("data/forward_test/v3")
MODELS_MANIFEST_PATH = STATE_ROOT / "v3_frozen_models_manifest.json"
PREDICTIONS_DIR = STATE_ROOT / "predictions"
REALIZED_RETURNS_DIR = STATE_ROOT / "realized_returns"
DAILY_DIR = STATE_ROOT / "daily"

STALE_FRACTION_ABORT_THRESHOLD = 0.5


def _predictions_path(observation_date: datetime.date) -> Path:
    return PREDICTIONS_DIR / f"{observation_date.isoformat()}.jsonl"


def _realized_returns_path(observation_date_str: str) -> Path:
    return REALIZED_RETURNS_DIR / f"{observation_date_str}.jsonl"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_settled(predictions_path: Path, realized_returns_path: Path) -> bool:
    """A day is settled once every one of its logged Predictions has a
    Realized Return (all 4 Horizons matured for every ticker/model) - at
    that point its files are permanently done and never opened again.
    Compares row COUNTS only (cheap line counts, no JSON parsing) so a
    settled day costs nothing beyond checking it stayed settled.
    """
    return _count_lines(realized_returns_path) >= _count_lines(predictions_path)


class SafeAbortError(RuntimeError):
    """Mirrors `pipeline.run_forward_test.SafeAbortError`'s reason-code
    convention: MARKET_DATA_UNAVAILABLE, STALE_THRESHOLD_EXCEEDED,
    NO_VALID_TRADING_DAY.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"SAFE_ABORT[{reason}]: {detail}")


def _verify_artifacts_unchanged(saved_manifest: dict[str, Any]) -> list[str]:
    """Phase V3-9: re-hashes each of the 16 on-disk Model Artifact files
    NOW and compares against the `model_hash` the manifest recorded at
    training time - closes the gap `verify_frozen_models_unchanged()`
    leaves by design (that function only re-checks code/config-level
    hashes, never the artifact bytes themselves), catching artifact
    corruption/tampering that a code-level check alone cannot see.

    Deliberately kept HERE (in scripts/, not inside the v3/ package)
    rather than in v3/frozen/manifest.py: v3.hash.current_v3_code_hash()
    hashes every *.py file under the WHOLE v3/ tree, so adding this
    check inside v3/ would itself shift code_hash and break verification
    against the ALREADY-TRAINED manifest - a self-defeating change for a
    Phase whose entire point is protecting that existing manifest. Uses
    the SAME sha256(model_to_string()) formula
    v3.models.model_manifest.compute_model_hash() already uses (not a
    new algorithm), applied to a freshly re-loaded lgb.Booster via the
    existing, unmodified v3.frozen.manifest.load_model_artifact(). Only
    reads the manifest's existing model_hash as an expected value -
    never recomputes or rewrites it.
    """
    mismatches = []
    for model in saved_manifest["models"]:
        booster = load_model_artifact(Path(model["artifact_path"]))
        current_hash = hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()
        if current_hash != model["model_hash"]:
            mismatches.append(f"{model['model_id']}.model_hash")
    return mismatches


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Phase V3-7 Forward Observation day")
    parser.add_argument(
        "--run-date", type=datetime.date.fromisoformat, default=datetime.date.today(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_date = args.run_date

    if not MODELS_MANIFEST_PATH.exists():
        print(f"No Frozen Models manifest at {MODELS_MANIFEST_PATH}.")
        print("Run scripts/train_v3_frozen_models.py once before starting Observation.")
        sys.exit(1)
    saved_manifest = load_manifest_raw(MODELS_MANIFEST_PATH)
    unchanged, mismatches = verify_frozen_models_unchanged(saved_manifest)
    if not unchanged:
        raise FrozenModelHashMismatchError(f"FROZEN_MODEL_HASH_MISMATCH: {mismatches}")
    artifact_mismatches = _verify_artifacts_unchanged(saved_manifest)
    if artifact_mismatches:
        raise FrozenModelHashMismatchError(f"FROZEN_MODEL_HASH_MISMATCH: {artifact_mismatches}")

    if not V1_FETCH_MANIFEST_PATH.exists():
        raise SafeAbortError(
            "MARKET_DATA_UNAVAILABLE",
            f"{V1_FETCH_MANIFEST_PATH} not found - run V1's forward test step first today",
        )
    fetch_manifest = json.loads(V1_FETCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    final_tickers = sorted(
        t for t, e in fetch_manifest.get("tickers", {}).items() if e.get("included_in_universe")
    )
    if not final_tickers:
        raise SafeAbortError("MARKET_DATA_UNAVAILABLE", "0 tickers in today's fetch manifest")

    stale_count = 0
    for ticker in final_tickers:
        try:
            df = load_ohlcv(ticker, V1_RAW_DIR)
        except FileNotFoundError:
            stale_count += 1
            continue
        integrity_result = check_data_integrity(df, ticker, expected_date=run_date)
        if integrity_result.is_stale:
            stale_count += 1
    stale_fraction = stale_count / len(final_tickers)
    if stale_fraction > STALE_FRACTION_ABORT_THRESHOLD:
        raise SafeAbortError(
            "STALE_THRESHOLD_EXCEEDED",
            f"{stale_count}/{len(final_tickers)} tickers ({stale_fraction:.0%}) are stale",
        )

    v3_config = load_v3_config().model_copy(update={"source_processed_dir": V1_PROCESSED_DIR})
    specs = [FrozenModelSpec(**m) for m in saved_manifest["models"]]

    topix = load_ohlcv("TOPIX", V1_PROCESSED_DIR)
    result = build_observation_entries(
        final_tickers, v3_config, run_date, specs, MARKET_REGIME_CONFIG, topix,
    )
    if not result.entries:
        raise SafeAbortError(
            "NO_VALID_TRADING_DAY", f"no Feature/prediction rows built for {run_date}"
        )

    predictions_path = _predictions_path(run_date)
    n_new_predictions = append_prediction_entries(predictions_path, result.entries)

    # Realized return maturity: rebuild the full Feature/Target dataset
    # from whatever OHLCV is available NOW (this naturally extends
    # further into the future than any individual prediction's own
    # observation date, as more days accumulate day over day) and check
    # every not-yet-SETTLED observation date's predictions against it -
    # a settled date (every row already has a Realized Return) is never
    # re-opened, so this cost stays bounded to the trailing handful of
    # still-maturing dates rather than the entire log history.
    ticker_frame = pd.DataFrame({"ticker": final_tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]].drop_duplicates(
        subset=["ticker"]
    )
    full_dataset_now = build_v3_dataset(final_tickers, v3_config)
    augmented_now = build_augmented_dataset(full_dataset_now, sector_map)

    n_new_realized = 0
    if PREDICTIONS_DIR.exists():
        for day_predictions_path in sorted(PREDICTIONS_DIR.glob("*.jsonl")):
            day_realized_path = _realized_returns_path(day_predictions_path.stem)
            if _is_settled(day_predictions_path, day_realized_path):
                continue
            day_predictions = load_all_entries(day_predictions_path)
            already_realized = load_existing_keys(day_realized_path)
            realized_entries = compute_realized_return_entries(
                day_predictions, augmented_now, already_realized
            )
            n_new_realized += append_realized_return_entries(day_realized_path, realized_entries)

    regime_df = compute_market_regime(topix, MARKET_REGIME_CONFIG)
    regime_today = regime_df.loc[regime_df["date"] == run_date, "regime"]

    print(f"run_date: {run_date}")
    print(f"universe_size: {result.universe_size}")
    print(f"regime: {regime_today.iloc[0] if len(regime_today) else 'UNKNOWN'}")
    print(f"prediction_entries_built: {len(result.entries)}")
    print(f"prediction_entries_new: {n_new_predictions}")
    print(f"realized_return_entries_new: {n_new_realized}")

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    daily_summary = {
        "run_date": run_date.isoformat(), "universe_size": result.universe_size,
        "prediction_entries_built": len(result.entries),
        "prediction_entries_new": n_new_predictions, "realized_return_entries_new": n_new_realized,
    }
    (DAILY_DIR / f"{run_date.isoformat()}.json").write_text(
        json.dumps(daily_summary, default=_json_default, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except SafeAbortError as exc:
        print(f"SAFE_ABORT[{exc.reason}]: {exc.detail}")
        sys.exit(2)
    except FrozenModelHashMismatchError as exc:
        print(str(exc))
        sys.exit(3)
