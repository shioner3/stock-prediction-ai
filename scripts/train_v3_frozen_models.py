"""Phase V3-7 CLI entry point: trains and permanently persists all 16
Frozen Models (4 Target definitions x 4 Horizons) exactly ONCE, on ALL
available data through T0 = 2026-08-20. Run this exactly once to
initialize `data/forward_test/v3/`; re-running it after that point is
only for a genuine new Strategy Version (never done casually - see
`v3/frozen/manifest.py`'s FROZEN_MODEL_HASH_MISMATCH check, which the
daily observation script enforces on every run).

Usage:
    python scripts/train_v3_frozen_models.py [--limit-tickers N]
"""

from __future__ import annotations

import argparse
import datetime
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from v3.config.loader import load_v3_config  # noqa: E402
from v3.frozen.manifest import ALL_MODEL_IDS, build_manifest, save_manifest  # noqa: E402
from v3.frozen.train import train_all_frozen_models  # noqa: E402
from v3.hash import hash_dataframe  # noqa: E402
from v3.residual.reproduce import build_augmented_dataset, verify_against_v3_3  # noqa: E402
from v3.robustness.aux_panel import attach_sector_and_scale  # noqa: E402
from v3.robustness.reproduce import build_frozen_dataset  # noqa: E402

T0 = datetime.date(2026, 8, 20)
STATE_ROOT = Path("data/forward_test/v3")
ARTIFACT_DIR = STATE_ROOT / "models"
MANIFEST_PATH = STATE_ROOT / "v3_frozen_models_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 16 Phase V3-7 Frozen Models")
    parser.add_argument("--limit-tickers", type=int, default=None, help="dev/testing only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if MANIFEST_PATH.exists():
        print(f"STOP: {MANIFEST_PATH} already exists - Frozen Models are already trained.")
        print("Re-training would violate the 'trained exactly once' Model Freeze rule.")
        print("If a genuine new Strategy Version is needed, that requires an explicit,")
        print("separately-versioned decision - never a silent re-run of this script.")
        sys.exit(1)

    v3_config = load_v3_config()

    print(f"STEP 1: building Full Universe dataset (T0={T0})")
    dataset = build_frozen_dataset(v3_config, limit_tickers=args.limit_tickers)
    print(f"  dataset rows: {len(dataset)}, columns: {len(dataset.columns)}")
    print(f"  date range: {dataset['date'].min()} .. {dataset['date'].max()}")
    if dataset["date"].max() > T0:
        print(f"STOP: dataset contains data after T0={T0} - would leak future information")
        sys.exit(1)

    print("\nSTEP 2: verifying hash match against V3-3/V3-4/V3-5's frozen spec")
    verification = verify_against_v3_3(dataset)
    print(f"  config_hash_match={verification.config_hash_match}")
    print(f"  feature_hash_match={verification.feature_hash_match}")
    print(f"  dataset_hash_match={verification.dataset_hash_match}")
    if not verification.all_match:
        print("STOP: hash mismatch - frozen V3 spec has drifted since V3-3/V3-4/V3-5")
        sys.exit(1)
    dataset_hash = hash_dataframe(dataset)

    print("\nSTEP 3: building augmented dataset (Beta-adjusted Residual / Sector-relative targets)")
    tickers = sorted(dataset["ticker"].unique())
    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]].drop_duplicates(
        subset=["ticker"]
    )
    augmented = build_augmented_dataset(dataset, sector_map)
    # The raw (non-augmented) dataset is fully subsumed by augmented (a
    # superset copy) - dropping the reference here frees ~1.7GB before
    # STEP 4's 16 memory-heavy training calls, which was necessary to
    # avoid a real MemoryError observed on the Full Universe run.
    del dataset
    gc.collect()

    print("\nSTEP 4: training all 16 Frozen Models (this will take a while)...")
    trained = train_all_frozen_models(augmented, T0, dataset_hash, ARTIFACT_DIR)
    for t in trained:
        print(
            f"  {t.spec.model_id}: rows={t.spec.training_row_count} "
            f"model_hash={t.spec.model_hash[:12]}..."
        )

    manifest = build_manifest(T0, [t.spec for t in trained])
    save_manifest(manifest, MANIFEST_PATH)
    print(f"\nsaved manifest: {MANIFEST_PATH}")
    print(f"model_ids trained: {len(trained)} / {len(ALL_MODEL_IDS)}")
    print("\nPhase V3-7 model training complete — models are now FROZEN.")
    print("Do not re-run this script casually; see the STOP guard above.")


if __name__ == "__main__":
    main()
