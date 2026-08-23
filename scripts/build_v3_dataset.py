"""Phase V3-1 CLI entry point: small-subset V3 Dataset generation (spec
section 35.9) - NOT a Full Universe run. Phase V3-1 explicitly stops
before Full-Universe/ML work (spec section 38); this script exists to
demonstrate and sanity-check the Dataset/Feature/Target/Leakage pipeline
end to end on a small ticker subset, and to produce the V3 manifest
(hash) for that subset's dataset.

Usage:
    python scripts/build_v3_dataset.py --limit-tickers 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from v3.config.loader import load_v3_config  # noqa: E402
from v3.dataset import build_v3_dataset, load_universe_tickers  # noqa: E402
from v3.features.registry import CORE_FEATURE_NAMES  # noqa: E402
from v3.hash import build_v3_manifest, save_v3_manifest  # noqa: E402
from v3.leakage.availability_check import check_v3_features_no_forward_reads  # noqa: E402
from v3.targets.registry import HORIZONS, TARGET_COLUMN_NAMES  # noqa: E402

DEFAULT_SUBSET_SIZE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small-subset Phase V3-1 Dataset")
    parser.add_argument("--limit-tickers", type=int, default=DEFAULT_SUBSET_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1_config = load_app_config(Path("config/settings.yaml"))
    setup_logging(v1_config.logging.level, v1_config.logging.log_dir)
    v3_config = load_v3_config()

    print("STEP 1: Repository confirm")
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    # .gitignore/README.md are the two files every prior Phase (V2-1's own
    # precedent) legitimately edits (a new data/v*/ ignore rule, a new
    # README section) - the actual invariant is "no V1/V2 SOURCE file
    # changed", not "the git tree is pristine".
    allowed_modified = {".gitignore", "README.md"}
    tracked_changes = [
        line
        for line in status.splitlines()
        if not line.startswith("??") and line[3:].strip() not in allowed_modified
    ]
    print(f"  tracked changes to V1/V2 files (should be 0): {len(tracked_changes)}")
    if tracked_changes:
        print("STOP: tracked V1/V2 files were modified - non-modification check failed")
        for line in tracked_changes:
            print(f"  {line}")
        sys.exit(1)

    print("\nSTEP 2: Mechanical Feature leakage check (v3/features/*.py, AST scan)")
    findings = check_v3_features_no_forward_reads()
    print(f"  findings: {len(findings)}")
    if findings:
        for f in findings:
            print(f"  LEAKAGE_FOUND: {f.file}:{f.line} - {f.reason}")
        sys.exit(1)

    tickers = load_universe_tickers(v3_config)[: args.limit_tickers]
    print(f"\nSTEP 3: building small-subset dataset ({len(tickers)} tickers, NOT Full Universe)")
    dataset = build_v3_dataset(tickers, v3_config)
    print(f"  dataset rows: {len(dataset)}")
    print(f"  dataset columns: {len(dataset.columns)}")
    if not dataset.empty:
        print(f"  date range: {dataset['date'].min()} .. {dataset['date'].max()}")

    print("\nSTEP 4: saving dataset + manifest")
    dataset_path = v3_config.v3_dataset_dir / f"v3_dataset_subset_{len(tickers)}tickers.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dataset_path, index=False)
    print(f"  saved dataset: {dataset_path}")

    manifest = build_v3_manifest(
        universe_size=len(tickers),
        date_range_start=dataset["date"].min() if not dataset.empty else None,
        date_range_end=dataset["date"].max() if not dataset.empty else None,
        feature_list=CORE_FEATURE_NAMES,
        target_list=TARGET_COLUMN_NAMES,
        horizons=list(HORIZONS),
        dataset=dataset,
    )
    manifest_path = v3_config.v3_manifests_dir / f"v3_manifest_subset_{len(tickers)}tickers.json"
    save_v3_manifest(manifest, manifest_path)
    print(f"  saved manifest: {manifest_path}")
    print(f"  code_hash={manifest.code_hash}")
    print(f"  config_hash={manifest.config_hash}")
    print(f"  feature_hash={manifest.feature_hash}")
    print(f"  dataset_hash={manifest.dataset_hash}")

    print("\nPhase V3-1 small-subset dataset generation complete.")
    print("Full Universe run / ML training NOT performed (spec section 38 - out of scope).")


if __name__ == "__main__":
    main()
