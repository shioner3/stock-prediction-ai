"""Phase V3-2 CLI entry point: train Baseline Models A/B/C on a small
ticker subset (spec section 9/18 - NOT Full Universe, NOT WFO). Verifies
the Dataset -> time_split -> fit -> predict -> basic evaluation ->
cross-sectional check -> Random baseline -> Feature importance pipeline
works end to end, and records the results as reference information only -
no conclusion about model usefulness is drawn here (spec section 24).

Usage:
    python scripts/train_v3_baseline.py --limit-tickers 40
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from v3.config.loader import load_v3_config  # noqa: E402
from v3.dataset import build_v3_dataset, load_universe_tickers, time_split  # noqa: E402
from v3.hash import hash_dataframe  # noqa: E402
from v3.leakage.availability_check import check_v3_features_no_forward_reads  # noqa: E402
from v3.models.data_prep import prepare_training_set  # noqa: E402
from v3.models.model_manifest import build_model_manifest, save_model_manifest  # noqa: E402
from v3.models.orchestrator import run_model_a, run_model_b, run_model_c  # noqa: E402
from v3.models.regression import fit_regression_model  # noqa: E402

DEFAULT_SUBSET_SIZE = 40
PRIMARY_TARGET = "target_raw_5d"
SECONDARY_TARGETS = (
    "target_raw_10d", "target_topix_relative_5d", "target_vol_adjusted_5d",
    "target_risk_adjusted_5d",
)


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase V3-2 Baseline Models")
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
    # V3 is free to modify its OWN files (v3/, tests/test_v3_*, scripts/
    # *_v3_*.py - spec section 2 "V3だけを変更する") plus the two root
    # files every prior Phase legitimately touches (.gitignore/README.md)
    # and pyproject.toml (this Phase's own dependency addition, spec
    # section 3). The real invariant is that nothing OUTSIDE that set -
    # i.e. any V1 or V2 source file - changed.
    allowed_modified_roots = {".gitignore", "README.md", "pyproject.toml"}
    allowed_modified_prefixes = ("v3/", "tests/test_v3", "scripts/train_v3", "scripts/build_v3")
    tracked_changes = [
        line
        for line in status.splitlines()
        if not line.startswith("??")
        and line[3:].strip() not in allowed_modified_roots
        and not line[3:].strip().startswith(allowed_modified_prefixes)
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
    print(f"  dataset rows: {len(dataset)}, columns: {len(dataset.columns)}")
    dataset_hash = hash_dataframe(dataset)
    print(f"  dataset_hash: {dataset_hash}")

    dates = sorted(dataset["date"].unique())
    split_idx = int(len(dates) * 0.7)
    train_end = dates[split_idx]
    test_start_idx = min(split_idx + 20, len(dates) - 1)  # 20-day embargo (>= largest Horizon)
    test_start = dates[test_start_idx]
    train, test = time_split(dataset, train_end=train_end, test_start=test_start)
    print(f"\nSTEP 4: time split - TRAIN {len(train)} rows ({dates[0]}..{train_end}), "
          f"TEST {len(test)} rows ({test_start}..{dates[-1]})")

    print(f"\nSTEP 5: training Model A/B/C on PRIMARY target ({PRIMARY_TARGET})")
    result_a = run_model_a(train, test, PRIMARY_TARGET)
    print(f"  Model A train: {result_a.train_metrics}")
    print(f"  Model A test:  {result_a.test_metrics}")
    print(f"  Model A cross-sectional Q5-Q1 spread: {result_a.cross_sectional.q5_q1_spread}")
    print(f"  Random baseline Q5-Q1 spread:         "
          f"{result_a.random_baseline_cross_sectional.q5_q1_spread}")
    top5_gain = sorted(result_a.feature_importance, key=lambda f: -f.gain)[:5]
    print(f"  top 5 features by gain: {[(f.feature, round(f.gain, 2)) for f in top5_gain]}")

    result_b = run_model_b(train, test, PRIMARY_TARGET)
    print(f"\n  Model B test: {result_b.test_metrics}")

    result_c = run_model_c(train, test, PRIMARY_TARGET)
    print(f"\n  Model C predictions: {len(result_c.quantile_predictions)} rows, "
          f"columns={list(result_c.quantile_predictions.columns)}")

    print(f"\nSTEP 6: target-switch check ({len(SECONDARY_TARGETS)} secondary targets)")
    secondary_results = {}
    for target_col in SECONDARY_TARGETS:
        r = run_model_a(train, test, target_col)
        secondary_results[target_col] = r
        print(f"  {target_col}: test n={r.test_metrics.n} spearman={r.test_metrics.spearman}")

    print("\nSTEP 7: saving model manifest")
    train_set = prepare_training_set(train, PRIMARY_TARGET)
    model_a_refit = fit_regression_model(train_set.X, train_set.y)
    manifest = build_model_manifest(
        "regression", PRIMARY_TARGET, model_a_refit, dataset_hash, train, test
    )
    manifest_path = v3_config.v3_models_dir / f"v3_2_model_a_{PRIMARY_TARGET}.json"
    save_model_manifest(manifest, manifest_path)
    print(f"  saved: {manifest_path}")
    print(f"  model_hash={manifest.model_hash}")

    print("\nPhase V3-2 baseline model training complete (small subset, NOT Full Universe).")
    print("Phase V3-2 complete — stopped before Full Universe OOS")


if __name__ == "__main__":
    main()
