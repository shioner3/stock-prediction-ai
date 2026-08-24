"""Phase V3-5 CLI entry point: Stock-Specific Residual ML Validation -
retrains Model A (frozen V3-2 Hyperparameters/seed, frozen V3-3 WFO) on
3 market-neutralized Targets (TOPIX-relative/Beta-adjusted Residual/
Sector-relative) to test whether a genuinely stock-specific ranking
edge survives, independent of Phase V3-4's own MARKET_TIMING_EDGE
finding on the Raw target. Runs to completion, then stops (spec section
39 - never proceeds to tuning, V1/V2 integration, or a UI).

Usage:
    python scripts/run_v3_5_residual_validation.py [--limit-tickers N]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import TransactionCostConfig, load_app_config  # noqa: E402
from storage.parquet_store import load_ohlcv  # noqa: E402
from v3.config.loader import load_v3_config  # noqa: E402
from v3.dataset import load_universe_tickers  # noqa: E402
from v3.residual.leakage_check import run_residual_shock_checks  # noqa: E402
from v3.residual.orchestrator import HORIZONS, V3_5Report, run_v3_5_analysis  # noqa: E402
from v3.residual.reproduce import (  # noqa: E402
    TARGET_A_RAW,
    build_augmented_dataset,
    load_v3_3_reference_hashes,
    reproduce_residual_predictions,
    verify_against_v3_3,
    verify_primary_reproduction,
)
from v3.robustness.aux_panel import attach_sector_and_scale  # noqa: E402
from v3.robustness.reproduce import (  # noqa: E402
    build_frozen_dataset,
    load_saved_predictions,
)
from v3.validation.wfo_config import MARKET_REGIME_CONFIG  # noqa: E402
from v3.validation.windows import get_v3_3_windows  # noqa: E402

LEAKAGE_SHOCK_CUTOFF = datetime.date(2024, 6, 1)
V3_4_SAVED_RAW_PREDICTIONS_DIR = Path("data/v3/robustness/predictions")
V3_5_PREDICTIONS_DIR = Path("data/v3/residual/predictions")


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _stringify_tuple_keys(obj: Any) -> Any:
    """`V3_5Report.light_results`/`.residual_strength_by_horizon` are
    keyed by `(definition, horizon)` tuples (a natural, ergonomic Python
    key - see `v3/residual/orchestrator.py`), but `json.dumps` only
    accepts str/int/float/bool/None dict keys (its `default=` hook never
    even runs for keys, only values) - recursively joins any tuple key
    into `"definition:horizon"` before serialization, everywhere in the
    (possibly deeply nested) dataclass-derived structure.
    """
    if isinstance(obj, dict):
        return {
            (":".join(str(part) for part in k) if isinstance(k, tuple) else k):
                _stringify_tuple_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_stringify_tuple_keys(item) for item in obj]
    return obj


def _print_summary(report: V3_5Report) -> None:
    print("\n--- Light battery (all 16 Target x Horizon combos, section 33) ---")
    for (definition, horizon), light in sorted(report.light_results.items()):
        r = light.ranking
        print(
            f"  {definition:<16} {horizon:>2}d  spread={r.q5_q1_spread}  "
            f"rank_ic={r.ic_summary.mean_ic}  pearson_ic={light.pearson_ic_mean}"
        )

    print("\n--- Market Neutralization comparison (section 17/32) ---")
    for row in report.market_neutralization_table:
        print(
            f"  {row.definition:<16} spread={row.q5_q1_spread} rank_ic={row.rank_ic} "
            f"top5_exp={row.top5_expectancy} top5_pf={row.top5_pf} "
            f"day_cluster_low={row.day_cluster_ci_low} block_low={row.block_ci_low} "
            f"perm_p={row.permutation_p} fdr_sig={row.fdr_significant} survives={row.survives}"
        )

    print("\n--- Residual strength (section 19) ---")
    for (definition, horizon), ratio in sorted(report.residual_strength_by_horizon.items()):
        print(f"  {definition:<16} {horizon:>2}d  residual_strength={ratio}")

    print(f"\n--- FDR ({len(report.fdr_results)} tests, section 27) ---")
    for key, fdr in sorted(report.fdr_results.items(), key=lambda kv: kv[1].raw_p_value):
        print(f"  {key:<24} raw_p={fdr.raw_p_value:.4f} adj_p={fdr.adjusted_p_value:.4f}")

    for definition, p in report.primary_results.items():
        print(f"\n--- Primary battery: {definition} ---")
        print(f"  regime_dependent={p.regime.regime_dependent}")
        print(f"  matched_control: n_matched={p.matched_control.n_matched}")
        for col, comp in p.matched_control.comparisons.items():
            print(
                f"    {col}: treatment={comp.treatment_stats.mean_return} "
                f"control={comp.control_stats.mean_return}"
            )
        print(f"  day_cluster_ci=[{p.spread_bootstrap.day_cluster.ci_low}, "
              f"{p.spread_bootstrap.day_cluster.ci_high}]")
        print(f"  block_ci=[{p.spread_bootstrap.block.ci_low}, {p.spread_bootstrap.block.ci_high}]")
        print(f"  permutation_q5_p={p.permutation_q5_p}")

    ec = report.edge_classification
    print(f"\n=== Edge Classification (section 18/31): {ec.classification.value} ===")
    for reason in ec.reasons:
        print(f"  - {reason}")
    print(
        f"  A(topix_rel_pos)={ec.criterion_a_topix_relative_positive} "
        f"B(beta_res_pos)={ec.criterion_b_beta_residual_positive} "
        f"C(bear_excl)={ec.criterion_c_survives_bear_exclusion} "
        f"D(topn_exp)={ec.criterion_d_topn_positive_expectancy} "
        f"E(day_cluster)={ec.criterion_e_day_cluster_ci_positive} "
        f"F(block)={ec.criterion_f_block_ci_positive} "
        f"G(perm)={ec.criterion_g_permutation_significant} "
        f"H(fdr)={ec.criterion_h_fdr_significant} raw_positive={ec.raw_positive}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase V3-5 Residual ML Validation")
    parser.add_argument("--limit-tickers", type=int, default=None, help="dev/testing only")
    parser.add_argument(
        "--skip-leakage-shock", action="store_true",
        help="dev/testing only - skip the slow shock check",
    )
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
    allowed_modified_roots = {".gitignore", "README.md", "pyproject.toml"}
    allowed_modified_prefixes = (
        "v3/", "tests/test_v3", "scripts/train_v3", "scripts/build_v3", "scripts/run_v3",
        "research/phase_v3_5_report.md",
    )
    tracked_changes = [
        line
        for line in status.splitlines()
        if not line.startswith("??")
        and line[3:].strip() not in allowed_modified_roots
        and not line[3:].strip().startswith(allowed_modified_prefixes)
    ]
    print(f"  tracked changes to V1/V2/V3-1..4 files (should be 0): {len(tracked_changes)}")
    if tracked_changes:
        print("STOP: tracked V1/V2/V3-1..4 files were modified")
        for line in tracked_changes:
            print(f"  {line}")
        sys.exit(1)

    print("\nSTEP 2: building Full Universe dataset (identical to V3-3/V3-4's own STEP 3)")
    dataset = build_frozen_dataset(v3_config, limit_tickers=args.limit_tickers)
    tickers = load_universe_tickers(v3_config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    print(f"  dataset rows: {len(dataset)}, columns: {len(dataset.columns)}")

    print("\nSTEP 2b: verifying hash match against Phase V3-3/V3-4's own saved report")
    reference = load_v3_3_reference_hashes()
    verification = verify_against_v3_3(dataset, reference)
    print(f"  config_hash_match={verification.config_hash_match}")
    print(f"  feature_hash_match={verification.feature_hash_match}")
    print(f"  dataset_hash_match={verification.dataset_hash_match}")
    if not verification.all_match:
        print("STOP: V3-3/V3-4 hash mismatch - frozen spec has drifted, per spec section 1/38")
        sys.exit(1)

    windows = get_v3_3_windows()
    print(f"\nSTEP 3: WFO windows ({len(windows)}, identical to V3-3/V3-4)")
    if len(windows) < 3:
        print("STOP: fewer than 3 OOS windows - WFO structure does not hold")
        sys.exit(1)

    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]].drop_duplicates(
        subset=["ticker"]
    )
    print("\nSTEP 4: building augmented dataset (Beta-adjusted Residual / Sector-relative Targets)")
    augmented_dataset = build_augmented_dataset(dataset, sector_map)

    if not args.skip_leakage_shock:
        print("\nSTEP 5: Beta/residual-Target Leakage re-verification (4 shock types)")
        shock_results = run_residual_shock_checks(
            tickers, v3_config, LEAKAGE_SHOCK_CUTOFF, Path("data/v3/tmp_shock_check_v3_5"),
            augmented_dataset,
        )
        for r in shock_results:
            print(
                f"  {r.label}: compared={r.n_rows_compared} mismatches={r.n_mismatches} "
                f"passed={r.passed}"
            )
        if not all(r.passed for r in shock_results):
            print("STOP: LEAKAGE_FOUND in Beta/residual-Target shock check")
            sys.exit(1)
    else:
        print("\nSTEP 5: Leakage re-verification SKIPPED (--skip-leakage-shock)")

    print("\nSTEP 6: loading V3-4's saved Raw-target predictions (Target A, no retraining)")
    raw_predictions = load_saved_predictions(V3_4_SAVED_RAW_PREDICTIONS_DIR)
    predictions_by_combo: dict[tuple[str, int], pd.DataFrame] = {
        (TARGET_A_RAW, horizon): raw_predictions[f"target_raw_{horizon}d"] for horizon in HORIZONS
    }

    print(
        "\nSTEP 6b: verifying reproduced Target A (Raw, 5d) spread matches "
        "V3-3/V3-4's recorded value"
    )
    if not verify_primary_reproduction(predictions_by_combo[(TARGET_A_RAW, 5)]):
        print("STOP: reproduced Target A spread does not match V3-3/V3-4's recorded value")
        sys.exit(1)

    print(
        "\nSTEP 7: training Model A on Targets B/C/D across all 4 Horizons "
        "(this will take a while)..."
    )
    new_predictions = reproduce_residual_predictions(augmented_dataset, windows, HORIZONS)
    predictions_by_combo.update(new_predictions)
    for (definition, horizon), df in sorted(new_predictions.items()):
        print(f"  {definition} {horizon}d: {len(df)} pooled OOS rows")

    V3_5_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for (definition, horizon), df in new_predictions.items():
        df.to_parquet(V3_5_PREDICTIONS_DIR / f"{definition}_{horizon}d.parquet", index=False)

    topix = load_ohlcv("TOPIX", v3_config.source_processed_dir)
    cost_tiers = TransactionCostConfig().tiers

    print("\nSTEP 8: running the Target-comparison / decomposition battery...")
    report = run_v3_5_analysis(
        augmented_dataset=augmented_dataset, predictions_by_combo=predictions_by_combo,
        tickers=tickers, topix_ohlcv=topix, market_regime_config=MARKET_REGIME_CONFIG,
        cost_tiers=cost_tiers, v3_config=v3_config,
    )
    _print_summary(report)

    save_path = Path("data/v3/reports/v3_5_residual_validation_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"report": _stringify_tuple_keys(dataclasses.asdict(report))}
    save_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {save_path}")
    print("\nPhase V3-5 complete — stopped after Residual ML Validation.")


if __name__ == "__main__":
    main()
