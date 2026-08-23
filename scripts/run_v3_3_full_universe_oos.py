"""Phase V3-3 CLI entry point: Full Universe Walk-Forward OOS Validation
of the FROZEN Phase V3-1/V3-2 Feature Registry, Target Registry, and
Model A/B/C structures (spec section 33 - runs to completion, then stops;
never proceeds to Feature/Hyperparameter tuning, Ensembling, V1
integration, or a UI).

Usage:
    python scripts/run_v3_3_full_universe_oos.py [--limit-tickers N]
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
from config.loader import load_app_config  # noqa: E402
from storage.parquet_store import load_ohlcv  # noqa: E402
from v2.config.loader import load_v2_config  # noqa: E402
from v3.config.loader import load_v3_config  # noqa: E402
from v3.dataset import build_v3_dataset, load_universe_tickers  # noqa: E402
from v3.hash import (  # noqa: E402
    current_v3_code_hash,
    current_v3_config_hash,
    current_v3_feature_hash,
    hash_dataframe,
)
from v3.leakage.availability_check import check_v3_features_no_forward_reads  # noqa: E402
from v3.validation.decision import V3_3DecisionInputs, classify_v3_3_decision  # noqa: E402
from v3.validation.leakage_check import run_full_universe_shock_checks  # noqa: E402
from v3.validation.orchestrator import (  # noqa: E402
    PRIMARY_TARGET_COL,
    V3_3Report,
    run_v3_3_validation,
)
from v3.validation.wfo_config import MARKET_REGIME_CONFIG  # noqa: E402
from v3.validation.windows import get_v3_3_windows  # noqa: E402

LEAKAGE_SHOCK_CUTOFF = datetime.date(2024, 6, 1)


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


def _build_decision_inputs(report: V3_3Report) -> V3_3DecisionInputs:
    primary = report.primary
    holding_spreads = [
        r.q5_q1_spread for r in primary.per_window_ranking if r.q5_q1_spread is not None
    ]
    window_direction_agreement = (
        sum(1 for s in holding_spreads if s > 0) / len(holding_spreads)
        if holding_spreads
        else None
    )
    q5_perm = next((p.result.p_value for p in primary.permutation if p.bucket_label == "Q5"), None)
    fdr_key = f"horizon:{PRIMARY_TARGET_COL}:Q5"
    fdr_result = report.fdr_results.get(fdr_key)
    event_spreads = [
        s.q5_q1_spread for label, s in primary.event.items()
        if label != "full_period" and s.q5_q1_spread is not None
    ]
    survives_event = all(s > 0 for s in event_spreads) if event_spreads else None

    return V3_3DecisionInputs(
        n_windows=len(report.windows),
        primary_q5_q1_spread=primary.pooled_ranking.q5_q1_spread,
        rank_ic_mean=primary.pooled_ranking.ic_summary.mean_ic,
        window_direction_agreement=window_direction_agreement,
        day_cluster_ci_low=primary.spread_bootstrap.day_cluster.ci_low,
        block_ci_low=primary.spread_bootstrap.block.ci_low,
        q5_permutation_p=q5_perm,
        fdr_significant=fdr_result.significant if fdr_result else None,
        survives_event_exclusion=survives_event,
        top5_mean_return=primary.topn[5].base.stats.mean_return,
        top10_mean_return=primary.topn[10].base.stats.mean_return,
        top20_mean_return=primary.topn[20].base.stats.mean_return,
        random_baseline_spread=report.benchmarks["random"].pooled_ranking.q5_q1_spread,
        momentum_baseline_spread=report.benchmarks["momentum"].pooled_ranking.q5_q1_spread,
        v2_score_baseline_spread=report.benchmarks["v2_score"].pooled_ranking.q5_q1_spread,
    )


def _print_summary(report: V3_3Report, decision) -> None:
    print(f"\nn_tickers: {report.n_tickers}")
    print(f"n_windows: {len(report.windows)}")
    for w in report.windows:
        print(
            f"  window {w.index}: train {w.train_start}..{w.train_end} "
            f"oos {w.oos_start}..{w.oos_end}"
        )

    print(f"\n--- Primary ({PRIMARY_TARGET_COL}) ---")
    print(f"  pooled Q5-Q1 spread: {report.primary.pooled_ranking.q5_q1_spread}")
    print(f"  pooled Rank IC mean: {report.primary.pooled_ranking.ic_summary.mean_ic}")
    print(f"  regression R2: {report.primary.regression_metrics.r2}")
    for w, r in zip(report.windows, report.primary.per_window_ranking, strict=True):
        print(f"    window {w.index}: spread={r.q5_q1_spread} IC={r.ic_summary.mean_ic}")

    print("\n  Top-N:")
    for n, m in report.primary.topn.items():
        print(
            f"    top{n}: mean={m.base.stats.mean_return} cumret={m.cumulative_return} "
            f"maxdd={m.max_drawdown} sharpe={m.sharpe} pf={m.base.stats.profit_factor}"
        )

    print("\n  Bootstrap (Q5-Q1 spread):")
    sb = report.primary.spread_bootstrap
    print(f"    day_cluster: [{sb.day_cluster.ci_low}, {sb.day_cluster.ci_high}]")
    print(f"    block:       [{sb.block.ci_low}, {sb.block.ci_high}]")

    print("\n  Permutation:")
    for p in report.primary.permutation:
        print(f"    {p.bucket_label}: p={p.result.p_value}")

    print("\n  Regime:")
    for rs in report.primary.regime:
        print(f"    {rs.label}: n={rs.n} spread={rs.q5_q1_spread}")

    print("\n  Benchmarks:")
    for label, b in report.benchmarks.items():
        print(f"    {label}: spread={b.pooled_ranking.q5_q1_spread}")

    print(f"\n--- Secondary targets ({len(report.secondary)}) ---")
    for target_col, secondary_result in report.secondary.items():
        print(
            f"  {target_col}: spread={secondary_result.pooled_ranking.q5_q1_spread} "
            f"IC={secondary_result.pooled_ranking.ic_summary.mean_ic}"
        )

    print(f"\n--- FDR ({len(report.fdr_results)} tests) ---")
    for key, fdr in sorted(report.fdr_results.items(), key=lambda kv: kv[1].raw_p_value):
        print(f"  {key:<30} raw_p={fdr.raw_p_value:.4f} adj_p={fdr.adjusted_p_value:.4f}")

    print(f"\n=== Decision: {decision.decision.value} ===")
    for reason in decision.reasons:
        print(f"  - {reason}")
    print(
        f"  A(predictive power)={decision.question_a_predictive_power} "
        f"B(ranking valid)={decision.question_b_ranking_valid} "
        f"C(Top-N viable)={decision.question_c_topn_viable} "
        f"D(beats V1/V2)={decision.question_d_beats_v1_v2}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase V3-3 Full Universe WFO OOS Validation")
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
    v2_config = load_v2_config()

    print("STEP 1: Repository confirm")
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    allowed_modified_roots = {".gitignore", "README.md", "pyproject.toml"}
    allowed_modified_prefixes = (
        "v3/", "tests/test_v3", "scripts/train_v3", "scripts/build_v3", "scripts/run_v3",
    )
    tracked_changes = [
        line
        for line in status.splitlines()
        if not line.startswith("??")
        and line[3:].strip() not in allowed_modified_roots
        and not line[3:].strip().startswith(allowed_modified_prefixes)
    ]
    print(f"  tracked changes to V1/V2 files (should be 0): {len(tracked_changes)}")
    if tracked_changes:
        print("STOP: tracked V1/V2 files were modified")
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

    tickers = load_universe_tickers(v3_config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    print(f"\nSTEP 3: building Full Universe dataset ({len(tickers)} tickers)")
    dataset = build_v3_dataset(tickers, v3_config)
    print(f"  dataset rows: {len(dataset)}, columns: {len(dataset.columns)}")
    print(f"  date range: {dataset['date'].min()} .. {dataset['date'].max()}")
    dataset_hash = hash_dataframe(dataset)
    print(f"  dataset_hash: {dataset_hash}")

    windows = get_v3_3_windows()
    print(f"\nSTEP 4: WFO windows ({len(windows)})")
    for w in windows:
        print(
            f"  window {w.index}: train {w.train_start}..{w.train_end} "
            f"oos {w.oos_start}..{w.oos_end}"
        )
    if len(windows) < 3:
        print("STOP: fewer than 3 OOS windows - WFO structure does not hold")
        sys.exit(1)

    if not args.skip_leakage_shock:
        print("\nSTEP 5: Full Universe Leakage re-verification (4 shock types)")
        shock_results = run_full_universe_shock_checks(
            tickers, v3_config, LEAKAGE_SHOCK_CUTOFF, Path("data/v3/tmp_shock_check"), dataset
        )
        for r in shock_results:
            print(
                f"  {r.label}: compared={r.n_rows_compared} mismatches={r.n_mismatches} "
                f"passed={r.passed}"
            )
        if not all(r.passed for r in shock_results):
            print("STOP: LEAKAGE_FOUND in Full Universe shock check")
            sys.exit(1)
    else:
        print("\nSTEP 5: Full Universe Leakage re-verification SKIPPED (--skip-leakage-shock)")

    topix = load_ohlcv("TOPIX", v3_config.source_processed_dir)

    print("\nSTEP 6: Full Universe WFO training + evaluation (this will take a while)...")
    report = run_v3_3_validation(
        dataset, tickers, v2_config, topix, MARKET_REGIME_CONFIG, windows=windows
    )

    decision_inputs = _build_decision_inputs(report)
    decision = classify_v3_3_decision(decision_inputs)
    _print_summary(report, decision)

    save_path = Path("data/v3/reports/v3_3_full_universe_oos_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": dataclasses.asdict(report),
        "decision_inputs": dataclasses.asdict(decision_inputs),
        "decision": dataclasses.asdict(decision),
        "code_hash": current_v3_code_hash(),
        "config_hash": current_v3_config_hash(),
        "feature_hash": current_v3_feature_hash(),
        "dataset_hash": dataset_hash,
    }
    save_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {save_path}")
    print("\nPhase V3-3 complete — stopped after Full Universe OOS Decision.")


if __name__ == "__main__":
    main()
