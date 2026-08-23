"""Phase V3-2 Leakage tests (spec section 16/21):

- Target exclusion test: the mechanical check
  (v3/models/data_prep.py::assert_no_target_leakage_in_features) against
  the REAL CORE_FEATURE_NAMES/TARGET_COLUMN_NAMES registries.
- Future shock model test: shock OHLCV strictly AFTER train_end (so the
  TRAINING split is completely untouched), rebuild the dataset, retrain,
  and confirm (a) the retrained model is BIT-IDENTICAL (same model_hash -
  proof the shock never reached training) and (b) every TEST-set
  prediction for a row dated <= the shock cutoff is unchanged (proof the
  already-leak-free V3-1 features, not the model, are what's being
  re-verified end to end through the model pipeline).
- V1/V2 isolation test: `v3/models/*.py` never imports V1's decision
  layers (signals/scoring/backtest/forward_test/ensemble) or writes into
  v2/.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from conftest import make_synthetic_ohlcv

from storage.parquet_store import load_ohlcv, save_ohlcv
from v3.config.loader import load_v3_config
from v3.dataset import build_v3_dataset, time_split
from v3.features.registry import CORE_FEATURE_NAMES
from v3.leakage.shock_tests import shock_price_after
from v3.models.data_prep import assert_no_target_leakage_in_features, prepare_training_set
from v3.models.model_manifest import compute_model_hash
from v3.models.regression import fit_regression_model, predict_regression
from v3.targets.registry import TARGET_COLUMN_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
# "scoring" is intentionally NOT a blanket-blocked directory here:
# scoring/scorer.py + scoring/pipeline.py compute V1's actual trading
# Score (decision-relevant, blocked below by full module path), but
# scoring/validation.py is a generic, already-approved-for-reuse
# statistics utility (assign_quantile_buckets() etc.) - V2's own
# orchestrator (v2/validation/orchestrator.py) already imports it
# directly, the same precedent v3/models/cross_sectional.py follows.
V1_DECISION_MODULES = [
    "signals", "scoring.scorer", "scoring.pipeline", "backtest", "forward_test", "ensemble",
]
TARGET_COL = "target_raw_5d"


def test_target_exclusion_check_passes_for_real_registries() -> None:
    assert_no_target_leakage_in_features(CORE_FEATURE_NAMES)


def test_target_exclusion_check_catches_every_registered_target() -> None:
    for target_col in TARGET_COLUMN_NAMES:
        try:
            assert_no_target_leakage_in_features([*CORE_FEATURE_NAMES, target_col])
        except ValueError:
            continue
        raise AssertionError(f"{target_col} was not detected as leaked into features")


def test_v3_models_never_imports_v1_decision_layers() -> None:
    offending = []
    for path in (REPO_ROOT / "v3" / "models").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == d or name.startswith(f"{d}.") for d in V1_DECISION_MODULES):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")
    assert not offending, offending


def _build_config(tmp_path: Path, n_tickers: int = 8, n_days: int = 500):
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifest.json"

    market = make_synthetic_ohlcv(n_days, seed=999, ticker="TOPIX")
    save_ohlcv(market, "TOPIX", processed_dir)

    tickers = [f"T{i}" for i in range(n_tickers)]
    manifest_tickers = {}
    for i, ticker in enumerate(tickers):
        ohlcv = make_synthetic_ohlcv(n_days, seed=i + 1, ticker=ticker)
        save_ohlcv(ohlcv, ticker, processed_dir)
        manifest_tickers[ticker] = {"included_in_universe": True}
    manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

    config = load_v3_config().model_copy(
        update={"source_universe_manifest": manifest_path, "source_processed_dir": processed_dir}
    )
    return config, tickers, processed_dir


def test_future_price_shock_never_changes_model_or_earlier_predictions(tmp_path: Path) -> None:
    config, tickers, processed_dir = _build_config(tmp_path)
    dataset = build_v3_dataset(tickers, config)
    dates = sorted(dataset["date"].unique())
    train_end = dates[int(len(dates) * 0.7)]
    test_start = dates[min(int(len(dates) * 0.7) + 20, len(dates) - 1)]
    baseline_train, baseline_test = time_split(dataset, train_end=train_end, test_start=test_start)

    baseline_train_set = prepare_training_set(baseline_train, TARGET_COL)
    baseline_model = fit_regression_model(baseline_train_set.X, baseline_train_set.y)

    # Shock cutoff is strictly AFTER train_end (training data untouched)
    # but INSIDE the test period, so this is a meaningful check, not a
    # vacuous one - some test rows are shocked, others (date <= cutoff)
    # are not.
    shock_cutoff = test_start
    shocked_dir = tmp_path / "processed_shocked"
    for ticker in [*tickers, "TOPIX"]:
        ohlcv = load_ohlcv(ticker, processed_dir)
        shocked = shock_price_after(ohlcv, shock_cutoff) if ticker != "TOPIX" else ohlcv
        save_ohlcv(shocked, ticker, shocked_dir)
    shocked_config = config.model_copy(update={"source_processed_dir": shocked_dir})
    shocked_dataset = build_v3_dataset(tickers, shocked_config)
    shocked_train, shocked_test = time_split(
        shocked_dataset, train_end=train_end, test_start=test_start
    )

    shocked_train_set = prepare_training_set(shocked_train, TARGET_COL)
    shocked_model = fit_regression_model(shocked_train_set.X, shocked_train_set.y)

    # (a) training data was entirely before train_end < shock_cutoff, so
    # the retrained model must be bit-identical.
    assert compute_model_hash(baseline_model) == compute_model_hash(shocked_model)

    # (b) predictions for TEST rows dated <= shock_cutoff must be unchanged
    # (there are none here since test_start == shock_cutoff and the test
    # split itself starts there - the meaningful boundary is thus at
    # date == test_start, which by construction is NOT shocked yet since
    # shock_price_after mutates rows with date > cutoff, not >=).
    baseline_at_cutoff = baseline_test[baseline_test["date"] == shock_cutoff]
    shocked_at_cutoff = shocked_test[shocked_test["date"] == shock_cutoff]
    if not baseline_at_cutoff.empty:
        baseline_X = baseline_at_cutoff[CORE_FEATURE_NAMES]
        shocked_X = shocked_at_cutoff[CORE_FEATURE_NAMES]
        baseline_pred_at_cutoff = predict_regression(baseline_model, baseline_X)
        shocked_pred_at_cutoff = predict_regression(shocked_model, shocked_X)
        assert np.array_equal(baseline_pred_at_cutoff, shocked_pred_at_cutoff)
