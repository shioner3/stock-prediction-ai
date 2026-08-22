"""V2 Leakage Tests (Phase V2-1 spec section 19/23):

1. Static (AST) dependency-direction check - V1 (features/signals/
   scoring/backtest/targets/ensemble/forward_test/every pipeline module
   except v2's own) must NEVER import v2/, mirroring
   tests/test_target_leakage.py's and tests/test_phase9_no_lookahead.py's
   existing pattern. v2/ importing FROM these V1 packages is the
   explicitly ALLOWED direction (reuse via import - spec section 3) and
   is checked separately as a sanity/reuse-is-real check.
2. Future-shock test: changing a LATER date's price must never change
   an EARLIER date's Feature/Rank/Score for ANY ticker, including the
   shocked one itself.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from storage.parquet_store import save_feature_panel, save_ohlcv
from v2.config.loader import V2Config, load_v2_config
from v2.pipeline import run_v2_ranking

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_CHECKED_DIRS = [
    "features", "signals", "scoring", "backtest", "targets", "ensemble", "forward_test",
]
RANK_SCORE_COLUMNS = [
    "total_score", "momentum_rank", "trend_rank", "volume_rank",
    "volatility_rank", "relative_strength_rank", "pullback_rank",
]


def _imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_v1_packages_never_import_v2() -> None:
    offending: list[str] = []
    for dirname in V1_CHECKED_DIRS:
        for path in (REPO_ROOT / dirname).rglob("*.py"):
            for name in _imported_module_names(path):
                if name == "v2" or name.startswith("v2."):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")

    for path in (REPO_ROOT / "pipeline").glob("*.py"):
        for name in _imported_module_names(path):
            if name == "v2" or name.startswith("v2."):
                offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")

    assert not offending, (
        f"V1 modules must never import v2/ (V2 depends on V1, never the reverse): {offending}"
    )


def test_v2_pipeline_imports_from_v1_not_the_reverse() -> None:
    """Sanity check on the ALLOWED direction: v2/pipeline.py should
    genuinely import FROM V1's storage/pipeline.universe_ingest -
    confirming the reuse is real, not just that the forbidden direction
    is absent because v2/ barely imports anything.
    """
    path = REPO_ROOT / "v2" / "pipeline.py"
    names = _imported_module_names(path)
    assert any(n.startswith("storage.") for n in names)
    assert any(n.startswith("pipeline.universe_ingest") for n in names)


def test_v2_features_adapter_imports_v1_compute_feature_panel() -> None:
    path = REPO_ROOT / "v2" / "features_adapter.py"
    names = _imported_module_names(path)
    assert any(n.startswith("features.") for n in names)


def test_v2_targets_adapter_imports_v1_forward_returns() -> None:
    path = REPO_ROOT / "v2" / "targets_adapter.py"
    names = _imported_module_names(path)
    assert any(n.startswith("targets.") for n in names)


@pytest.fixture
def v2_config_factory(tmp_path: Path):
    def _build(shock_last_day: bool) -> V2Config:
        features_dir = tmp_path / f"features_{shock_last_day}"
        processed_dir = tmp_path / f"processed_{shock_last_day}"
        manifest_path = tmp_path / f"manifest_{shock_last_day}.json"

        market = make_synthetic_ohlcv(200, seed=999, ticker="TOPIX")
        save_ohlcv(market, "TOPIX", processed_dir)

        tickers = [f"T{i}" for i in range(10)]
        manifest_tickers = {}
        for i, ticker in enumerate(tickers):
            ohlcv = make_synthetic_ohlcv(200, seed=i + 1, ticker=ticker)
            if shock_last_day and ticker == "T0":
                ohlcv = ohlcv.copy()
                ohlcv.loc[ohlcv.index[-1], "close"] *= 5.0
            panel = compute_feature_panel(ohlcv, market_df=market)
            save_feature_panel(panel, ticker, features_dir)
            manifest_tickers[ticker] = {"included_in_universe": True}
        manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

        return load_v2_config().model_copy(
            update={
                "source_universe_manifest": manifest_path,
                "source_features_dir": features_dir,
                "source_processed_dir": processed_dir,
            }
        )

    return _build


def test_future_shock_never_changes_earlier_dates(v2_config_factory) -> None:
    baseline_config = v2_config_factory(shock_last_day=False)
    shocked_config = v2_config_factory(shock_last_day=True)
    tickers = [f"T{i}" for i in range(10)]

    baseline = run_v2_ranking(baseline_config, tickers=tickers)
    shocked = run_v2_ranking(shocked_config, tickers=tickers)

    early_dates = sorted(baseline["date"].unique())[:-1]
    b = (
        baseline[baseline["date"].isin(early_dates)]
        .sort_values(["date", "ticker"]).reset_index(drop=True)
    )
    s = (
        shocked[shocked["date"].isin(early_dates)]
        .sort_values(["date", "ticker"]).reset_index(drop=True)
    )

    assert list(b["ticker"]) == list(s["ticker"])
    for col in RANK_SCORE_COLUMNS:
        assert np.allclose(
            b[col].to_numpy(dtype=float), s[col].to_numpy(dtype=float), equal_nan=True
        ), f"{col} changed on a date earlier than the shock - leakage"
