"""Phase 9/11 section 19: static (AST-based) verification that the only
allowed dependency direction is

    Signal -> Backtest / Analysis

never the reverse. Mirrors tests/test_target_leakage.py's existing
pattern (which guards targets/ the same way) but covers Phase 8/9/11's
analysis-only modules specifically: features/ and signals/ must never
import pipeline.run_phase8_analysis, pipeline.run_phase9_analysis,
pipeline.run_phase11_research, or any of the new Phase 8/9 backtest/
analysis modules, since importing one of these INTO Signal generation
would be exactly the "analysis result feeds back into Signal" loop the
spec forbids - this is also Phase 11 section 32's prohibition #1/#7
(Signals must never change based on analysis results) at the import-graph
level.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_DIRS = ["features", "signals"]

FORBIDDEN_MODULES = [
    "pipeline.run_phase8_analysis",
    "pipeline.run_phase9_analysis",
    "pipeline.run_phase11_research",
    "pipeline.run_phase12_ensemble",
    "pipeline.run_phase13_conditional_analysis",
    "pipeline.run_phase14_validation",
    "ensemble.signal_count",
    "ensemble.combinations",
    "ensemble.frequency",
    "ensemble.decision",
    "ensemble.portfolio_sim",
    "backtest.day_cluster_bootstrap",
    "backtest.ticker_cluster_bootstrap",
    "backtest.block_bootstrap",
    "backtest.episode_analysis",
    "backtest.event_concentration",
    "backtest.timing_shift",
    "backtest.leave_one_period_out",
    "backtest.regime_reproducibility",
    "backtest.conditional_edge_decision",
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


def test_signal_generation_never_imports_phase8_or_phase9_analysis_modules() -> None:
    offending: list[str] = []
    for dirname in CHECKED_DIRS:
        for path in (REPO_ROOT / dirname).rglob("*.py"):
            for name in _imported_module_names(path):
                if name in FORBIDDEN_MODULES or any(
                    name.startswith(f"{m}.") for m in FORBIDDEN_MODULES
                ):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")

    assert not offending, (
        f"features/ and signals/ must never import a Phase 8/9 analysis module "
        f"(analysis results must never feed back into Signal generation): {offending}"
    )


def test_phase9_analysis_module_imports_from_signals_not_the_reverse() -> None:
    """Sanity check on the ALLOWED direction: pipeline/run_phase9_analysis.py
    itself should import FROM signals/backtest (to read already-computed
    Signal output), confirming the module actually exercises the
    dependency it's meant to - not just that the forbidden direction is
    literally absent because the file barely imports anything.
    """
    path = REPO_ROOT / "pipeline" / "run_phase9_analysis.py"
    names = _imported_module_names(path)
    assert any(n.startswith("backtest.") for n in names)
    assert any(n.startswith("storage.") for n in names)


def test_phase11_research_module_imports_from_signals_not_the_reverse() -> None:
    path = REPO_ROOT / "pipeline" / "run_phase11_research.py"
    names = _imported_module_names(path)
    assert any(n.startswith("backtest.") for n in names)
    assert any(n.startswith("pipeline.") for n in names)


def test_phase12_ensemble_module_imports_from_signals_not_the_reverse() -> None:
    path = REPO_ROOT / "pipeline" / "run_phase12_ensemble.py"
    names = _imported_module_names(path)
    assert any(n.startswith("backtest.") for n in names)
    assert any(n.startswith("ensemble.") for n in names)
    assert any(n == "signals.registry" for n in names)


def test_phase13_conditional_analysis_module_imports_from_signals_not_the_reverse() -> None:
    path = REPO_ROOT / "pipeline" / "run_phase13_conditional_analysis.py"
    names = _imported_module_names(path)
    assert any(n.startswith("backtest.") for n in names)
    assert any(n.startswith("targets.") for n in names)
    assert any(n.startswith("ensemble.") for n in names)


def test_phase14_validation_module_imports_from_signals_not_the_reverse() -> None:
    path = REPO_ROOT / "pipeline" / "run_phase14_validation.py"
    names = _imported_module_names(path)
    assert any(n.startswith("backtest.") for n in names)
    assert any(n.startswith("pipeline.run_phase13_conditional_analysis") for n in names)
    assert any(n.startswith("pipeline.run_phase9_analysis") for n in names)


def test_ensemble_package_never_imported_by_signals_or_features() -> None:
    """The new ensemble/ package (Phase 12) reads Signal/Score output -
    it must never be imported the other way, same rule as Phase 8/9/11's
    analysis modules.
    """
    offending: list[str] = []
    for dirname in CHECKED_DIRS:
        for path in (REPO_ROOT / dirname).rglob("*.py"):
            for name in _imported_module_names(path):
                if name.startswith("ensemble"):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")
    assert not offending, f"features/signals must never import ensemble/: {offending}"
