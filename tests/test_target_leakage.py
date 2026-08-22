"""Target Leakage Test (Phase 5 section 25): statically verify, via AST
inspection (not just convention), that none of features/, signals/, or
scoring/ import targets/forward_returns - or targets/ at all.

This is deliberately independent of tests/test_no_lookahead.py's
existing features-only dependency check (which predates targets/ and
only guards against a module that didn't exist yet in Phase 2/3); this
file is the canonical, Phase 5-complete version covering all three
directories the spec calls out.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_DIRS = ["features", "signals", "scoring"]


def _imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_no_directory_imports_targets() -> None:
    offending: list[str] = []
    for dirname in CHECKED_DIRS:
        for path in (REPO_ROOT / dirname).rglob("*.py"):
            for name in _imported_module_names(path):
                if name == "targets" or name.startswith("targets."):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")

    assert not offending, (
        f"features/, signals/, and scoring/ must never import targets/ "
        f"(research-only, forward-looking data): {offending}"
    )


def test_targets_module_itself_does_not_import_features_signals_scoring() -> None:
    """The reverse-direction sanity check: targets/forward_returns.py
    should depend only on pandas/numpy over an OHLCV DataFrame, never
    reach back into features/signals/scoring (which would suggest it's
    silently coupled to Score/Signal internals rather than being a pure
    function of raw price data).
    """
    offending: list[str] = []
    for path in (REPO_ROOT / "targets").rglob("*.py"):
        for name in _imported_module_names(path):
            if name.split(".")[0] in ("features", "signals", "scoring"):
                offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")

    assert not offending, f"targets/ must not depend on features/signals/scoring: {offending}"
