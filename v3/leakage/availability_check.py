"""Mechanical AVAILABLE_AT<=t verification (spec section 6): every V3
Feature must be computed using ONLY backward-looking operations. This
module statically (via AST) scans `v3/features/*.py` for:

  A. Any `.shift(...)` call with a NEGATIVE argument - a forward read
     (features/breakout.py's own module docstring calls this out as the
     one operation that must never appear in a Feature computation).
  B. Any import of `targets.forward_returns` - the ONLY module in this
     project allowed to look forward. Importing it into a Feature module
     would be the clearest possible sign of Target leakage, mirroring
     `tests/test_target_leakage.py`'s existing check for V1's own
     features/signals/scoring packages.

A feature failing either check is LEAKAGE_FOUND (spec section 23/34) and
must halt Phase development for investigation, never be silently patched
around. Scoped to `v3/features/` only - `v3/targets/` is EXPECTED to read
forward (see v3/targets/registry.py's module docstring on the Feature/
Target boundary), so it is never scanned by this check.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

V3_FEATURES_DIR = Path("v3/features")


@dataclass(frozen=True)
class LeakageFinding:
    file: str
    line: int
    reason: str


def _literal_int(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _negative_shift_calls(tree: ast.AST, filename: str) -> list[LeakageFinding]:
    findings = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
        ):
            continue
        candidates = list(node.args) + [
            kw.value for kw in node.keywords if kw.arg in (None, "periods")
        ]
        for arg in candidates:
            value = _literal_int(arg)
            if value is not None and value < 0:
                findings.append(
                    LeakageFinding(filename, node.lineno, f"shift({value}) - forward read")
                )
    return findings


def _forward_return_imports(tree: ast.AST, filename: str) -> list[LeakageFinding]:
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "forward_returns" in node.module:
            findings.append(LeakageFinding(filename, node.lineno, f"imports {node.module}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "forward_returns" in alias.name:
                    findings.append(LeakageFinding(filename, node.lineno, f"imports {alias.name}"))
    return findings


def check_v3_features_no_forward_reads(
    features_dir: Path = V3_FEATURES_DIR,
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    for path in sorted(features_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        findings.extend(_negative_shift_calls(tree, str(path)))
        findings.extend(_forward_return_imports(tree, str(path)))
    return findings
