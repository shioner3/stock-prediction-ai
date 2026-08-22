"""Signal Architecture: a Signal answers exactly one question - "did this
setup occur on this date?" - and nothing else.

A Signal must never carry a return, expected return, score, or any
forward-looking value. That separation (Signal vs Score) is enforced
structurally here: SignalRecord has no field for anything but
triggered/not-triggered, and every compute_*_signal() function below
takes a Feature panel (already-computed, backward-looking columns only)
and returns a plain boolean Series. Score is not implemented until
Phase 5, and even then, Score reads Signal output - Signal must never
read Score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class SignalMeta:
    """Identity and required inputs for one Signal - used by the registry
    (signals/registry.py) to iterate over all 12 Signals generically in
    the pipeline and in tests, and to fail loudly if a Feature panel is
    missing a column a Signal needs, rather than silently producing an
    all-False/all-NaN result.
    """

    name: str
    direction: Direction
    description: str
    required_columns: tuple[str, ...]
    signal_version: str = "v1"


def require_columns(panel: pd.DataFrame, meta: SignalMeta) -> None:
    missing = [c for c in meta.required_columns if c not in panel.columns]
    if missing:
        raise ValueError(
            f"signal '{meta.name}' requires columns {missing} that are missing from the "
            "Feature panel - compute_feature_panel() must run (with a matching config) "
            "before signals can be evaluated"
        )
