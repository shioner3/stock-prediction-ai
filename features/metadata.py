"""Lightweight per-feature metadata.

Enough to document each feature's formula and warmup period in one place,
and to let tests assert warmup/NaN behaviour generically across every
feature (see tests/test_features_pipeline.py). Deliberately NOT a
plugin/registry system - features/pipeline.py still just calls plain
functions in a fixed order; this is only a declarative side-list for
documentation and testing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMeta:
    name: str
    description: str
    formula: str
    lookback: int
    required_columns: tuple[str, ...]
    # Minimum number of price observations (rows, 1-indexed count) needed
    # before the value is non-NaN. E.g. warmup_period=20 means rows
    # 0..18 (the first 19 rows) are NaN and row index 19 (the 20th
    # observation) is the first valid value.
    warmup_period: int
    # Informational only in Phase 2 - not consumed by any code yet.
    # "bullish_high": higher value conventionally read as more bullish.
    # "neutral": no inherent directional reading.
    directionality: str
