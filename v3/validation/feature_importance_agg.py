"""Aggregates per-window Feature Importance (spec section 23) into one
mean Gain/Split per feature across all WFO windows - purely descriptive;
the Feature Registry itself is never touched based on this (spec's
explicit prohibition, carried over from Phase V3-2).
"""

from __future__ import annotations

from dataclasses import dataclass

from v3.models.importance import FeatureImportance


@dataclass(frozen=True)
class AggregatedImportance:
    feature: str
    mean_gain: float
    mean_split: float
    n_windows: int


def aggregate_importance_across_windows(
    per_window_importance: list[list[FeatureImportance]],
) -> list[AggregatedImportance]:
    gains: dict[str, list[float]] = {}
    splits: dict[str, list[float]] = {}
    for window_importance in per_window_importance:
        for item in window_importance:
            gains.setdefault(item.feature, []).append(item.gain)
            splits.setdefault(item.feature, []).append(item.split)

    results = [
        AggregatedImportance(
            feature=feature,
            mean_gain=sum(gains[feature]) / len(gains[feature]),
            mean_split=sum(splits[feature]) / len(splits[feature]),
            n_windows=len(gains[feature]),
        )
        for feature in gains
    ]
    return sorted(results, key=lambda r: -r.mean_gain)
