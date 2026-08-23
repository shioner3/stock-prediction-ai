from __future__ import annotations

from v3.features.registry import (
    AVAILABILITY_CONDITIONAL,
    AVAILABILITY_CORE,
    CONDITIONAL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    FEATURE_REGISTRY,
    feature_registry_by_name,
)
from v3.targets.registry import HORIZONS, TARGET_COLUMN_NAMES, TARGET_REGISTRY


def test_feature_registry_names_are_unique() -> None:
    names = [f.name for f in FEATURE_REGISTRY]
    assert len(names) == len(set(names))


def test_feature_registry_every_entry_has_required_fields() -> None:
    for f in FEATURE_REGISTRY:
        assert f.name
        assert f.category
        assert f.formula
        assert f.required_history >= 1
        assert f.availability in (AVAILABILITY_CORE, AVAILABILITY_CONDITIONAL)
        assert "leakage" not in f.leakage_risk.lower() or "none" in f.leakage_risk.lower()
        assert f.source


def test_core_and_conditional_partition_the_registry() -> None:
    assert set(CORE_FEATURE_NAMES) | set(CONDITIONAL_FEATURE_NAMES) == {
        f.name for f in FEATURE_REGISTRY
    }
    assert set(CORE_FEATURE_NAMES).isdisjoint(CONDITIONAL_FEATURE_NAMES)


def test_feature_registry_by_name_lookup() -> None:
    by_name = feature_registry_by_name()
    assert by_name["return_5d"].category == "momentum"
    assert by_name["rsi_5"].category == "oscillator"


def test_target_registry_has_16_entries_4_horizons_x_4_variants() -> None:
    assert len(TARGET_REGISTRY) == 16
    assert len(TARGET_COLUMN_NAMES) == 16
    assert len(TARGET_COLUMN_NAMES) == len(set(TARGET_COLUMN_NAMES))
    for horizon in HORIZONS:
        assert f"target_raw_{horizon}d" in TARGET_COLUMN_NAMES
        assert f"target_topix_relative_{horizon}d" in TARGET_COLUMN_NAMES
        assert f"target_vol_adjusted_{horizon}d" in TARGET_COLUMN_NAMES
        assert f"target_risk_adjusted_{horizon}d" in TARGET_COLUMN_NAMES


def test_every_target_uses_future_data() -> None:
    # Targets are explicitly the future-looking side of the Feature/Target
    # boundary (spec section 2) - this is correct, not a leakage bug.
    assert all(t.uses_future_data for t in TARGET_REGISTRY)
