from __future__ import annotations

from v3.config.loader import load_v3_config


def test_load_v3_config_defaults() -> None:
    config = load_v3_config()
    assert config.horizons.days == [5, 10, 15, 20]
    assert set(config.universe.segments) == {"Prime", "Standard", "Growth"}


def test_v3_source_dirs_are_read_only_v1_locations() -> None:
    config = load_v3_config()
    assert "phase7" in str(config.source_processed_dir)
    assert "phase7" in str(config.source_features_dir)


def test_v3_output_dirs_are_isolated_under_data_v3() -> None:
    config = load_v3_config()
    for path in (
        config.v3_dataset_dir, config.v3_manifests_dir, config.v3_models_dir, config.v3_reports_dir,
    ):
        assert str(path).replace("\\", "/").startswith("data/v3/")
