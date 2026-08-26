from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from v2.config.loader import V2Config, load_v2_config


def test_default_config_file_loads_and_weights_sum_to_one() -> None:
    config = load_v2_config()
    weights = config.score_weights
    total = (
        weights.momentum + weights.trend + weights.volume
        + weights.relative_strength + weights.pullback + weights.volatility
    )
    assert total == pytest.approx(1.0)


def test_default_config_matches_spec_forward_windows() -> None:
    config = load_v2_config()
    assert config.forward_windows == [5, 10, 15, 20]


def test_load_v2_config_rejects_weights_not_summing_to_one(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_v2_settings.yaml"
    bad_path.write_text(
        yaml.safe_dump({"score_weights": {"momentum": 0.5, "trend": 0.6}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must sum to 1.0"):
        load_v2_config(bad_path)


def test_v2config_is_independent_pydantic_model() -> None:
    """V2Config must not be (or inherit from) V1's AppConfig - genuinely
    separate config trees (spec section 20).
    """
    from config.loader import AppConfig

    assert not issubclass(V2Config, AppConfig)
