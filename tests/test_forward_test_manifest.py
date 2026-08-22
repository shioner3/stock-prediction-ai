from __future__ import annotations

from datetime import date
from pathlib import Path

from config.loader import AppConfig
from forward_test.manifest import (
    build_manifest,
    compute_strategy_hashes,
    load_manifest_raw,
    save_manifest,
    verify_strategy_hashes_unchanged,
)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {"start_date": "2020-01-01"},
            "universe": {"master_list_path": "data/reference/jpx_listed_companies.sample.csv"},
        }
    )


def test_compute_strategy_hashes_is_deterministic() -> None:
    a = compute_strategy_hashes()
    b = compute_strategy_hashes()
    assert a == b


def test_compute_strategy_hashes_fields_are_nonempty_hex() -> None:
    hashes = compute_strategy_hashes()
    for value in (
        hashes.features_hash, hashes.signals_hash, hashes.scoring_hash,
        hashes.backtest_engine_hash, hashes.market_regime_hash, hashes.config_hash,
    ):
        assert isinstance(value, str)
        assert len(value) == 64  # sha256 hex digest length
        int(value, 16)  # must actually be hex


def test_build_manifest_captures_frozen_settings(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = build_manifest(
        config, "v1", date(2026, 8, 20), "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    assert manifest.strategy_version == "v1"
    assert manifest.t0 == date(2026, 8, 20)
    assert manifest.hold_days == config.backtest.hold_days
    assert manifest.target_direction == "LONG"
    assert manifest.target_signal_name == "long_oversold_rebound"
    assert manifest.universe_segments == config.universe.segments
    assert manifest.initial_capital == 10_000_000.0


def test_save_and_load_manifest_roundtrip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = build_manifest(
        config, "v1", date(2026, 8, 20), "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    raw = load_manifest_raw(path)
    assert raw["strategy_version"] == "v1"
    assert raw["t0"] == "2026-08-20"
    assert raw["hashes"]["features_hash"] == manifest.hashes.features_hash


def test_verify_strategy_hashes_unchanged_true_for_freshly_saved_manifest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = build_manifest(
        config, "v1", date(2026, 8, 20), "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    unchanged, mismatches = verify_strategy_hashes_unchanged(load_manifest_raw(path))
    assert unchanged is True
    assert mismatches == []


def test_verify_strategy_hashes_unchanged_detects_tampered_hash(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = build_manifest(
        config, "v1", date(2026, 8, 20), "LONG", "long_oversold_rebound",
        initial_capital=10_000_000.0, per_trade_notional_fraction=0.01,
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    raw = load_manifest_raw(path)
    raw["hashes"]["features_hash"] = "0" * 64  # simulate a code change
    unchanged, mismatches = verify_strategy_hashes_unchanged(raw)
    assert unchanged is False
    assert mismatches == ["features_hash"]
