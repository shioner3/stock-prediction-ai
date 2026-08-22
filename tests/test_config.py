from __future__ import annotations

from pathlib import Path

from config.loader import load_app_config, load_phase9_config, load_universe_filters


def test_load_app_config_reads_settings_yaml() -> None:
    config = load_app_config(Path("config/settings.yaml"))
    assert config.data.provider == "yfinance"
    assert config.universe.segments == ["Prime", "Standard", "Growth"]
    assert config.data.market_index.ticker == "1306.T"
    assert config.data.market_index.name == "TOPIX Proxy"
    assert config.data.market_index.type == "ETF_PROXY"


def test_load_universe_filters_reads_filters_yaml() -> None:
    filters = load_universe_filters(Path("config/universe_filters.yaml"))
    assert filters.price.min_close_price == 100
    assert filters.liquidity.min_avg_volume_20d == 10_000


def test_load_phase9_config_reads_phase9_settings_yaml() -> None:
    config = load_phase9_config(Path("config/phase9_settings.yaml"))
    assert config.day_cluster_bootstrap.n_resamples == 10_000
    assert config.day_cluster_bootstrap.seed == 44
    assert config.block_bootstrap.block_length_days == 5
    assert config.block_bootstrap.seed == 45
    assert config.timing_placebo.offsets == [-15, -10, -5, -3, -1, 5, 10]
    assert config.winsorization.lower_percentile == 0.01
    assert config.winsorization.upper_percentile == 0.99


def test_load_phase9_config_defaults_when_file_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    config = load_phase9_config(empty)
    assert config.block_bootstrap.block_length_days == 5
