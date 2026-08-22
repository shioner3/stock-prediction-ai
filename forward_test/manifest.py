"""Phase 10 section 2/21: Frozen Strategy manifest.

Hashes the EXACT code this Forward Test is locked to (Feature/Signal/
Score computation, Backtest Entry/Exit, Market Regime classification),
plus the existing config_hash mechanism (config/settings.yaml +
config/universe_filters.yaml, unchanged from Phase 6.5-9). If any of
these hashes ever changes mid-Forward-Test, spec section 2 requires the
run to be treated as INVALID for the old Strategy Version - see
verify_strategy_hashes_unchanged() and section 11's "Strategy Version 1 /
Strategy Version 2" rule (a code fix starts a NEW version; it never
silently continues under the old T0).

Deliberately hashes the WHOLE features/, signals/, and scoring/
directories (not just long_oversold_rebound.py) - Score depends on every
feature category, and signals/pipeline.py + signals/registry.py are part
of the same execution path even though only long_oversold_rebound is
traded here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from common.hashing import hash_files
from config.loader import AppConfig

FROZEN_CODE_DIRS = ("features", "signals", "scoring")
FROZEN_BACKTEST_ENGINE_FILE = Path("backtest/engine.py")
FROZEN_MARKET_REGIME_FILE = Path("backtest/market_regime.py")
FROZEN_CONFIG_FILES = [Path("config/settings.yaml"), Path("config/universe_filters.yaml")]


def _all_py_files(dirname: str) -> list[Path]:
    return sorted(Path(dirname).rglob("*.py"))


@dataclass(frozen=True)
class StrategyHashes:
    features_hash: str
    signals_hash: str
    scoring_hash: str
    backtest_engine_hash: str
    market_regime_hash: str
    config_hash: str


def compute_strategy_hashes() -> StrategyHashes:
    return StrategyHashes(
        features_hash=hash_files(_all_py_files("features")),
        signals_hash=hash_files(_all_py_files("signals")),
        scoring_hash=hash_files(_all_py_files("scoring")),
        backtest_engine_hash=hash_files([FROZEN_BACKTEST_ENGINE_FILE]),
        market_regime_hash=hash_files([FROZEN_MARKET_REGIME_FILE]),
        config_hash=hash_files(FROZEN_CONFIG_FILES),
    )


@dataclass
class ForwardTestManifest:
    strategy_version: str
    t0: date
    hashes: StrategyHashes
    hold_days: int
    transaction_cost_bps: float
    market_regime_lookback_days: int
    market_regime_bull_threshold: float
    market_regime_bear_threshold: float
    target_direction: str
    target_signal_name: str
    universe_segments: list[str]
    data_source: str
    initial_capital: float
    per_trade_notional_fraction: float
    created_at: str


def build_manifest(
    config: AppConfig,
    strategy_version: str,
    t0: date,
    target_direction: str,
    target_signal_name: str,
    initial_capital: float,
    per_trade_notional_fraction: float,
) -> ForwardTestManifest:
    base_tier = next(t for t in config.validation.transaction_cost.tiers if t.name == "base")
    return ForwardTestManifest(
        strategy_version=strategy_version,
        t0=t0,
        hashes=compute_strategy_hashes(),
        hold_days=config.backtest.hold_days,
        transaction_cost_bps=base_tier.round_trip_bps,
        market_regime_lookback_days=config.validation.market_regime.lookback_days,
        market_regime_bull_threshold=config.validation.market_regime.bull_threshold,
        market_regime_bear_threshold=config.validation.market_regime.bear_threshold,
        target_direction=target_direction,
        target_signal_name=target_signal_name,
        universe_segments=list(config.universe.segments),
        data_source=config.data.provider,
        initial_capital=initial_capital,
        per_trade_notional_fraction=per_trade_notional_fraction,
        created_at=datetime.now().isoformat(),
    )


def _json_default(obj: object) -> object:
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_manifest(manifest: ForwardTestManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), default=_json_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_strategy_hashes_unchanged(saved_manifest: dict) -> tuple[bool, list[str]]:
    """Recomputes hashes NOW and compares against what a previously-saved
    manifest recorded. Returns (unchanged, mismatched_field_names).
    """
    current = compute_strategy_hashes()
    saved_hashes = saved_manifest["hashes"]
    mismatches = [
        field_name
        for field_name in (
            "features_hash", "signals_hash", "scoring_hash",
            "backtest_engine_hash", "market_regime_hash", "config_hash",
        )
        if getattr(current, field_name) != saved_hashes[field_name]
    ]
    return len(mismatches) == 0, mismatches
