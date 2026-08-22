from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from config.loader import SignalsConfig
from features.pipeline import compute_feature_panel
from signals.pipeline import RECORD_COLUMNS, compute_signal_panel, compute_signal_records
from signals.registry import SIGNAL_REGISTRY, all_signal_meta


def _full_panel(n: int = 400, seed: int = 1) -> pd.DataFrame:
    ohlcv = make_synthetic_ohlcv(n, seed=seed)
    return compute_feature_panel(ohlcv)


def test_registry_has_twelve_signals() -> None:
    assert len(SIGNAL_REGISTRY) == 12


def test_registry_has_six_long_and_six_short() -> None:
    config = SignalsConfig()
    metas = all_signal_meta(config)
    long_count = sum(1 for m in metas if m.direction.value == "LONG")
    short_count = sum(1 for m in metas if m.direction.value == "SHORT")
    assert long_count == 6
    assert short_count == 6


def test_signal_names_are_unique() -> None:
    config = SignalsConfig()
    names = [m.name for m in all_signal_meta(config)]
    assert len(names) == len(set(names))


def test_compute_signal_panel_produces_boolean_columns() -> None:
    panel = _full_panel()
    config = SignalsConfig()
    result = compute_signal_panel(panel, config)

    assert result["ticker"].equals(panel["ticker"])
    assert result["date"].equals(panel["date"])
    signal_cols = [c for c in result.columns if c not in ("ticker", "date")]
    assert len(signal_cols) == 12
    for col in signal_cols:
        assert result[col].dtype == bool


def test_compute_signal_records_is_long_format_triggered_only() -> None:
    panel = _full_panel()
    config = SignalsConfig()
    records = compute_signal_records(panel, config)

    assert list(records.columns) == RECORD_COLUMNS
    assert records["triggered"].all()
    assert set(records["direction"]) <= {"LONG", "SHORT"}


def test_compute_signal_records_matches_wide_panel_trigger_count() -> None:
    panel = _full_panel()
    config = SignalsConfig()
    wide = compute_signal_panel(panel, config)
    records = compute_signal_records(panel, config)

    signal_cols = [c for c in wide.columns if c not in ("ticker", "date")]
    expected_total = int(wide[signal_cols].sum().sum())
    assert len(records) == expected_total


# --- Signal Independence (section "22. Signal Independence") ---------------


def test_changing_one_signal_config_does_not_affect_another() -> None:
    panel = _full_panel()

    config_a = SignalsConfig()
    config_b = SignalsConfig()
    config_b.long.breakout.volume_multiple = 100.0  # effectively disables long_breakout

    records_a = compute_signal_records(panel, config_a)
    records_b = compute_signal_records(panel, config_b)

    pullback_a = records_a[records_a["signal_name"] == "long_pullback"]
    pullback_b = records_b[records_b["signal_name"] == "long_pullback"]
    pd.testing.assert_frame_equal(
        pullback_a.reset_index(drop=True), pullback_b.reset_index(drop=True)
    )

    breakout_a = records_a[records_a["signal_name"] == "long_breakout"]
    breakout_b = records_b[records_b["signal_name"] == "long_breakout"]
    # With volume_multiple=100, long_breakout should trigger less often
    # (or equally often, never more).
    assert len(breakout_b) <= len(breakout_a)


def test_each_signal_module_only_reads_its_own_required_columns() -> None:
    """A weaker, structural companion to the independence test above:
    every Signal's required_columns is a fixed, declared set - two
    Signals never secretly share mutable state (there is none - every
    compute_signal() is a pure function of (panel, config)).
    """
    config = SignalsConfig()
    metas = all_signal_meta(config)
    for m in metas:
        assert len(m.required_columns) > 0
        assert "signal_name" not in m.required_columns  # never reads another Signal's output


# --- No cross-contamination between repeated calls --------------------------


def test_compute_signal_panel_is_deterministic() -> None:
    panel = _full_panel()
    config = SignalsConfig()
    first = compute_signal_panel(panel, config)
    second = compute_signal_panel(panel, config)
    pd.testing.assert_frame_equal(first, second)


def test_empty_signal_records_when_nothing_triggers() -> None:
    n = 10  # far too short for any signal's warmup to clear
    panel = _full_panel(n=n)
    config = SignalsConfig()
    records = compute_signal_records(panel, config)
    assert records.empty
    assert list(records.columns) == RECORD_COLUMNS


def test_signal_panel_has_no_nan_triggered_values() -> None:
    """Warmup rows have NaN Feature inputs; the boolean output must never
    itself be NaN (that would break Parquet's bool dtype / .all() checks
    downstream) - compute_signal() always resolves NaN inputs to False.
    """
    panel = _full_panel(n=10)
    config = SignalsConfig()
    wide = compute_signal_panel(panel, config)
    signal_cols = [c for c in wide.columns if c not in ("ticker", "date")]
    for col in signal_cols:
        assert not wide[col].isna().any()
        assert np.isin(wide[col].unique(), [True, False]).all()
