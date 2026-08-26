from __future__ import annotations

from v3.validation.wfo_config import WFO_CONFIG
from v3.validation.windows import get_v3_3_windows


def test_generates_multiple_non_overlapping_oos_windows() -> None:
    windows = get_v3_3_windows()
    assert len(windows) >= 3
    for i in range(1, len(windows)):
        assert windows[i].oos_start >= windows[i - 1].oos_end


def test_every_window_has_a_real_embargo_gap() -> None:
    windows = get_v3_3_windows()
    for w in windows:
        assert w.train_end == w.validation_start
        assert w.validation_end == w.oos_start
        assert w.validation_end > w.validation_start


def test_windows_are_deterministic() -> None:
    assert get_v3_3_windows() == get_v3_3_windows()


def test_wfo_config_matches_pre_registered_values() -> None:
    assert WFO_CONFIG.train_months == 18
    assert WFO_CONFIG.validation_months == 1
    assert WFO_CONFIG.oos_months == 6
    assert WFO_CONFIG.step_months == 6
