from __future__ import annotations

from datetime import date

from backtest.walk_forward import generate_windows
from config.loader import WalkForwardConfig


def test_generates_expected_number_of_windows_for_known_data_range() -> None:
    # Matches the real dataset used for Phase 6's actual OOS run
    # (2022-01-04 .. 2024-06-28) - see README's "OOS期間の固定".
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    assert len(windows) == 5


def test_windows_are_strictly_time_ordered() -> None:
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    for i in range(len(windows) - 1):
        assert windows[i].train_start < windows[i + 1].train_start
        assert windows[i].oos_start < windows[i + 1].oos_start
        assert windows[i].oos_end <= windows[i + 1].oos_start


def test_window_boundaries_are_internally_consistent() -> None:
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    for w in windows:
        assert w.train_start < w.train_end
        assert w.train_end == w.validation_start
        assert w.validation_start < w.validation_end
        assert w.validation_end == w.oos_start
        assert w.oos_start < w.oos_end


def test_first_window_train_start_equals_data_start() -> None:
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    assert windows[0].train_start == date(2022, 1, 4)


def test_first_window_matches_hand_computed_dates() -> None:
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    w0 = windows[0]
    assert w0.train_start == date(2022, 1, 4)
    assert w0.train_end == date(2023, 1, 4)
    assert w0.validation_start == date(2023, 1, 4)
    assert w0.validation_end == date(2023, 4, 4)
    assert w0.oos_start == date(2023, 4, 4)
    assert w0.oos_end == date(2023, 7, 4)
    assert w0.oos_truncated is False


def test_last_window_is_truncated_when_data_runs_out() -> None:
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    assert windows[-1].oos_truncated is True
    assert windows[-1].oos_end == date(2024, 6, 28)


def test_step_shifts_train_start_by_step_months() -> None:
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    assert windows[1].train_start == date(2022, 4, 4)
    assert windows[2].train_start == date(2022, 7, 4)


def test_insufficient_data_gives_zero_windows() -> None:
    # Only 6 months of data - not enough for even one 12mo TRAIN period.
    windows = generate_windows(date(2024, 1, 1), date(2024, 6, 30), WalkForwardConfig())
    assert windows == []


def test_severely_truncated_trailing_window_is_dropped() -> None:
    # Data ends only 2 weeks into what would be the OOS period -
    # completeness way below the default 0.5 threshold.
    config = WalkForwardConfig(train_months=12, validation_months=3, oos_months=3, step_months=3)
    windows = generate_windows(date(2022, 1, 4), date(2023, 4, 18), config)
    # Window 0's OOS starts 2023-04-04 and data ends 2023-04-18 - 14 of
    # ~91 days present (~15% completeness) - must be dropped.
    for w in windows:
        assert w.oos_start != date(2023, 4, 4)


def test_min_oos_completeness_zero_keeps_even_a_one_day_window() -> None:
    config = WalkForwardConfig(
        train_months=12, validation_months=3, oos_months=3, step_months=3,
        min_oos_completeness=0.0,
    )
    windows = generate_windows(date(2022, 1, 4), date(2023, 4, 5), config)
    assert len(windows) == 1
    assert windows[0].oos_truncated is True


def test_generation_is_deterministic() -> None:
    a = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    b = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    assert a == b


def test_no_overlap_between_consecutive_oos_periods_under_default_config() -> None:
    """step_months == oos_months by default, so OOS periods must be
    back-to-back with no gap and no overlap.
    """
    windows = generate_windows(date(2022, 1, 4), date(2024, 6, 28), WalkForwardConfig())
    for i in range(len(windows) - 1):
        assert windows[i].oos_end == windows[i + 1].oos_start
