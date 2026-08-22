from __future__ import annotations

import numpy as np
import pytest
from conftest import make_panel

from config.loader import ScoringConfig
from scoring.scorer import (
    SUBSCORE_COLUMNS,
    compute_momentum_score,
    compute_relative_score,
    compute_risk_score,
    compute_setup_score,
    compute_total_score,
    compute_trend_score,
    compute_volume_score,
)
from signals.base import Direction


def _all_true_long_panel() -> dict:
    """A single-row panel where every LONG condition across every
    category is satisfied - used as the "should score the max" fixture.
    """
    return dict(
        close=[110.0],
        sma_20=[105.0], sma_50=[95.0], sma_75=[90.0],
        sma_20_slope=[0.01], sma_50_slope=[0.01],
        return_1d=[0.02], return_5d=[0.10], return_10d=[0.10],
        rsi_14=[60.0], macd_hist=[1.0],
        volume_ratio_20d=[3.0], volume_ratio_5d=[2.0], volume_trend=[0.05],
        rs_5d=[0.10], rs_20d=[0.10], rs_60d=[0.10],
        atr_pct=[0.01], volatility_20d=[0.005],
    )


def _all_true_short_panel() -> dict:
    return dict(
        close=[90.0],
        sma_20=[95.0], sma_50=[105.0], sma_75=[110.0],
        sma_20_slope=[-0.01], sma_50_slope=[-0.01],
        return_1d=[-0.02], return_5d=[-0.10], return_10d=[-0.10],
        rsi_14=[40.0], macd_hist=[-1.0],
        volume_ratio_20d=[3.0], volume_ratio_5d=[2.0], volume_trend=[0.05],
        rs_5d=[-0.10], rs_20d=[-0.10], rs_60d=[-0.10],
        atr_pct=[0.01], volatility_20d=[0.005],
    )


# --- Category ranges ---------------------------------------------------------


def test_trend_score_max_when_all_long_conditions_met() -> None:
    config = ScoringConfig()
    panel = make_panel(**_all_true_long_panel())
    score = compute_trend_score(panel, Direction.LONG, config.trend)
    assert score.iloc[0] == 20


def test_trend_score_zero_when_no_conditions_met() -> None:
    config = ScoringConfig()
    panel = make_panel(
        close=[90.0], sma_20=[95.0], sma_50=[105.0],
        sma_20_slope=[-0.01], sma_50_slope=[-0.01],
    )
    score = compute_trend_score(panel, Direction.LONG, config.trend)
    assert score.iloc[0] == 0


def test_trend_score_is_not_a_sign_flip_between_directions() -> None:
    """A pure uptrend panel should score well for LONG and poorly for
    SHORT - not "whatever LONG didn't get".
    """
    config = ScoringConfig()
    panel = make_panel(**_all_true_long_panel())
    long_score = compute_trend_score(panel, Direction.LONG, config.trend)
    short_score = compute_trend_score(panel, Direction.SHORT, config.trend)
    assert long_score.iloc[0] == 20
    assert short_score.iloc[0] == 0


def test_momentum_score_range() -> None:
    config = ScoringConfig()
    panel_long = make_panel(**_all_true_long_panel())
    panel_short = make_panel(**_all_true_short_panel())
    assert compute_momentum_score(panel_long, Direction.LONG, config.momentum).iloc[0] == 20
    assert compute_momentum_score(panel_short, Direction.SHORT, config.momentum).iloc[0] == 20
    assert compute_momentum_score(panel_long, Direction.SHORT, config.momentum).iloc[0] == 0


def test_volume_score_range() -> None:
    config = ScoringConfig()
    panel_long = make_panel(**_all_true_long_panel())
    score = compute_volume_score(panel_long, Direction.LONG, config.volume)
    assert score.iloc[0] == 15


def test_volume_score_direction_specific_condition() -> None:
    """The price-move+volume confirmation leg differs by direction even
    though the volume-only legs (ratio, trend) don't.
    """
    config = ScoringConfig()
    panel = make_panel(**_all_true_long_panel())  # return_1d=+0.02 (an UP day)
    long_score = compute_volume_score(panel, Direction.LONG, config.volume)
    short_score = compute_volume_score(panel, Direction.SHORT, config.volume)
    assert long_score.iloc[0] == 15
    assert short_score.iloc[0] == 10  # loses the direction-specific 5pts


def test_risk_score_is_direction_agnostic() -> None:
    config = ScoringConfig()
    panel = make_panel(atr_pct=[0.01], volatility_20d=[0.005])
    long_score = compute_risk_score(panel, config.risk)
    assert long_score.iloc[0] == 10
    # compute_risk_score takes no direction argument at all - agnostic by construction.


def test_setup_score_range() -> None:
    config = ScoringConfig()
    panel_long = make_panel(**_all_true_long_panel())
    score = compute_setup_score(panel_long, Direction.LONG, config.setup)
    assert score.iloc[0] == 20


def test_setup_score_uses_stricter_bar_than_a_signal_trigger() -> None:
    """return_5d=0.04 clears long_momentum_continuation's own 0.03
    trigger bar but NOT Setup's stricter 0.05 bar - proving Setup isn't
    just re-reading the trigger condition.
    """
    config = ScoringConfig()
    panel = make_panel(
        close=[100.0], return_5d=[0.04], return_1d=[0.0],
        volume_ratio_20d=[0.5], sma_20=[100.0], sma_50=[90.0], sma_75=[80.0],
        atr_pct=[0.05],
    )
    score = compute_setup_score(panel, Direction.LONG, config.setup)
    assert score.iloc[0] < 20  # the return_5d leg alone should not have scored


# --- Relative Score mapping (section 9: not a sign flip) --------------------


@pytest.mark.parametrize(
    ("rs", "expected_points"),
    [
        (0.10, 5), (0.05, 5), (0.03, 4), (0.02, 4), (0.01, 3),
        (0.0, 3), (-0.01, 2), (-0.05, 1), (-0.10, 0),
    ],
)
def test_relative_score_long_mapping(rs: float, expected_points: float) -> None:
    config = ScoringConfig()
    panel = make_panel(rs_5d=[rs], rs_20d=[-999.0], rs_60d=[-999.0])  # isolate rs_5d's contribution
    score = compute_relative_score(panel, Direction.LONG, config.relative)
    assert score.iloc[0] == expected_points


@pytest.mark.parametrize(
    ("rs", "expected_points"),
    [(-0.10, 5), (-0.05, 5), (-0.03, 4), (0.0, 3), (0.05, 1), (0.10, 0)],
)
def test_relative_score_short_mapping_is_not_a_sign_flip_of_long(
    rs: float, expected_points: float
) -> None:
    """SHORT scores are computed by mapping -rs through the SAME ladder
    LONG uses - verified here against hand-computed expectations,
    distinct from simply negating the LONG score.
    """
    config = ScoringConfig()
    # +999 (not -999): SHORT negates the value before mapping, so a
    # "neutral/isolating" placeholder for SHORT must be a large POSITIVE
    # number (so -placeholder falls below every threshold -> 0pts) -
    # the mirror image of LONG's -999 placeholder.
    panel = make_panel(rs_5d=[rs], rs_20d=[999.0], rs_60d=[999.0])
    score = compute_relative_score(panel, Direction.SHORT, config.relative)
    assert score.iloc[0] == expected_points


def test_relative_score_nan_gives_zero_not_nan() -> None:
    config = ScoringConfig()
    panel = make_panel(rs_5d=[np.nan], rs_20d=[np.nan], rs_60d=[np.nan])
    score = compute_relative_score(panel, Direction.LONG, config.relative)
    assert score.iloc[0] == 0
    assert not np.isnan(score.iloc[0])


def test_relative_score_max_is_fifteen() -> None:
    config = ScoringConfig()
    panel = make_panel(rs_5d=[1.0], rs_20d=[1.0], rs_60d=[1.0])
    score = compute_relative_score(panel, Direction.LONG, config.relative)
    assert score.iloc[0] == 15


# --- Total Score --------------------------------------------------------------


def test_total_score_equals_sum_of_subscores() -> None:
    config = ScoringConfig()
    panel = make_panel(**_all_true_long_panel())
    out = compute_total_score(panel, Direction.LONG, config)
    manual_sum = out[SUBSCORE_COLUMNS].sum(axis=1)
    assert (out["total_score"] == manual_sum).all()


def test_total_score_is_bounded_0_to_100() -> None:
    config = ScoringConfig()
    panel_max = make_panel(**_all_true_long_panel())
    panel_min = make_panel(
        close=[90.0], sma_20=[95.0], sma_50=[105.0], sma_75=[110.0],
        sma_20_slope=[-0.01], sma_50_slope=[-0.01],
        return_1d=[-0.02], return_5d=[-0.10], return_10d=[-0.10],
        rsi_14=[40.0], macd_hist=[-1.0],
        volume_ratio_20d=[0.1], volume_ratio_5d=[0.1], volume_trend=[-0.05],
        rs_5d=[-0.10], rs_20d=[-0.10], rs_60d=[-0.10],
        atr_pct=[0.10], volatility_20d=[0.10],
    )
    out_max = compute_total_score(panel_max, Direction.LONG, config)
    out_min = compute_total_score(panel_min, Direction.LONG, config)
    assert out_max["total_score"].iloc[0] == 100
    assert out_min["total_score"].iloc[0] == 0
    assert 0 <= out_max["total_score"].iloc[0] <= 100
    assert 0 <= out_min["total_score"].iloc[0] <= 100


def test_total_score_columns_match_declared_weights() -> None:
    config = ScoringConfig()
    assert config.weights.trend + config.weights.momentum + config.weights.volume \
        + config.weights.relative + config.weights.setup + config.weights.risk == 100


def test_missing_required_column_raises() -> None:
    config = ScoringConfig()
    panel = make_panel(close=[100.0])  # missing sma_20 etc.
    with pytest.raises(KeyError):
        compute_trend_score(panel, Direction.LONG, config.trend)
