from __future__ import annotations

from v3.models.config import (
    BASELINE_PARAMS,
    PRIMARY_HORIZON,
    PRIMARY_TARGET_VARIANT,
    QUANTILES,
    RANDOM_SEED,
)


def test_baseline_params_are_conservative_fixed_values() -> None:
    assert BASELINE_PARAMS.n_estimators == 300
    assert BASELINE_PARAMS.learning_rate == 0.03
    assert BASELINE_PARAMS.random_state == RANDOM_SEED


def test_as_kwargs_covers_every_documented_hyperparameter() -> None:
    kwargs = BASELINE_PARAMS.as_kwargs()
    for key in (
        "n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples",
        "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "random_state",
    ):
        assert key in kwargs


def test_primary_target_is_5d_raw() -> None:
    assert PRIMARY_HORIZON == 5
    assert PRIMARY_TARGET_VARIANT == "raw"


def test_quantiles_are_010_050_090() -> None:
    assert QUANTILES == (0.1, 0.5, 0.9)
