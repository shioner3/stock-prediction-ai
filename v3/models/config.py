"""Phase V3-2 Baseline hyperparameters (spec section 8): fixed BEFORE any
training run in this Phase and never adjusted afterward, regardless of
how the resulting metrics look. These are explicitly a conservative
starting point, not a tuned/optimal configuration - spec section 8's own
"この値は「最適値」ではない". Hyperparameter search belongs to a future
Phase, against pre-registered OOS criteria, never against these same
values chased after seeing a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RANDOM_SEED = 42

# spec section 14: Primary Baseline target - 5d horizon, Raw variant.
# The other 3 horizons (10/15/20d) and 3 variants (TOPIX-relative/
# Vol-adjusted/Risk-adjusted) are only checked for "can the pipeline
# handle them" (spec section 14), never compared to pick a winner here.
PRIMARY_HORIZON = 5
PRIMARY_TARGET_VARIANT = "raw"

# Model C (spec section 7): 3 quantiles - lower/median/upper.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class LightGBMBaselineParams:
    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = RANDOM_SEED

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "random_state": self.random_state,
        }


BASELINE_PARAMS = LightGBMBaselineParams()
