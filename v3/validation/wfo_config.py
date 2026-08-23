"""Phase V3-3 pre-registered configuration (spec section 6/7/19/20/26):
every value below is fixed BEFORE the real Full Universe run and never
adjusted after seeing any result - this module IS the pre-registration
record.

WFO window sizing (train_months=18/validation_months=1/oos_months=6/
step_months=6) is chosen from the ACTUAL available data span
(2022-01-04..2026-08-20, ~55 months, established in Phase V3-1/V3-2) to
produce several (6) large, non-overlapping, sequential OOS windows -
large enough for Full-Universe daily-panel training to be statistically
meaningful, few enough to keep total Model-fit compute tractable across
the (3 Models x 4 Horizons x 4 Variants) combinatorial space this Phase
must at least partially cover (see v3/validation/orchestrator.py's tiered
depth). validation_months=1 (~21 calendar days, approximating the 20
TRADING days spec section 6 names) is this Phase's EMBARGO - see this
package's own __init__.py docstring for why V1's "VALIDATION" segment is
reused for that role rather than adding a second window-generation
primitive.

Permutation n_permutations values reuse Phase V2-2's own already-
documented tractability finding (research/phase_v2_2_report.md section 26):
a Score/prediction-derived quantile bucket is a full 20% of a
multi-million-row population, so 10,000 permutations per call is not
tractable at this scale - fixed here at 1,000 (primary) / 300 (FDR
sweep) BEFORE this Phase's own run, not after observing a slowdown.
"""

from __future__ import annotations

from config.loader import (
    BlockBootstrapConfig,
    BootstrapConfig,
    DayClusterBootstrapConfig,
    MarketRegimeConfig,
    PermutationConfig,
    WalkForwardConfig,
)

DATA_START = "2022-01-04"
DATA_END = "2026-08-20"

WFO_CONFIG = WalkForwardConfig(
    train_months=18, validation_months=1, oos_months=6, step_months=6, min_oos_completeness=0.5
)

PRIMARY_HORIZON = 5
PRIMARY_TARGET_VARIANT = "raw"
PRIMARY_TARGET_COL = "target_raw_5d"
SECONDARY_HORIZON_TARGETS = ("target_raw_10d", "target_raw_15d", "target_raw_20d")
SECONDARY_VARIANT_TARGETS = (
    "target_topix_relative_5d", "target_vol_adjusted_5d", "target_risk_adjusted_5d",
)

TOP_N_VALUES = (5, 10, 20)

TRADE_LEVEL_BOOTSTRAP_CONFIG = BootstrapConfig(n_resamples=10_000, seed=342, confidence_level=0.95)
DAY_CLUSTER_BOOTSTRAP_CONFIG = DayClusterBootstrapConfig(
    n_resamples=10_000, seed=344, confidence_level=0.95
)
BLOCK_BOOTSTRAP_CONFIG = BlockBootstrapConfig(
    block_length_days=5, n_resamples=10_000, seed=345, confidence_level=0.95
)
PERMUTATION_CONFIG = PermutationConfig(n_permutations=1_000, seed=343, forward_window=5)
FDR_SWEEP_PERMUTATION_CONFIG = PermutationConfig(n_permutations=300, seed=346, forward_window=5)
MARKET_REGIME_CONFIG = MarketRegimeConfig()

SIGNIFICANCE_ALPHA = 0.05
MIN_WINDOW_DIRECTION_AGREEMENT = 0.6  # >= 60% of OOS windows must agree in direction
