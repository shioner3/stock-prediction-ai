"""Frozen numeric values carried over from Phase V3-3's own report
(`research/phase_v3_3_report.md`, sections 16/22/23) - the 3 rule-based
Baseline spreads (Random/Momentum/V2 Score), the primary target's FDR
significance flag, and the Primary Q5-Q1 spread itself. These are NOT
re-derived by Phase V3-4: rebuilding the 3 Baselines/FDR flag would
require re-running Model B/C and the full 16-test FDR sweep, none of
which this Phase's decomposition work touches, and `reproduce.py`'s
narrow hash verification (config_hash/feature_hash/dataset_hash all
bit-identical to V3-3) already PROVES these values are unchanged - V3-3's
own pipeline is fully deterministic (empirically re-confirmed across two
independent V3-3 Full Universe runs). Reusing the already-published,
already-committed number is more honest than re-deriving a value that is
mathematically guaranteed to come out identical.

`PRIMARY_Q5_Q1_SPREAD_REFERENCE` plays a different role: it is the
EMPIRICAL substitute for the "model_hash matches" check spec section 1
asks for, which V3-3's own saved report never actually persisted a
`model_hash` field to compare against (see `reproduce.py`'s
`HashVerification` docstring). If Phase V3-4's regenerated Primary
predictions come from the SAME deterministic training code/data as V3-3,
their Q5-Q1 spread MUST reproduce this value (within floating-point
tolerance) - `check_primary_spread_reproduction()` verifies this and
should STOP the Phase if it does not.
"""

from __future__ import annotations

RANDOM_BASELINE_Q5_Q1_SPREAD = 0.0000486
MOMENTUM_BASELINE_Q5_Q1_SPREAD = -0.00651
V2_SCORE_BASELINE_Q5_Q1_SPREAD = -0.00056
PRIMARY_FDR_SIGNIFICANT = True  # horizon:target_raw_5d:Q5 raw_p=0.0000 adj_p=0.0000
PRIMARY_Q5_Q1_SPREAD_REFERENCE = 0.00383  # research/phase_v3_3_report.md section 12
PRIMARY_SPREAD_REPRODUCTION_TOLERANCE = 0.0001  # 0.01 percentage points


def check_primary_spread_reproduction(reproduced_spread: float | None) -> bool:
    if reproduced_spread is None:
        return False
    diff = abs(reproduced_spread - PRIMARY_Q5_Q1_SPREAD_REFERENCE)
    return diff <= PRIMARY_SPREAD_REPRODUCTION_TOLERANCE
