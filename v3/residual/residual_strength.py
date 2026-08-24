"""Spec section 19: residual_strength = residual_Q5Q1 / original_Q5Q1,
per (Target definition, Horizon) pair - how much of the Raw Target's
Q5-Q1 spread survives once market components are removed. NA (not 0 or
an arbitrary large number) whenever the denominator is too close to zero
for the ratio to be meaningful - a fabricated ratio would misrepresent a
case where the ORIGINAL edge itself was negligible.
"""

from __future__ import annotations

MIN_DENOMINATOR = 1e-4  # 0.01 percentage points - below this, "original" spread is itself noise


def residual_strength(residual_q5_q1: float | None, original_q5_q1: float | None) -> float | None:
    if residual_q5_q1 is None or original_q5_q1 is None:
        return None
    if abs(original_q5_q1) < MIN_DENOMINATOR:
        return None
    return residual_q5_q1 / original_q5_q1
