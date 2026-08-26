"""Implausible-value exclusion for Model predictions' `actual` column
(spec section 5's explicit authorization: "明らかなデータ破損について
は既存のData Integrityルールに従って機械的に処理する。処理内容はログ
に残す。"). Reuses `v2.stats.exclude_implausible_returns()`/
`MAX_PLAUSIBLE_FORWARD_RETURN` (V1/V2, unmodified - the SAME bound
Phase V2-1 already established for exactly this class of issue, chosen
long before this Phase ran, not tuned to today's specific outlier) -
never a new, bespoke threshold invented after seeing this Phase's data.

Discovered empirically during this Phase's own smoke test (see
research/phase_v3_3_report.md "Bugs discovered"): `target_risk_adjusted_5d`
(Raw Return / |MAE|) can take extreme values (observed up to ~395) when
its MAE denominator is near zero - a mathematically-expected property of
a ratio target with a small denominator, NOT a Target Registry defect
(v3/targets/compute.py is untouched), but one that overflows a downstream
cumulative-return calculation (`v3/validation/topn_portfolio.py`'s
`np.cumprod`) if left unfiltered. This module is the fix: a Full Universe
DATA INTEGRITY step applied at the validation layer, identical in kind to
V1/V2's own existing outlier guard, never a Target/Feature/Model
redefinition.
"""

from __future__ import annotations

import logging

import pandas as pd

from v2.stats import MAX_PLAUSIBLE_FORWARD_RETURN, exclude_implausible_returns

logger = logging.getLogger(__name__)


def clean_predictions(
    predictions: pd.DataFrame, actual_col: str = "actual", label: str = ""
) -> pd.DataFrame:
    n_before = len(predictions)
    cleaned = exclude_implausible_returns(predictions, actual_col, MAX_PLAUSIBLE_FORWARD_RETURN)
    n_excluded = n_before - len(cleaned)
    if n_excluded > 0:
        logger.info(
            "V3-3 data integrity: excluded %d/%d implausible rows (|%s| > %.1f) %s",
            n_excluded, n_before, actual_col, MAX_PLAUSIBLE_FORWARD_RETURN, label,
        )
    return cleaned
