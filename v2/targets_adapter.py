"""V2 Forward Target adapter (spec section 5).

A pure passthrough to targets.forward_returns.compute_forward_returns()
(unmodified V1 code) - all 7 of FORWARD_WINDOWS=(1,3,5,7,10,15,20) are
retained (spec's own "最低限...を保持する" wording), even though V2's
own ranking/reporting focuses on the 5/10/15/20 subset
(v2/config/v2_settings.yaml::forward_windows). Kept as its own module
(rather than calling targets.forward_returns directly from
v2/pipeline.py) purely so every V2->V1 reuse point has one obvious,
greppable adapter file, matching v2/features_adapter.py's pattern.
"""

from __future__ import annotations

import pandas as pd

from targets.forward_returns import FORWARD_WINDOWS, compute_forward_returns

__all__ = ["FORWARD_WINDOWS", "build_v2_forward_targets"]


def build_v2_forward_targets(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return compute_forward_returns(ohlcv)
