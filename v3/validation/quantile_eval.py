"""Model C (Quantile Regression) evaluation (spec section 12): does the
predicted interval width relate to actual return dispersion? Genuinely
new - neither V1 nor V2 has a quantile model to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileCalibrationResult:
    n: int
    monotonic_fraction: float  # fraction of rows where q0.1 <= q0.5 <= q0.9
    coverage_below_q10: float  # fraction of actual values < predicted q0.1 (target ~0.10)
    coverage_above_q90: float  # fraction of actual values > predicted q0.9 (target ~0.10)
    coverage_within_interval: float  # fraction of actual values in [q0.1, q0.9] (target ~0.80)
    upside_width_vs_actual_corr: float | None  # corr(q0.9-q0.5, actual) among actual > 0 rows
    downside_width_vs_actual_corr: float | None  # corr(q0.5-q0.1, -actual) among actual < 0 rows


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_quantile_calibration(predictions: pd.DataFrame) -> QuantileCalibrationResult:
    """predictions: date/ticker/actual/q0.1/prediction(=q0.5)/q0.9 (see
    v3/validation/train_predict.py::run_model_c_on_window()'s output
    shape).
    """
    valid = predictions.dropna(subset=["q0.1", "prediction", "q0.9", "actual"]).copy()
    q10, q50, q90, actual = (
        valid["q0.1"].to_numpy(), valid["prediction"].to_numpy(), valid["q0.9"].to_numpy(),
        valid["actual"].to_numpy(),
    )
    n = len(valid)
    monotonic = ((q10 <= q50 + 1e-12) & (q50 <= q90 + 1e-12)).mean() if n else 0.0

    below_q10 = float((actual < q10).mean()) if n else 0.0
    above_q90 = float((actual > q90).mean()) if n else 0.0
    within = float(((actual >= q10) & (actual <= q90)).mean()) if n else 0.0

    upside_mask = actual > 0
    upside_width = q90[upside_mask] - q50[upside_mask]
    upside_corr = _pearson(upside_width, actual[upside_mask])

    downside_mask = actual < 0
    downside_width = q50[downside_mask] - q10[downside_mask]
    downside_corr = _pearson(downside_width, -actual[downside_mask])

    return QuantileCalibrationResult(
        n=n, monotonic_fraction=float(monotonic), coverage_below_q10=below_q10,
        coverage_above_q90=above_q90, coverage_within_interval=within,
        upside_width_vs_actual_corr=upside_corr, downside_width_vs_actual_corr=downside_corr,
    )
