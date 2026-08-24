"""v3/residual/targets.py - Beta-adjusted Residual / Sector-relative
Target construction, verified against hand-computed values for all 4
Horizons, plus the implausibility-masking behavior learned from Phase
V3-4's own 3 bugs (applied proactively here, not discovered via a bug).
"""

from __future__ import annotations

import pandas as pd

from v3.residual.targets import (
    RESIDUAL_TARGET_COLUMNS,
    VARIANT_BETA_RESIDUAL,
    VARIANT_SECTOR_RELATIVE,
    compute_residual_targets,
    residual_target_column_name,
)
from v3.targets.registry import HORIZONS, target_column_name


def _tiny_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2023-01-02"] * 4)
    tickers = ["A", "B", "C", "D"]
    data = {"date": dates, "ticker": tickers}
    # raw_h = topix_relative_h + market_forward_h; market_forward_h = 0.01*h/5
    # for every horizon, so it's easy to hand-verify.
    for h in HORIZONS:
        market_forward = 0.01 * h / 5
        raw = [0.05, 0.03, -0.01, 0.02]
        data[target_column_name("raw", h)] = raw
        data[target_column_name("topix_relative", h)] = [r - market_forward for r in raw]
    dataset = pd.DataFrame(data)

    beta_panel = pd.DataFrame({"date": dates, "ticker": tickers, "beta": [1.0, 2.0, 0.5, 1.0]})
    sector_map = pd.DataFrame({"ticker": tickers, "sector33": ["Tech", "Tech", "Bank", "Bank"]})
    return dataset, beta_panel, sector_map


def test_residual_target_columns_exist_for_all_horizons() -> None:
    dataset, beta_panel, sector_map = _tiny_dataset()
    out = compute_residual_targets(dataset, beta_panel, sector_map)
    for h in HORIZONS:
        assert residual_target_column_name(VARIANT_BETA_RESIDUAL, h) in out.columns
        assert residual_target_column_name(VARIANT_SECTOR_RELATIVE, h) in out.columns
    assert set(RESIDUAL_TARGET_COLUMNS) == {
        residual_target_column_name(v, h)
        for v in (VARIANT_BETA_RESIDUAL, VARIANT_SECTOR_RELATIVE) for h in HORIZONS
    }


def test_beta_residual_formula_all_horizons() -> None:
    dataset, beta_panel, sector_map = _tiny_dataset()
    out = compute_residual_targets(dataset, beta_panel, sector_map)
    raw = [0.05, 0.03, -0.01, 0.02]
    betas = [1.0, 2.0, 0.5, 1.0]
    for h in HORIZONS:
        market_forward = 0.01 * h / 5
        expected = [r - b * market_forward for r, b in zip(raw, betas, strict=True)]
        actual = out[residual_target_column_name(VARIANT_BETA_RESIDUAL, h)].tolist()
        for e, a in zip(expected, actual, strict=True):
            assert abs(e - a) < 1e-9, (h, expected, actual)


def test_sector_relative_formula_all_horizons() -> None:
    dataset, beta_panel, sector_map = _tiny_dataset()
    out = compute_residual_targets(dataset, beta_panel, sector_map)
    raw = [0.05, 0.03, -0.01, 0.02]
    tech_mean = (raw[0] + raw[1]) / 2
    bank_mean = (raw[2] + raw[3]) / 2
    expected_means = [tech_mean, tech_mean, bank_mean, bank_mean]
    for h in HORIZONS:
        expected = [r - m for r, m in zip(raw, expected_means, strict=True)]
        actual = out[residual_target_column_name(VARIANT_SECTOR_RELATIVE, h)].tolist()
        for e, a in zip(expected, actual, strict=True):
            assert abs(e - a) < 1e-9, (h, expected, actual)


def test_extreme_beta_masked_before_residual_computation() -> None:
    dataset, beta_panel, sector_map = _tiny_dataset()
    beta_panel = beta_panel.copy()
    beta_panel.loc[beta_panel["ticker"] == "A", "beta"] = 1_000_000.0  # implausible
    out = compute_residual_targets(dataset, beta_panel, sector_map)
    row_a = out[out["ticker"] == "A"].iloc[0]
    assert pd.isna(row_a[residual_target_column_name(VARIANT_BETA_RESIDUAL, 5)])


def test_implausible_full_universe_row_does_not_poison_sector_mean() -> None:
    dataset, beta_panel, sector_map = _tiny_dataset()
    poisoned = dataset.copy()
    for h in HORIZONS:
        poisoned.loc[poisoned["ticker"] == "A", target_column_name("raw", h)] = 20.0
    out_poisoned = compute_residual_targets(poisoned, beta_panel, sector_map)
    row_b = out_poisoned[out_poisoned["ticker"] == "B"].iloc[0]
    # B is A's sector-mate (Tech) - A's implausible raw value must be
    # excluded from the sector mean before it can pollute B's own
    # sector-relative target.
    expected_sector_relative_b = 0.03 - 0.03  # only B's own value survives the mean
    actual_b = row_b[residual_target_column_name(VARIANT_SECTOR_RELATIVE, 5)]
    assert abs(actual_b - expected_sector_relative_b) < 1e-9
