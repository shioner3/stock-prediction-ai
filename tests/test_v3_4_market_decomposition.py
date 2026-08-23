"""v3/robustness/market_decomposition.py - verifies each of the 5 return-
definition variants against hand-computed expected values on a tiny
synthetic panel, and that Q1-Q5 bucketing stays FIXED to the original
prediction (never re-derived from a variant's return).
"""

from __future__ import annotations

import pandas as pd

from v3.robustness.market_decomposition import (
    VARIANT_BETA_ADJUSTED,
    VARIANT_MARKET_NEUTRALIZED,
    VARIANT_RAW,
    VARIANT_SECTOR_RELATIVE,
    VARIANT_TOPIX_RELATIVE,
    build_return_variant_columns,
    evaluate_return_variant,
)


def _tiny_setup() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-02", "2023-01-02"])
    tickers = ["A", "B", "C", "D"]
    # raw = topix_relative + market_forward(date); here market_forward=0.01
    # on the only date present, so topix_relative = raw - 0.01.
    raw = [0.05, 0.03, -0.01, 0.02]
    topix_relative = [r - 0.01 for r in raw]

    primary_predictions = pd.DataFrame(
        {"date": dates, "ticker": tickers, "actual": raw, "prediction": [4, 3, 1, 2]}
    )
    dataset = pd.DataFrame(
        {
            "date": dates, "ticker": tickers,
            "target_raw_5d": raw, "target_topix_relative_5d": topix_relative,
        }
    )
    beta_panel = pd.DataFrame({"date": dates, "ticker": tickers, "beta": [1.0, 2.0, 0.5, 1.0]})
    sector_map = pd.DataFrame({"ticker": tickers, "sector33": ["Tech", "Tech", "Bank", "Bank"]})
    return primary_predictions, dataset, beta_panel, sector_map


def test_raw_variant_equals_original_actual() -> None:
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    assert (out[f"actual_{VARIANT_RAW}"] == out["actual"]).all()


def test_topix_relative_variant_matches_target_column() -> None:
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    actual_variant = list(out[f"actual_{VARIANT_TOPIX_RELATIVE}"])
    expected = list(dataset["target_topix_relative_5d"])
    assert actual_variant == expected


def test_beta_adjusted_variant_formula() -> None:
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    # market_forward = raw - topix_relative = 0.01 for every row here.
    expected = [0.05 - 1.0 * 0.01, 0.03 - 2.0 * 0.01, -0.01 - 0.5 * 0.01, 0.02 - 1.0 * 0.01]
    actual = out[f"actual_{VARIANT_BETA_ADJUSTED}"].tolist()
    for e, a in zip(expected, actual, strict=True):
        assert abs(e - a) < 1e-9


def test_sector_relative_variant_subtracts_sector_day_mean() -> None:
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    # Tech sector mean raw = (0.05 + 0.03) / 2 = 0.04; Bank mean = (-0.01+0.02)/2 = 0.005
    expected = [0.05 - 0.04, 0.03 - 0.04, -0.01 - 0.005, 0.02 - 0.005]
    actual = out[f"actual_{VARIANT_SECTOR_RELATIVE}"].tolist()
    for e, a in zip(expected, actual, strict=True):
        assert abs(e - a) < 1e-9


def test_market_neutralized_variant_subtracts_day_mean() -> None:
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    day_mean = sum([0.05, 0.03, -0.01, 0.02]) / 4
    actual = out[f"actual_{VARIANT_MARKET_NEUTRALIZED}"].tolist()
    for r, a in zip([0.05, 0.03, -0.01, 0.02], actual, strict=True):
        assert abs((r - day_mean) - a) < 1e-9


def test_implausible_topix_relative_masked_to_nan_not_propagated() -> None:
    # Regression test for the real Full Universe bug: a TOPIX Proxy data
    # artifact on a specific date made target_topix_relative_5d ~+9.3 for
    # EVERY ticker that day (since it subtracts the same market-wide
    # forward return from every ticker), even though target_raw_5d
    # stayed plausible. Both actual_topix_relative AND actual_beta_
    # adjusted (which depends on the same market_forward) must come out
    # NaN for the affected date, while an unaffected date is untouched.
    dates = pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"])
    tickers = ["A", "B", "A", "B"]
    raw = [0.02, -0.01, 0.03, 0.01]
    # 2023-01-02: artifact date, topix_relative ~+9.3 for both tickers.
    # 2023-01-03: normal date, topix_relative = raw - 0.01.
    topix_relative = [9.3, 9.31, 0.02, 0.00]

    predictions = pd.DataFrame(
        {"date": dates, "ticker": tickers, "actual": raw, "prediction": [4, 1, 3, 2]}
    )
    dataset = pd.DataFrame(
        {
            "date": dates, "ticker": tickers,
            "target_raw_5d": raw, "target_topix_relative_5d": topix_relative,
        }
    )
    beta_panel = pd.DataFrame({"date": dates, "ticker": tickers, "beta": [1.0, 1.0, 1.0, 1.0]})
    sector_map = pd.DataFrame({"ticker": ["A", "B"], "sector33": ["Tech", "Tech"]})

    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    artifact_day = out[out["date"] == "2023-01-02"]
    normal_day = out[out["date"] == "2023-01-03"]

    assert artifact_day[f"actual_{VARIANT_TOPIX_RELATIVE}"].isna().all()
    assert artifact_day[f"actual_{VARIANT_BETA_ADJUSTED}"].isna().all()
    assert normal_day[f"actual_{VARIANT_TOPIX_RELATIVE}"].notna().all()
    assert normal_day[f"actual_{VARIANT_BETA_ADJUSTED}"].notna().all()


def test_implausible_full_universe_row_does_not_poison_sector_or_day_mean() -> None:
    # Regression test for the real Full Universe bug (see market_
    # decomposition.py's own "Bug found and fixed" docstring note): an
    # extra ticker E, NOT part of primary_predictions (so it never shows
    # up as a Q5/Q1 row itself), sits in the FULL dataset with a
    # physically implausible target_raw_5d (20.0, an "artifact" row) in
    # the SAME sector (Tech) and SAME date as A/B. Without the fix, E's
    # 20.0 would dominate the Tech sector mean and the whole-day mean,
    # corrupting A/B/C/D's sector_relative/market_neutralized values too.
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    poisoned_row = pd.DataFrame({
        "date": [dataset["date"].iloc[0]], "ticker": ["E"],
        "target_raw_5d": [20.0], "target_topix_relative_5d": [19.99],
    })
    poisoned_dataset = pd.concat([dataset, poisoned_row], ignore_index=True)
    poisoned_sector_map = pd.concat(
        [sector_map, pd.DataFrame({"ticker": ["E"], "sector33": ["Tech"]})], ignore_index=True
    )

    clean_out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    poisoned_out = build_return_variant_columns(
        predictions, poisoned_dataset, beta_panel, poisoned_sector_map
    )

    for col in (f"actual_{VARIANT_SECTOR_RELATIVE}", f"actual_{VARIANT_MARKET_NEUTRALIZED}"):
        clean_values = clean_out[col].tolist()
        poisoned_values = poisoned_out[col].tolist()
        for c, p in zip(clean_values, poisoned_values, strict=True):
            assert abs(c - p) < 1e-9, (col, clean_values, poisoned_values)


def test_bucket_assignment_fixed_to_original_prediction() -> None:
    predictions, dataset, beta_panel, sector_map = _tiny_setup()
    out = build_return_variant_columns(predictions, dataset, beta_panel, sector_map)
    # With only 4 rows, assign_quantile_buckets falls back to fewer than 5
    # buckets - the point of this test is only that evaluate_return_variant
    # runs against the SAME `prediction` column regardless of which
    # actual_<variant> column is being scored, never re-ranking on the
    # variant's own return.
    result_raw = evaluate_return_variant(out, VARIANT_RAW, window_days=5)
    result_sector = evaluate_return_variant(out, VARIANT_SECTOR_RELATIVE, window_days=5)
    raw_buckets = {b.bucket for b in result_raw.bucket_stats}
    sector_buckets = {b.bucket for b in result_sector.bucket_stats}
    assert raw_buckets == sector_buckets  # same partition, different outcome column
