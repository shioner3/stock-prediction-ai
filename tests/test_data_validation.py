from __future__ import annotations

from datetime import date

import pandas as pd

from validation.ohlcv import clean_ohlcv, validate_ohlcv


def test_valid_data_has_no_issues(sample_ohlcv_df: pd.DataFrame) -> None:
    report = validate_ohlcv(sample_ohlcv_df, "7203")
    assert report.is_clean
    assert report.row_count == 10


def test_missing_columns_reported() -> None:
    df = pd.DataFrame({"ticker": ["7203"], "date": [date(2024, 1, 1)]})
    report = validate_ohlcv(df, "7203")
    assert not report.is_clean
    assert report.issues[0].rule == "missing_columns"


def test_non_positive_price_detected(sample_ohlcv_df: pd.DataFrame) -> None:
    df = sample_ohlcv_df.copy()
    df.loc[3, "close"] = 0.0
    report = validate_ohlcv(df, "7203")
    rules = [i.rule for i in report.issues]
    assert "non_positive_price" in rules


def test_high_below_low_detected(sample_ohlcv_df: pd.DataFrame) -> None:
    df = sample_ohlcv_df.copy()
    df.loc[2, "high"] = df.loc[2, "low"] - 1
    report = validate_ohlcv(df, "7203")
    rules = [i.rule for i in report.issues]
    assert "high_below_low" in rules


def test_negative_volume_detected(sample_ohlcv_df: pd.DataFrame) -> None:
    df = sample_ohlcv_df.copy()
    df.loc[5, "volume"] = -100
    report = validate_ohlcv(df, "7203")
    rules = [i.rule for i in report.issues]
    assert "negative_volume" in rules


def test_duplicate_date_detected(sample_ohlcv_df: pd.DataFrame) -> None:
    df = sample_ohlcv_df.copy()
    df.loc[1, "date"] = df.loc[0, "date"]
    report = validate_ohlcv(df, "7203")
    rules = [i.rule for i in report.issues]
    assert "duplicate_date" in rules


def test_unsorted_dates_detected(sample_ohlcv_df: pd.DataFrame) -> None:
    df = sample_ohlcv_df.sample(frac=1, random_state=1).reset_index(drop=True)
    report = validate_ohlcv(df, "7203")
    rules = [i.rule for i in report.issues]
    assert "unsorted_dates" in rules


def test_clean_ohlcv_drops_bad_rows_but_reports_them(sample_ohlcv_df: pd.DataFrame) -> None:
    df = sample_ohlcv_df.copy()
    bad_date = df.loc[4, "date"]
    df.loc[4, "close"] = -1.0

    cleaned, report = clean_ohlcv(df, "7203")

    assert not report.is_clean
    assert bad_date not in cleaned["date"].tolist()
    assert len(cleaned) == len(df) - 1


def test_clean_ohlcv_is_noop_on_valid_data(sample_ohlcv_df: pd.DataFrame) -> None:
    cleaned, report = clean_ohlcv(sample_ohlcv_df, "7203")
    assert report.is_clean
    assert len(cleaned) == len(sample_ohlcv_df)
