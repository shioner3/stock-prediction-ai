"""OHLCV data-quality validation.

Anomalies are never silently corrected. validate_ohlcv() reports every
issue found; clean_ohlcv() drops the affected rows but always returns the
report alongside the cleaned data so the pipeline can log exactly what was
removed and why (nothing is fixed in place without a trace).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type

import pandas as pd

from providers.base import OHLCV_COLUMNS


@dataclass
class ValidationIssue:
    ticker: str
    date: date_type | None
    rule: str
    detail: str


@dataclass
class ValidationReport:
    ticker: str
    row_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> ValidationReport:
    issues: list[ValidationIssue] = []

    missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(
            ValidationIssue(ticker, None, "missing_columns", f"missing columns: {missing_cols}")
        )
        return ValidationReport(ticker=ticker, row_count=len(df), issues=issues)

    for col in ("open", "high", "low", "close"):
        bad = df[df[col] <= 0]
        for _, row in bad.iterrows():
            issues.append(
                ValidationIssue(ticker, row["date"], "non_positive_price", f"{col}={row[col]}")
            )

    bad_hl = df[df["high"] < df["low"]]
    for _, row in bad_hl.iterrows():
        issues.append(
            ValidationIssue(
                ticker, row["date"], "high_below_low", f"high={row['high']} low={row['low']}"
            )
        )

    bad_vol = df[df["volume"] < 0]
    for _, row in bad_vol.iterrows():
        issues.append(
            ValidationIssue(ticker, row["date"], "negative_volume", f"volume={row['volume']}")
        )

    dupes = df[df.duplicated(subset="date", keep=False)]
    for d in sorted(dupes["date"].unique()):
        issues.append(
            ValidationIssue(ticker, d, "duplicate_date", f"date {d} appears more than once")
        )

    if not df["date"].is_monotonic_increasing:
        issues.append(
            ValidationIssue(
                ticker, None, "unsorted_dates", "date column is not monotonically increasing"
            )
        )

    return ValidationReport(ticker=ticker, row_count=len(df), issues=issues)


def clean_ohlcv(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Return (cleaned_df, report). Rows tied to a dated issue are dropped;
    report.issues lists every row removed and the rule it violated.
    """
    report = validate_ohlcv(df, ticker)
    if not report.issues:
        return df.sort_values("date").reset_index(drop=True), report

    if any(issue.rule == "missing_columns" for issue in report.issues):
        return df.iloc[0:0], report

    bad_dates = {issue.date for issue in report.issues if issue.date is not None}
    cleaned = df[~df["date"].isin(bad_dates)].copy()
    cleaned = (
        cleaned.sort_values("date")
        .drop_duplicates(subset="date", keep="first")
        .reset_index(drop=True)
    )
    return cleaned, report
