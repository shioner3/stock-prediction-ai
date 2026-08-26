"""Fills in Realized Returns for previously-logged Predictions once their
Horizon has matured (enough calendar days have passed AND been fetched
for `Close[observation_date + horizon]` to exist). Reuses the SAME
`target_{definition}_{horizon}d` columns every V3 Phase already computes
(`v3.dataset.build_v3_dataset()` for Raw/TOPIX-relative, `v3.residual.
reproduce.build_augmented_dataset()` for Beta-adjusted Residual/Sector-
relative) - a maturable realized return is nothing but that SAME column,
looked up at the (observation_date, ticker) row, now that enough future
data has arrived for it to be non-NaN. No new return-computation formula.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from v3.frozen.observation_log import RealizedReturnLogEntry
from v3.residual.reproduce import target_column_for


def compute_realized_return_entries(
    pending_predictions: list[dict], augmented_dataset_with_future: pd.DataFrame,
    already_realized_keys: set[tuple[str, str, str]] | None = None,
) -> list[RealizedReturnLogEntry]:
    """pending_predictions: raw dicts loaded from predictions_log.jsonl
    (PredictionLogEntry shape). augmented_dataset_with_future: the SAME
    Feature/Target dataset used for training/observation, rebuilt against
    whatever OHLCV history is available NOW (naturally extends further
    into the future as more days pass and get fetched).
    """
    if already_realized_keys is None:
        already_realized_keys = set()

    lookup_cols = ["date", "ticker"] + sorted({
        target_column_for(p["target_definition"], p["horizon"]) for p in pending_predictions
    })
    lookup = augmented_dataset_with_future[lookup_cols].copy()
    # Normalize to plain datetime.date - the dataset's own "date" column
    # may be a pandas Timestamp (from a parquet/CSV round-trip) or already
    # a datetime.date, and the lookup key built below is always a plain
    # date; mismatched types here silently make every .loc lookup miss
    # (KeyError -> treated as "not yet matured" - a real bug caught by
    # this module's own test suite before any real run).
    lookup["date"] = pd.to_datetime(lookup["date"]).dt.date
    lookup = lookup.set_index(["date", "ticker"])

    logged_at = datetime.now().isoformat()
    entries: list[RealizedReturnLogEntry] = []
    for p in pending_predictions:
        key = (p["observation_date"], p["ticker"], p["model_id"])
        if key in already_realized_keys:
            continue
        obs_date = pd.to_datetime(p["observation_date"]).date()
        target_col = target_column_for(p["target_definition"], p["horizon"])
        try:
            value = lookup.loc[(obs_date, p["ticker"]), target_col]
        except KeyError:
            continue
        if pd.isna(value):
            continue
        entries.append(
            RealizedReturnLogEntry(
                observation_date=p["observation_date"], ticker=p["ticker"], model_id=p["model_id"],
                target_definition=p["target_definition"], horizon=p["horizon"],
                realized_return=float(value), realized_date=datetime.now().date().isoformat(),
                logged_at=logged_at,
            )
        )
    return entries
