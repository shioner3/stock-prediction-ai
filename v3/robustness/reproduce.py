"""Spec section 1/23: freeze-and-verify V3-3 reproduction.

Rebuilds the Full Universe dataset and WFO windows with COMPLETELY
UNMODIFIED V3-3 code (`v3.dataset.build_v3_dataset`, `v3.validation.
windows.get_v3_3_windows`), verifies the resulting code_hash/config_hash/
feature_hash/dataset_hash are bit-identical to Phase V3-3's own saved
values (`data/v3/reports/v3_3_full_universe_oos_report.json`) - a hash
mismatch means V3-1/V3-2/V3-3's frozen spec drifted and this Phase must
STOP per spec section 25, not proceed.

Then re-runs `v3.validation.train_predict.run_model_a_on_window()`
(unmodified) for the Primary target (`target_raw_5d`) and the 3 secondary
Horizon targets (`target_raw_10d/15d/20d`) across all 6 windows, to
regenerate raw per-row OOS predictions for this Phase's decomposition
work (V3-3's own JSON report holds these rows too, but at ~4GB it is not
practical to re-parse here). This is a REPRODUCTION of already-frozen
Phase V3-3 combinations, not a new independent OOS run (spec section 2) -
verified as such by comparing the regenerated Primary Q5-Q1 spread
against V3-3's own reported value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.walk_forward import WalkForwardWindow
from v3.config.loader import V3Config
from v3.dataset import build_v3_dataset, load_universe_tickers
from v3.hash import (
    current_v3_code_hash,
    current_v3_config_hash,
    current_v3_feature_hash,
    hash_dataframe,
)
from v3.validation.train_predict import run_model_a_on_window
from v3.validation.wfo_config import PRIMARY_TARGET_COL, SECONDARY_HORIZON_TARGETS
from v3.validation.windows import get_v3_3_windows

V3_3_REPORT_PATH = Path("data/v3/reports/v3_3_full_universe_oos_report.json")
REPRODUCE_TARGETS = (PRIMARY_TARGET_COL, *SECONDARY_HORIZON_TARGETS)  # 5d, 10d, 15d, 20d (raw)

_HASH_FIELD_RE = re.compile(r'"(code_hash|config_hash|feature_hash|dataset_hash)":\s*"([0-9a-f]+)"')


@dataclass(frozen=True)
class V3_3ReferenceHashes:
    code_hash: str
    config_hash: str
    feature_hash: str
    dataset_hash: str


def load_v3_3_reference_hashes(
    path: Path = V3_3_REPORT_PATH, tail_bytes: int = 8192
) -> V3_3ReferenceHashes:
    """Reads only the LAST `tail_bytes` of the (multi-GB) V3-3 report JSON
    - the 4 hash fields are written last (see `scripts/
    run_v3_3_full_universe_oos.py`'s payload dict key order), so a full
    parse is never needed just to recover them.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - tail_bytes))
        tail = f.read().decode("utf-8", errors="ignore")
    found = dict(_HASH_FIELD_RE.findall(tail))
    missing = {"code_hash", "config_hash", "feature_hash", "dataset_hash"} - found.keys()
    if missing:
        raise ValueError(f"could not recover hash fields {missing} from tail of {path}")
    return V3_3ReferenceHashes(**found)  # type: ignore[arg-type]


@dataclass(frozen=True)
class HashVerification:
    """`code_hash_match` is reported for transparency only and NEVER
    gates `all_match`: `v3.hash.current_v3_code_hash()` hashes every
    `.py` file under the whole `v3/` tree, which now also includes this
    Phase's OWN new `v3/robustness/` package - comparing it against
    V3-3's pre-`robustness/` code_hash would always mismatch, for a
    reason that has nothing to do with V3-1/V3-2/V3-3 drifting. The
    actual frozen-spec gate (matching spec section 1's "V3-3の...
    dataset_hash/config_hashを保存し...一致確認する") is `config_hash`
    (Feature/Target Registry + V3 settings) and `dataset_hash` (the
    built Full Universe dataset itself) - both narrow, both directly
    tied to what must not have changed. `feature_hash` (narrower than
    code_hash - only `v3/features/`+`v3/targets/`) is included as a
    bonus, strictly-stronger-than-required check. Model determinism
    (the spec's third named hash, `model_hash`) is verified empirically
    instead, via `v3_3_reference.py`'s recorded Primary Q5-Q1 spread
    reproduction check, since Phase V3-3's own saved report never
    actually persisted a `model_hash` field to compare against.
    """

    code_hash_match: bool
    config_hash_match: bool
    feature_hash_match: bool
    dataset_hash_match: bool
    current: V3_3ReferenceHashes
    reference: V3_3ReferenceHashes

    @property
    def all_match(self) -> bool:
        return self.config_hash_match and self.feature_hash_match and self.dataset_hash_match


def verify_against_v3_3(
    dataset: pd.DataFrame, reference: V3_3ReferenceHashes | None = None
) -> HashVerification:
    if reference is None:
        reference = load_v3_3_reference_hashes()
    current = V3_3ReferenceHashes(
        code_hash=current_v3_code_hash(),
        config_hash=current_v3_config_hash(),
        feature_hash=current_v3_feature_hash(),
        dataset_hash=hash_dataframe(dataset),
    )
    return HashVerification(
        code_hash_match=current.code_hash == reference.code_hash,
        config_hash_match=current.config_hash == reference.config_hash,
        feature_hash_match=current.feature_hash == reference.feature_hash,
        dataset_hash_match=current.dataset_hash == reference.dataset_hash,
        current=current, reference=reference,
    )


def build_frozen_dataset(v3_config: V3Config, limit_tickers: int | None = None) -> pd.DataFrame:
    """Identical call to Phase V3-3's own STEP 3 - no argument differs."""
    tickers = load_universe_tickers(v3_config)
    if limit_tickers:
        tickers = tickers[:limit_tickers]
    return build_v3_dataset(tickers, v3_config)


def reproduce_predictions(
    dataset: pd.DataFrame, windows: list[WalkForwardWindow] | None = None,
    targets: tuple[str, ...] = REPRODUCE_TARGETS,
) -> dict[str, pd.DataFrame]:
    """Re-runs the frozen Model A train/predict loop for each target in
    `targets` across every window, pooling each target's OOS predictions
    into one DataFrame (date/ticker/actual/prediction). Bit-identical to
    what Phase V3-3's own orchestrator already computed internally for
    these same 4 combinations (Primary + the 3 secondary Horizons) - see
    module docstring for why this Phase regenerates rather than re-reads
    V3-3's saved JSON.
    """
    if windows is None:
        windows = get_v3_3_windows()
    out: dict[str, pd.DataFrame] = {}
    for target_col in targets:
        pooled = []
        for window in windows:
            wp = run_model_a_on_window(dataset, window, target_col)
            tagged = wp.predictions.copy()
            tagged["window_index"] = window.index
            pooled.append(tagged)
        out[target_col] = pd.concat(pooled, ignore_index=True)
    return out


def save_predictions(predictions: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for target_col, df in predictions.items():
        path = out_dir / f"{target_col}.parquet"
        df.to_parquet(path, index=False)
        paths[target_col] = path
    return paths


def load_saved_predictions(
    out_dir: Path, targets: tuple[str, ...] = REPRODUCE_TARGETS
) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(out_dir / f"{t}.parquet") for t in targets}
