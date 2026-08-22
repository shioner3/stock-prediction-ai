"""No-lookahead verification for the Feature layer.

This is the single most important test file in the project (see README's
"NO LOOKAHEAD" section). It implements the four tests specified for
Phase 2:

    A. Truncation Test        - Feature(t) is identical whether computed
                                 from a dataset ending at t, or from a
                                 longer dataset and read at row t.
    B. Future Perturbation     - randomising all of t+1..end leaves
       Test                     Feature(t) completely unchanged.
    C. Feature Dependency Test - features/ never imports a forward-return
                                 module.
    D. Mathematical Property   - sanity checks that don't depend on any
       Test                     lookahead question (monotonic price ->
                                 positive slope, flat price -> zero
                                 return/volatility, etc).

Phase 3 adds a market benchmark (Relative Strength) as a second input, so
Test A and Test B below now also supply a market_df (truncated/perturbed
in step with the stock data) to prove those same guarantees extend to
rs_5d/20d/60d. A market-only counterpart of Test B - perturbing the
market's future rows while leaving the stock alone - is added separately
below ("RS Test C"), since that failure mode doesn't exist for any other
feature. RS's date-alignment invariants (row-order independence, missing
market dates never forward-filled) live in
tests/test_features_relative_strength.py, which is where they're most
directly exercised.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel

_OHLCV_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}


def _feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in _OHLCV_COLUMNS]


# --- Test A: Truncation Test ------------------------------------------------


def test_truncation_feature_matches_between_short_and_long_datasets() -> None:
    full = make_synthetic_ohlcv(300, seed=1)
    full_market = make_synthetic_ohlcv(300, seed=101, ticker="TOPIX")
    panel_full = compute_feature_panel(full, market_df=full_market)
    feature_cols = _feature_columns(panel_full)

    for t in (100, 150, 200, 250, 290):
        truncated = full.iloc[: t + 1].reset_index(drop=True)
        truncated_market = full_market.iloc[: t + 1].reset_index(drop=True)
        panel_truncated = compute_feature_panel(truncated, market_df=truncated_market)

        row_full = panel_full.loc[t, feature_cols].astype(float)
        row_truncated = panel_truncated.iloc[-1][feature_cols].astype(float)

        pd.testing.assert_series_equal(row_full, row_truncated, check_names=False, obj=f"t={t}")


# --- Test B: Future Perturbation Test ---------------------------------------


def test_perturbing_future_rows_does_not_change_past_features() -> None:
    """Stock's own future rows are perturbed; market_df is supplied but
    held fixed - this is Test B for every feature including RS (RS's
    market-side counterpart is the dedicated test below, "RS Test C").
    """
    base = make_synthetic_ohlcv(300, seed=2)
    market = make_synthetic_ohlcv(300, seed=102, ticker="TOPIX")
    panel_base = compute_feature_panel(base, market_df=market)
    feature_cols = _feature_columns(panel_base)

    rng = np.random.default_rng(42)
    for t in (100, 150, 200, 250):
        perturbed = base.copy()
        future_mask = perturbed.index > t
        n_future = int(future_mask.sum())
        for col, low, high in [
            ("open", 0.5, 1.5),
            ("high", 0.5, 1.5),
            ("low", 0.5, 1.5),
            ("close", 0.5, 1.5),
            ("volume", 0.5, 3.0),
        ]:
            perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] * rng.uniform(
                low, high, size=n_future
            )

        panel_perturbed = compute_feature_panel(perturbed, market_df=market)

        row_base = panel_base.loc[t, feature_cols].astype(float)
        row_perturbed = panel_perturbed.loc[t, feature_cols].astype(float)

        pd.testing.assert_series_equal(
            row_base, row_perturbed, check_names=False, obj=f"t={t}"
        )


def test_perturbing_future_market_rows_does_not_change_past_rs() -> None:
    """RS Test C: the mirror image of the test above - stock_df is held
    fixed and only the market benchmark's future rows are perturbed.
    This failure mode is specific to Relative Strength (no other feature
    reads a second DataFrame), so it needs its own test.
    """
    stock = make_synthetic_ohlcv(300, seed=3)
    market_base = make_synthetic_ohlcv(300, seed=103, ticker="TOPIX")
    panel_base = compute_feature_panel(stock, market_df=market_base)

    rng = np.random.default_rng(43)
    for t in (100, 150, 200, 250):
        market_perturbed = market_base.copy()
        future_mask = market_perturbed.index > t
        n_future = int(future_mask.sum())
        market_perturbed.loc[future_mask, "close"] = market_perturbed.loc[
            future_mask, "close"
        ] * rng.uniform(0.5, 1.5, size=n_future)

        panel_perturbed = compute_feature_panel(stock, market_df=market_perturbed)

        for window in (5, 20, 60):
            col = f"rs_{window}d"
            base_val = panel_base.loc[t, col]
            perturbed_val = panel_perturbed.loc[t, col]
            if pd.isna(base_val):
                assert pd.isna(perturbed_val), f"t={t} {col}"
            else:
                assert np.isclose(base_val, perturbed_val), f"t={t} {col}"


# --- Test C: Feature Dependency Test ----------------------------------------


def test_feature_modules_do_not_import_forward_returns() -> None:
    features_dir = Path(__file__).resolve().parent.parent / "features"
    offending: list[str] = []

    for path in features_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            if any(n and ("targets" in n or "forward_returns" in n) for n in names):
                offending.append(f"{path.name}: {names}")

    assert not offending, f"features/ must not import targets/forward_returns: {offending}"


def test_feature_modules_do_not_import_providers() -> None:
    """Feature code takes an OHLCV DataFrame as input; it must never call
    a vendor Provider directly (that would break the "Feature/Signal
    layers never know about yfinance" separation from Phase 1).
    """
    features_dir = Path(__file__).resolve().parent.parent / "features"
    offending: list[str] = []

    for path in features_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            if any(n and n.startswith("providers") for n in names):
                offending.append(f"{path.name}: {names}")

    assert not offending, f"features/ must not import providers/: {offending}"


# --- Test D: Mathematical Property Test -------------------------------------


def test_monotonic_increasing_price_gives_positive_sma_slope() -> None:
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(1000.0, 2000.0, n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 50_000.0),
        }
    )
    panel = compute_feature_panel(df)
    tail = panel.iloc[250:]

    assert (tail["sma_5_slope"] > 0).all()
    assert (tail["sma_20_slope"] > 0).all()
    assert (tail["sma_50_slope"] > 0).all()
    assert (tail["sma_200_slope"] > 0).all()


def test_monotonic_decreasing_price_gives_negative_sma_slope() -> None:
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(2000.0, 1000.0, n)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 50_000.0),
        }
    )
    panel = compute_feature_panel(df)
    tail = panel.iloc[250:]

    assert (tail["sma_20_slope"] < 0).all()


def test_flat_price_series_gives_zero_return_and_volatility_and_neutral_rsi() -> None:
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 1500.0)
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": [d.date() for d in dates],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 50_000.0),
        }
    )
    panel = compute_feature_panel(df)
    tail = panel.iloc[70:]

    assert np.allclose(tail["return_1d"], 0.0)
    assert np.allclose(tail["return_5d"], 0.0)
    assert np.allclose(tail["volatility_20d"], 0.0)
    assert np.allclose(tail["rsi_14"], 50.0)
    assert np.allclose(tail["macd"], 0.0, atol=1e-9)


# --- Phase 8 Test D: Data Order ----------------------------------------------
#
# compute_feature_panel()'s own docstring documents a PRECONDITION: input
# must already be "sorted ascending by date" - it does not defend against
# unsorted input itself (rolling()/shift() would silently give wrong
# results on a scrambled frame, which is expected given that contract, not
# a look-ahead bug). The claim actually worth testing is about the REAL
# pipeline: validation/ohlcv.py::clean_ohlcv() is unconditionally called
# on every provider fetch BEFORE any Feature is computed, and it always
# sorts - so no code path in this project ever hands compute_feature_panel
# unsorted data, regardless of what order a Provider's raw response
# happens to arrive in.


def test_clean_ohlcv_output_is_order_independent_and_correctly_sorted() -> None:
    from validation.ohlcv import clean_ohlcv

    ordered = make_synthetic_ohlcv(150, seed=42)
    shuffled = ordered.sample(frac=1, random_state=7).reset_index(drop=True)

    cleaned_ordered, report_ordered = clean_ohlcv(ordered, "TEST")
    cleaned_shuffled, report_shuffled = clean_ohlcv(shuffled, "TEST")

    assert report_ordered.is_clean
    # The shuffled input correctly gets FLAGGED ("unsorted_dates") - this
    # is validate_ohlcv() surfacing a genuine data-quality observation
    # about the RAW input, not a bug. What matters for Test D is that the
    # CLEANED output is identical either way (clean_ohlcv() always sorts,
    # regardless of what it flagged).
    assert not report_shuffled.is_clean
    assert any(i.rule == "unsorted_dates" for i in report_shuffled.issues)
    assert cleaned_shuffled["date"].is_monotonic_increasing
    pd.testing.assert_frame_equal(cleaned_ordered, cleaned_shuffled)


def test_feature_panel_identical_after_pipeline_cleaning_regardless_of_fetch_order() -> None:
    from validation.ohlcv import clean_ohlcv

    ordered = make_synthetic_ohlcv(300, seed=43)
    market = make_synthetic_ohlcv(300, seed=143, ticker="TOPIX")
    shuffled = ordered.sample(frac=1, random_state=11).reset_index(drop=True)

    cleaned_ordered, _ = clean_ohlcv(ordered, "TEST")
    cleaned_shuffled, _ = clean_ohlcv(shuffled, "TEST")

    panel_ordered = compute_feature_panel(cleaned_ordered, market_df=market)
    panel_shuffled = compute_feature_panel(cleaned_shuffled, market_df=market)

    pd.testing.assert_frame_equal(panel_ordered, panel_shuffled)
