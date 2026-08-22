from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universe.build import apply_static_filters
from universe.jpx_master import (
    REQUIRED_RAW_COLUMNS,
    load_jpx_master,
    parse_jpx_master,
)


def _write_fake_jpx_xlsx(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    return path


def _row(code: str, name: str, segment: str, sector33: str = "銀行業") -> dict:
    return {
        "日付": 20260731,
        "コード": code,
        "銘柄名": name,
        "市場・商品区分": segment,
        "33業種コード": "9050",
        "33業種区分": sector33,
        "17業種コード": "16",
        "17業種区分": "金融（除く銀行）",
        "規模コード": "6",
        "規模区分": "TOPIX Small 1",
    }


def test_parse_jpx_master_maps_all_known_segments(tmp_path: Path) -> None:
    rows = [
        _row("7203", "トヨタ自動車", "プライム（内国株式）"),
        _row("2371", "カカクコム", "スタンダード（内国株式）"),
        _row("142A", "ジンジブ", "グロース（内国株式）"),
        _row("1306", "TOPIX ETF", "ETF・ETN"),
        _row("8951", "NBF", "REIT・ベンチャーファンド・カントリーファンド・インフラファンド"),
        _row("131A", "PRO銘柄", "PRO Market"),
        _row("9999", "外国株サンプル", "プライム（外国株式）"),
        _row("0000", "出資証券サンプル", "出資証券"),
    ]
    path = _write_fake_jpx_xlsx(tmp_path / "fake.xlsx", rows)

    out = parse_jpx_master(path)

    assert list(out.columns) == [
        "code", "name", "market_segment", "instrument_type",
        "sector33", "sector17", "scale", "snapshot_date",
    ]
    lookup = out.set_index("code")
    assert lookup.loc["7203", ["market_segment", "instrument_type"]].tolist() == ["Prime", "STOCK"]
    assert lookup.loc["2371", ["market_segment", "instrument_type"]].tolist() == [
        "Standard", "STOCK",
    ]
    assert lookup.loc["142A", ["market_segment", "instrument_type"]].tolist() == [
        "Growth", "STOCK",
    ]
    assert lookup.loc["1306", "instrument_type"] == "ETF"
    assert lookup.loc["8951", "instrument_type"] == "REIT"
    assert lookup.loc["131A", "instrument_type"] == "PRO_MARKET"
    assert lookup.loc["9999", "instrument_type"] == "FOREIGN_STOCK"
    assert lookup.loc["0000", "instrument_type"] == "OTHER"


def test_parse_jpx_master_alphanumeric_code_preserved_as_string(tmp_path: Path) -> None:
    path = _write_fake_jpx_xlsx(
        tmp_path / "fake.xlsx", [_row("130A", "サンプル", "グロース（内国株式）")]
    )
    out = parse_jpx_master(path)
    assert out["code"].iloc[0] == "130A"


def test_parse_jpx_master_unrecognized_segment_becomes_other(tmp_path: Path) -> None:
    path = _write_fake_jpx_xlsx(
        tmp_path / "fake.xlsx", [_row("9999", "サンプル", "未知の区分")]
    )
    out = parse_jpx_master(path)
    assert out.loc[0, "market_segment"] == "OTHER"
    assert out.loc[0, "instrument_type"] == "OTHER"


def test_parse_jpx_master_missing_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_excel(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        parse_jpx_master(path)


def test_required_raw_columns_matches_real_jpx_layout() -> None:
    # Locks in the verified real-file column layout (2026-08) so a
    # silent JPX format change is caught immediately rather than
    # producing a wrong/empty Universe.
    assert REQUIRED_RAW_COLUMNS == [
        "日付", "コード", "銘柄名", "市場・商品区分",
        "33業種コード", "33業種区分", "17業種コード", "17業種区分",
        "規模コード", "規模区分",
    ]


def test_load_jpx_master_reuses_cached_file_without_refetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_fake_jpx_xlsx(
        tmp_path / "cached.xlsx", [_row("7203", "トヨタ自動車", "プライム（内国株式）")]
    )

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("fetch_jpx_master_file should not be called when cache exists")

    monkeypatch.setattr("universe.jpx_master.fetch_jpx_master_file", _fail_if_called)

    out = load_jpx_master(path)
    assert len(out) == 1


def test_parsed_jpx_master_flows_through_apply_static_filters(tmp_path: Path) -> None:
    """End-to-end: parse_jpx_master()'s output is directly consumable by
    universe/build.py's EXISTING apply_static_filters() with no changes
    to that function's filtering semantics.
    """
    rows = [
        _row("7203", "トヨタ自動車", "プライム（内国株式）"),
        _row("2371", "カカクコム", "スタンダード（内国株式）"),
        _row("142A", "ジンジブ", "グロース（内国株式）"),
        _row("1306", "TOPIX ETF", "ETF・ETN"),
        _row("8951", "NBF", "REIT・ベンチャーファンド・カントリーファンド・インフラファンド"),
        _row("131A", "PRO銘柄", "PRO Market"),
    ]
    path = _write_fake_jpx_xlsx(tmp_path / "fake.xlsx", rows)
    master = parse_jpx_master(path)

    result = apply_static_filters(
        master, segments=["Prime", "Standard", "Growth"], exclude_etf=True, exclude_reit=True
    )

    assert sorted(result.included["code"]) == ["142A", "2371", "7203"]
    assert sorted(result.excluded["code"]) == ["1306", "131A", "8951"]


@pytest.mark.network
def test_fetch_real_jpx_master_file(tmp_path: Path) -> None:
    """Real network smoke test - deselected by default. Run explicitly
    with: pytest -m network
    """
    from universe.jpx_master import fetch_jpx_master_file

    dest = tmp_path / "data_j.xls"
    fetch_jpx_master_file(dest)
    df = parse_jpx_master(dest)
    assert len(df) > 3000
    assert (df["instrument_type"] == "STOCK").sum() > 3000
