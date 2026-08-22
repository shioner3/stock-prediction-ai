"""Real JPX listed-issues master list (Phase 6.5): fetches and parses the
official "東証上場銘柄一覧" file JPX publishes for free, no account or
API key required.

    https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
    -> data_j.xls (legacy .xls format, updated periodically by JPX)

This is a CURRENT-DAY SNAPSHOT ONLY - JPX does not publish delisted-
security history or historical listing dates through this free channel
(the only JPX source found with that information, J-Quants DataCube,
requires account registration and was intentionally not pursued - see
README's "Survivorship Bias" section). Every Universe built from this
file is therefore a "Current Universe" (today's constituents projected
backward), not a "Historical Universe" - callers must record
survivorship_bias_warning=True alongside any result derived from it.

Real column layout (verified against the live file, 2026-08):
    日付, コード, 銘柄名, 市場・商品区分, 33業種コード, 33業種区分,
    17業種コード, 17業種区分, 規模コード, 規模区分

市場・商品区分 alone answers BOTH questions universe/build.py needs -
which market segment (Prime/Standard/Growth) AND whether the row is an
ordinary domestic stock at all (vs ETF/ETN/REIT/PRO Market/foreign
stock/other) - unlike the hand-written Phase 1 sample CSV, which
invented a sector33-keyword-based ETF/REIT heuristic because no real
data was available yet. parse_jpx_master() below produces the more
reliable classification directly from this column; universe/build.py's
classify_instrument_type() (sector33-keyword based) remains as a
fallback for callers still using the old sample-CSV shape.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_JPX_MASTER_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

# 市場・商品区分 (raw JPX text) -> (normalized market_segment, instrument_type).
# market_segment is deliberately NOT normalized for non-domestic-stock rows
# (kept as a stable internal label, never "Prime"/"Standard"/"Growth") so
# that universe/build.py's existing `~df["market_segment"].isin(segments)`
# check excludes them automatically - no new filter flag was introduced.
_SEGMENT_MAP: dict[str, tuple[str, str]] = {
    "プライム（内国株式）": ("Prime", "STOCK"),
    "スタンダード（内国株式）": ("Standard", "STOCK"),
    "グロース（内国株式）": ("Growth", "STOCK"),
    "ETF・ETN": ("ETF_ETN", "ETF"),
    "REIT・ベンチャーファンド・カントリーファンド・インフラファンド": ("REIT", "REIT"),
    "PRO Market": ("PRO_MARKET", "PRO_MARKET"),
    "プライム（外国株式）": ("FOREIGN_STOCK", "FOREIGN_STOCK"),
    "スタンダード（外国株式）": ("FOREIGN_STOCK", "FOREIGN_STOCK"),
    "グロース（外国株式）": ("FOREIGN_STOCK", "FOREIGN_STOCK"),
    "出資証券": ("OTHER", "OTHER"),
}

REQUIRED_RAW_COLUMNS = [
    "日付", "コード", "銘柄名", "市場・商品区分",
    "33業種コード", "33業種区分", "17業種コード", "17業種区分",
    "規模コード", "規模区分",
]


@dataclass(frozen=True)
class JpxMasterFetchResult:
    path: Path
    snapshot_date: str  # YYYYMMDD, from the file's own 日付 column
    row_count: int


def fetch_jpx_master_file(dest_path: Path, url: str = DEFAULT_JPX_MASTER_URL) -> Path:
    """Downloads the raw .xls file as-is (no parsing) - a real HTTP GET
    against JPX's own server, no scraping of a third-party mirror. Uses
    a browser-like User-Agent because JPX's server returns errors to
    some default Python user agents.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        dest_path.write_bytes(response.read())
    logger.info("JPX master file downloaded: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    return dest_path


def parse_jpx_master(xls_path: Path) -> pd.DataFrame:
    """Returns columns: code, name, market_segment, instrument_type,
    sector33, sector17, scale, snapshot_date. One row per security in
    the file (both ordinary stocks and everything universe/build.py's
    static filters are meant to exclude - filtering happens in
    universe/build.py, not here).
    """
    raw = pd.read_excel(xls_path, dtype={"コード": str})
    missing = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(
            f"JPX master file at {xls_path} is missing expected columns: {sorted(missing)} "
            "- JPX may have changed the file layout; update universe/jpx_master.py"
        )

    unmapped = set(raw["市場・商品区分"].unique()) - set(_SEGMENT_MAP)
    if unmapped:
        logger.warning(
            "JPX master file contains %d unrecognized 市場・商品区分 value(s), "
            "classified as OTHER/OTHER: %s", len(unmapped), sorted(unmapped),
        )

    def _map_segment(raw_value: str) -> tuple[str, str]:
        return _SEGMENT_MAP.get(raw_value, ("OTHER", "OTHER"))

    mapped = raw["市場・商品区分"].apply(_map_segment)

    out = pd.DataFrame(
        {
            "code": raw["コード"].astype(str).str.strip(),
            "name": raw["銘柄名"].astype(str).str.strip(),
            "market_segment": mapped.apply(lambda t: t[0]),
            "instrument_type": mapped.apply(lambda t: t[1]),
            "sector33": raw["33業種区分"].astype(str),
            "sector17": raw["17業種区分"].astype(str),
            "scale": raw["規模区分"].astype(str),
            "snapshot_date": raw["日付"].astype(str),
        }
    )
    return out


def load_jpx_master(
    dest_path: Path, url: str = DEFAULT_JPX_MASTER_URL, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch (if dest_path doesn't already exist, or force_refresh) and
    parse in one step - the entry point pipeline code should call.
    """
    dest_path = Path(dest_path)
    if force_refresh or not dest_path.exists():
        fetch_jpx_master_file(dest_path, url=url)
    else:
        logger.info("reusing cached JPX master file: %s", dest_path)
    return parse_jpx_master(dest_path)
