"""Real JPX listed-issues master list (Phase 6.5): fetches and parses the
official "東証上場銘柄一覧" file JPX publishes for free, no account or
API key required.

    https://www.jpx.co.jp/markets/statistics-equities/misc/01.html

The actual Excel attachment URL is discovered automatically from the
official JPX page because JPX periodically replaces the attachment path.

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
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

logger = logging.getLogger(__name__)


# Stable official JPX page.
#
# IMPORTANT:
# Do not hard-code the actual Excel attachment URL here.
# JPX periodically replaces that attachment URL.
JPX_MASTER_PAGE_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
)

# Backward compatibility:
# Existing callers/tests may import DEFAULT_JPX_MASTER_URL.
#
# This now points to the stable JPX page, NOT to the old data_j.xls
# attachment URL.
DEFAULT_JPX_MASTER_URL = JPX_MASTER_PAGE_URL


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
    "REIT・ベンチャーファンド・カントリーファンド・インフラファンド": (
        "REIT",
        "REIT",
    ),
    "PRO Market": ("PRO_MARKET", "PRO_MARKET"),
    "プライム（外国株式）": ("FOREIGN_STOCK", "FOREIGN_STOCK"),
    "スタンダード（外国株式）": ("FOREIGN_STOCK", "FOREIGN_STOCK"),
    "グロース（外国株式）": ("FOREIGN_STOCK", "FOREIGN_STOCK"),
    "出資証券": ("OTHER", "OTHER"),
}


REQUIRED_RAW_COLUMNS = [
    "日付",
    "コード",
    "銘柄名",
    "市場・商品区分",
    "33業種コード",
    "33業種区分",
    "17業種コード",
    "17業種区分",
    "規模コード",
    "規模区分",
]


@dataclass(frozen=True)
class JpxMasterFetchResult:
    path: Path
    snapshot_date: str  # YYYYMMDD, from the file's own 日付 column
    row_count: int


class _JpxLinkParser(HTMLParser):
    """Extract href/text pairs from the official JPX page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return

        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return

        text = "".join(self._current_text).strip()
        self.links.append((self._current_href, text))

        self._current_href = None
        self._current_text = []


def _fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch a URL using a browser-like request."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Referer": JPX_MASTER_PAGE_URL,
        },
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _discover_jpx_master_url() -> str:
    """Discover the current JPX listed-issues Excel URL.

    JPX periodically replaces the actual Excel attachment URL.
    Never depend on the old /tvdivq.../data_j.xls path.
    """

    html = _fetch_url(JPX_MASTER_PAGE_URL).decode(
        "utf-8",
        errors="replace",
    )

    parser = _JpxLinkParser()
    parser.feed(html)

    candidates: list[tuple[str, str]] = []

    for href, text in parser.links:
        absolute_url = urljoin(JPX_MASTER_PAGE_URL, href)
        lower_url = absolute_url.lower()

        # JPX currently publishes this master as a legacy .xls file.
        if ".xls" not in lower_url:
            continue

        candidates.append((absolute_url, text))

    if not candidates:
        raise RuntimeError(
            "Could not find an Excel (.xls) download link on the official "
            f"JPX listed-issues page: {JPX_MASTER_PAGE_URL}"
        )

    # Prefer a link whose visible text identifies the listed-issues file.
    preferred = [
        url
        for url, text in candidates
        if "東証上場銘柄一覧" in text
    ]

    if preferred:
        selected = preferred[0]
    else:
        # The filename is expected to remain data_j.xls even though the
        # attachment directory can change.
        data_j_candidates = [
            url
            for url, _ in candidates
            if url.rstrip("/").lower().endswith("/data_j.xls")
        ]

        if data_j_candidates:
            selected = data_j_candidates[0]
        elif len(candidates) == 1:
            selected = candidates[0][0]
        else:
            raise RuntimeError(
                "Multiple Excel links were found on the JPX listed-issues "
                "page, but none could be identified safely: "
                f"{candidates}"
            )

    logger.info("Discovered current JPX master URL: %s", selected)

    return selected


def fetch_jpx_master_file(
    dest_path: Path,
    url: str | None = None,
) -> Path:
    """Downloads the raw .xls file as-is.

    If url is omitted, the current Excel attachment URL is discovered
    automatically from JPX's official listed-issues page.

    The downloaded file is first written to a temporary file and
    validated with pandas. Only after successful validation is the
    existing destination replaced. This prevents a failed/partial
    download from destroying a previously valid cache.
    """

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve compatibility with callers/tests that explicitly provide
    # a non-default URL.
    if url is None or url == DEFAULT_JPX_MASTER_URL:
        url = _discover_jpx_master_url()

    logger.info("Downloading JPX master file: %s", url)

    data = _fetch_url(url)

    if not data:
        raise RuntimeError(
            "JPX master download returned an empty response."
        )

    # Write to a temporary file first.
    tmp_path = dest_path.with_suffix(
        dest_path.suffix + ".tmp"
    )

    try:
        tmp_path.write_bytes(data)

        # Validate that pandas can actually read the downloaded file.
        raw = pd.read_excel(
            tmp_path,
            dtype={"コード": str},
        )

        missing = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)

        if missing:
            raise ValueError(
                "Downloaded JPX master is missing expected columns: "
                f"{sorted(missing)}"
            )

        # A valid JPX listed-issues master should contain thousands of
        # securities. This protects against an HTML error page being
        # returned with HTTP 200.
        if len(raw) < 100:
            raise ValueError(
                "Downloaded JPX master contains suspiciously few rows: "
                f"{len(raw)}"
            )

        # Atomic replacement after validation.
        tmp_path.replace(dest_path)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info(
        "JPX master file downloaded and validated: %s (%d bytes)",
        dest_path,
        dest_path.stat().st_size,
    )

    return dest_path


def parse_jpx_master(xls_path: Path) -> pd.DataFrame:
    """Returns columns: code, name, market_segment, instrument_type,
    sector33, sector17, scale, snapshot_date. One row per security in
    the file (both ordinary stocks and everything universe/build.py's
    static filters are meant to exclude - filtering happens in
    universe/build.py, not here).
    """

    raw = pd.read_excel(
        xls_path,
        dtype={"コード": str},
    )

    missing = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)

    if missing:
        raise ValueError(
            f"JPX master file at {xls_path} is missing expected columns: "
            f"{sorted(missing)} - JPX may have changed the file layout; "
            "update universe/jpx_master.py"
        )

    unmapped = (
        set(raw["市場・商品区分"].unique())
        - set(_SEGMENT_MAP)
    )

    if unmapped:
        logger.warning(
            "JPX master file contains %d unrecognized "
            "市場・商品区分 value(s), classified as OTHER/OTHER: %s",
            len(unmapped),
            sorted(unmapped),
        )

    def _map_segment(
        raw_value: str,
    ) -> tuple[str, str]:
        return _SEGMENT_MAP.get(
            raw_value,
            ("OTHER", "OTHER"),
        )

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
    dest_path: Path,
    url: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and parse the JPX master.

    If the destination file does not exist, the current JPX Excel URL
    is automatically discovered.

    If force_refresh=True, the current JPX Excel URL is discovered and
    downloaded even when a cached file already exists.

    If an explicit non-default URL is supplied, that URL is preserved
    for backward compatibility and testing.
    """

    dest_path = Path(dest_path)

    if force_refresh or not dest_path.exists():
        fetch_jpx_master_file(
            dest_path,
            url=url,
        )
    else:
        logger.info(
            "reusing cached JPX master file: %s",
            dest_path,
        )

    return parse_jpx_master(dest_path)