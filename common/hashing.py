"""Config/data hashing (Phase 6 section 28-29): lets every Walk Forward
result be traced back to the exact config and data files that produced
it - "どの設定で得られたOOS結果か" should always be answerable from the
result alone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_files(paths: list[Path]) -> str:
    """A single combined sha256 over multiple files, order-stable
    (sorted by path string) so the result doesn't depend on argument
    order - both the path and the file's bytes are hashed, so a rename
    with identical content still changes the hash.
    """
    hasher = hashlib.sha256()
    for path in sorted(paths, key=str):
        hasher.update(str(path).encode("utf-8"))
        hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()
