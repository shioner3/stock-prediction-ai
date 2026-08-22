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
    (sorted by POSIX path string) so the result doesn't depend on
    argument order - both the path and the file's bytes are hashed, so
    a rename with identical content still changes the hash.

    Uses `Path.as_posix()`, NOT `str(path)`, for the path component of
    the hash input - `str()` renders OS-native separators ("\\" on
    Windows, "/" on POSIX), which would make the exact same file set
    with byte-identical content hash differently depending on which OS
    computed it. `as_posix()` always yields "/"-separated paths on every
    platform, so the hash is a pure function of (relative path string,
    file bytes) - platform-independent, as an integrity hash meant to be
    verified across environments (e.g. Windows locally, Linux in CI)
    must be. See tests/test_common_hashing.py's
    test_hash_files_platform_independent_of_path_separator_style.
    """
    hasher = hashlib.sha256()
    for path in sorted(paths, key=lambda p: Path(p).as_posix()):
        hasher.update(Path(path).as_posix().encode("utf-8"))
        hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()
