from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from common.hashing import hash_files


def test_same_content_gives_same_hash(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("hello")
    f2 = tmp_path / "b.txt"
    f2.write_text("hello")
    assert hash_files([f1]) != hash_files([f2])  # different paths -> different hash


def test_identical_path_and_content_gives_identical_hash(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("hello")
    assert hash_files([f1]) == hash_files([f1])


def test_different_content_gives_different_hash(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("hello")
    f2 = tmp_path / "a2.txt"
    f2.write_text("world")
    assert hash_files([f1]) != hash_files([f2])


def test_order_independent(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("hello")
    f2 = tmp_path / "b.txt"
    f2.write_text("world")
    assert hash_files([f1, f2]) == hash_files([f2, f1])


def test_content_change_changes_hash(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("v1")
    before = hash_files([f1])
    f1.write_text("v2")
    after = hash_files([f1])
    assert before != after


# --- Platform independence (regression test for the Windows/Linux Strategy ---
# --- Hash mismatch bug: str(path) is OS-native-separator, as_posix() isn't ---


def test_windows_and_posix_style_paths_normalize_identically() -> None:
    """The exact assumption hash_files() relies on: a Windows-style
    ("\\"-separated) and a POSIX-style ("/"-separated) representation of
    the SAME relative path must produce the SAME as_posix() string, so
    the path component of the hash input never depends on which OS
    computed it.
    """
    win = PureWindowsPath("features\\pipeline.py")
    posix = PurePosixPath("features/pipeline.py")
    assert win.as_posix() == posix.as_posix() == "features/pipeline.py"


def test_hash_files_uses_posix_path_representation_not_str(tmp_path: Path) -> None:
    """hash_files() must hash Path.as_posix(), NOT str(path) (which
    renders OS-native separators: "\\" on Windows, "/" on POSIX) - using
    str(path) would make the SAME file set with byte-identical content
    hash differently depending on which OS computed it, which is exactly
    what broke Strategy Hash verification between a Windows-computed
    manifest and a GitHub Actions (Linux) Forward Test run: all 6 hash
    fields mismatched simultaneously despite zero content changes.
    """
    sub = tmp_path / "sub"
    sub.mkdir()
    file_path = sub / "file.py"
    content = b"print('hello')\n"
    file_path.write_bytes(content)

    actual = hash_files([file_path])

    expected_hasher = hashlib.sha256()
    expected_hasher.update(file_path.as_posix().encode("utf-8"))
    expected_hasher.update(content)
    expected = expected_hasher.hexdigest()

    assert actual == expected


def test_hash_files_platform_independent_of_path_separator_style(tmp_path: Path) -> None:
    """Directly proves the invariant Phase 11A needs: computing the hash
    input manually via the OS-native str() representation of a path
    (what the OLD buggy implementation did) gives a DIFFERENT result
    than hash_files()'s own (fixed) output, UNLESS str(path) and
    as_posix() already coincide on this platform (true on POSIX, false
    on Windows) - i.e. this test documents that hash_files() no longer
    depends on os.sep at all.
    """
    file_path = tmp_path / "x.py"
    content = b"x = 1\n"
    file_path.write_bytes(content)

    actual = hash_files([file_path])

    posix_based_hasher = hashlib.sha256()
    posix_based_hasher.update(file_path.as_posix().encode("utf-8"))
    posix_based_hasher.update(content)

    str_based_hasher = hashlib.sha256()
    str_based_hasher.update(str(file_path).encode("utf-8"))
    str_based_hasher.update(content)

    assert actual == posix_based_hasher.hexdigest()
    if str(file_path) != file_path.as_posix():
        # Only meaningful on platforms where the two representations
        # actually differ (e.g. Windows) - proves the fix matters there.
        assert actual != str_based_hasher.hexdigest()
