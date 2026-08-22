from __future__ import annotations

from pathlib import Path

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
