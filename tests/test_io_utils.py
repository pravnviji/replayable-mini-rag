"""Tests for io_utils: atomic writes, JSON/JSONL round-trips, hashing, listing."""

from pathlib import Path

import pytest

from minirag import io_utils


def test_now_iso_format():
    ts = io_utils.now_iso()
    assert ts.endswith("Z")
    assert "T" in ts


def test_write_and_read_json_roundtrip(tmp_path: Path):
    path = tmp_path / "sub" / "data.json"  # parent created automatically
    data = {"b": 2, "a": [1, 2, 3]}
    io_utils.write_json(path, data)
    assert path.is_file()
    assert io_utils.read_json(path) == data


def test_write_json_is_sorted(tmp_path: Path):
    path = tmp_path / "d.json"
    io_utils.write_json(path, {"b": 1, "a": 2})
    text = io_utils.read_text(path)
    assert text.index('"a"') < text.index('"b"')


def test_append_and_read_jsonl(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    io_utils.append_jsonl(path, {"n": 1})
    io_utils.append_jsonl(path, {"n": 2})
    records = io_utils.read_jsonl(path)
    assert records == [{"n": 1}, {"n": 2}]


def test_read_jsonl_missing_returns_empty(tmp_path: Path):
    assert io_utils.read_jsonl(tmp_path / "nope.jsonl") == []


def test_write_and_read_text(tmp_path: Path):
    path = tmp_path / "a.txt"
    io_utils.write_text(path, "hello")
    assert io_utils.read_text(path) == "hello"


def test_list_txt_files_sorted(tmp_path: Path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "b.txt").write_text("b")
    (d / "a.txt").write_text("a")
    (d / "ignore.md").write_text("x")
    files = io_utils.list_txt_files(d)
    assert [f.name for f in files] == ["a.txt", "b.txt"]


def test_list_txt_files_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        io_utils.list_txt_files(tmp_path / "missing")


def test_sha256_text_deterministic():
    assert io_utils.sha256_text("abc") == io_utils.sha256_text("abc")
    assert io_utils.sha256_text("abc") != io_utils.sha256_text("abd")


def test_stable_hash_obj_order_independent():
    assert io_utils.stable_hash_obj({"a": 1, "b": 2}) == io_utils.stable_hash_obj(
        {"b": 2, "a": 1}
    )


def test_atomic_write_leaves_no_tmp(tmp_path: Path):
    path = tmp_path / "x.json"
    io_utils.write_json(path, {"k": "v"})
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_iter_lines():
    assert io_utils.iter_lines(["a", "b", "c"]) == "a\nb\nc"
