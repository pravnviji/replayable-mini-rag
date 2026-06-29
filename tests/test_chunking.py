"""Chunking determinism and correctness."""

from pathlib import Path

from minirag.chunking import build_chunks, chunk_text


def test_chunk_text_windows_and_overlap():
    text = "abcdefghij"  # 10 chars
    windows = chunk_text(text, size=4, overlap=1)
    # step = 3 -> starts at 0, 3, 6; the window at 6 reaches the end so we stop.
    assert [(s, e) for s, e, _ in windows] == [(0, 4), (3, 7), (6, 10)]
    assert windows[0][2] == "abcd"
    assert windows[-1][2] == "ghij"


def test_chunk_text_drops_whitespace_only():
    text = "aa\n\n\n\n\n\n\n\nbb"
    windows = chunk_text(text, size=3, overlap=0)
    assert all(piece.strip() for _, _, piece in windows)


def test_build_chunks_is_deterministic(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "b.txt").write_text("Beta document content here.")
    (docs / "a.txt").write_text("Alpha document content here.")

    first = build_chunks(docs, size=10, overlap=2)
    second = build_chunks(docs, size=10, overlap=2)

    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
    # Sorted by filename: a.txt chunks come before b.txt.
    assert first[0].document_name == "a.txt"
    assert first[0].chunk_id == "a-0000"
    # Each chunk_id is unique.
    assert len({c.chunk_id for c in first}) == len(first)


def test_chunk_offsets_match_source(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    content = "The quick brown fox jumps over the lazy dog."
    (docs / "doc.txt").write_text(content)
    chunks = build_chunks(docs, size=12, overlap=3)
    for c in chunks:
        assert content[c.start_char:c.end_char] == c.text
