"""Tests for index metadata construction."""

from pathlib import Path

from minirag.indexing import build_index_metadata, write_index_metadata
from minirag.schemas import Chunk, Policy


def _chunks():
    return [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=3, text="aaa"),
        Chunk(chunk_id="a-0001", document_name="a.txt", start_char=3, end_char=6, text="bbb"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=3, text="ccc"),
    ]


def test_metadata_records_mode_and_counts():
    meta = build_index_metadata(_chunks(), Policy(), mode="keyword")
    assert meta["retrieval_mode"] == "keyword"
    assert meta["num_chunks"] == 3
    assert meta["num_documents"] == 2
    assert meta["chunks_per_document"] == {"a.txt": 2, "b.txt": 1}
    assert meta["chunk_ids"] == ["a-0000", "a-0001", "b-0000"]
    assert meta["embed_model"] is None


def test_metadata_embedding_mode_records_model():
    meta = build_index_metadata(_chunks(), Policy(), mode="embedding", embed_model="nomic")
    assert meta["retrieval_mode"] == "embedding"
    assert meta["embed_model"] == "nomic"
    assert meta["config"]["embed_model"] == "nomic"


def test_metadata_config_hash_is_stable():
    a = build_index_metadata(_chunks(), Policy(), mode="keyword")
    b = build_index_metadata(_chunks(), Policy(), mode="keyword")
    assert a["config_hash"] == b["config_hash"]


def test_write_index_metadata(tmp_path: Path):
    meta = build_index_metadata(_chunks(), Policy(), mode="keyword")
    out = tmp_path / "index.json"
    write_index_metadata(meta, out)
    assert out.is_file()
