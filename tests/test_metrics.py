"""Tests for retrieval metrics (recall@k / hit@k), including the skip path."""

from pathlib import Path

from minirag import metrics
from minirag.schemas import Chunk, Query, QueryRetrieval, RetrievedChunk


def _chunks():
    return [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=3, text="x"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=3, text="y"),
    ]


def _retrievals():
    return [
        QueryRetrieval(
            query_id="Q1", question="?",
            retrieved_chunks=[
                RetrievedChunk(chunk_id="a-0000", document_name="a.txt", rank=1, retrieval_score=1.0),
            ],
        )
    ]


def test_skips_without_annotations(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?")]
    result = metrics.compute_metrics(
        queries, _retrievals(), _chunks(), top_k=1, out_path=tmp_path / "m.json"
    )
    assert result is None
    assert not (tmp_path / "m.json").exists()


def test_has_annotations_detection():
    assert metrics.has_annotations([Query(query_id="Q1", question="?", expected_chunk_ids=["c"])])
    assert not metrics.has_annotations([Query(query_id="Q1", question="?")])


def test_recall_and_hit_with_chunk_annotations(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?", expected_chunk_ids=["a-0000", "b-0000"])]
    out = tmp_path / "m.json"
    result = metrics.compute_metrics(queries, _retrievals(), _chunks(), top_k=1, out_path=out)
    assert result is not None
    assert result["num_annotated_queries"] == 1
    # 1 of 2 expected chunks retrieved -> recall 0.5, hit 1.
    assert result["mean_recall_at_k"] == 0.5
    assert result["hit_rate_at_k"] == 1.0
    assert out.exists()


def test_document_level_annotations(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?", expected_documents=["a.txt"])]
    result = metrics.compute_metrics(
        queries, _retrievals(), _chunks(), top_k=1, out_path=tmp_path / "m.json"
    )
    assert result["per_query"][0]["hit_at_k"] == 1
    assert result["per_query"][0]["recall_at_k"] == 1.0


def test_miss_when_expected_not_retrieved(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?", expected_chunk_ids=["b-0000"])]
    result = metrics.compute_metrics(
        queries, _retrievals(), _chunks(), top_k=1, out_path=tmp_path / "m.json"
    )
    assert result["hit_rate_at_k"] == 0.0
