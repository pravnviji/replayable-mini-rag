"""Tests for deterministic retrieval metrics (recall@k / hit@k)."""

from pathlib import Path

from minirag import metrics
from minirag.schemas import Chunk, Query, QueryRetrieval, RetrievedChunk


def _chunks():
    return [
        Chunk(chunk_id="c0", document_name="a.txt", start_char=0, end_char=3, text="aaa"),
        Chunk(chunk_id="c1", document_name="a.txt", start_char=3, end_char=6, text="bbb"),
        Chunk(chunk_id="c2", document_name="b.txt", start_char=0, end_char=3, text="ccc"),
    ]


def _retrieval(query_id, chunk_ids):
    return QueryRetrieval(
        query_id=query_id,
        question="q",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=cid,
                document_name="a.txt" if cid != "c2" else "b.txt",
                rank=i + 1,
                retrieval_score=1.0 - i * 0.1,
            )
            for i, cid in enumerate(chunk_ids)
        ],
    )


def test_has_annotations():
    assert metrics.has_annotations([Query(query_id="Q1", question="?")]) is False
    assert metrics.has_annotations(
        [Query(query_id="Q1", question="?", expected_chunk_ids=["c0"])]
    ) is True


def test_compute_metrics_returns_none_without_annotations(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?")]
    out = tmp_path / "metrics.json"
    result = metrics.compute_metrics(
        queries, [_retrieval("Q1", ["c0"])], _chunks(), top_k=2, out_path=out
    )
    assert result is None
    assert not out.exists()  # nothing written when skipped


def test_chunk_level_recall_and_hit(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?", expected_chunk_ids=["c0", "c2"])]
    out = tmp_path / "metrics.json"
    result = metrics.compute_metrics(
        queries, [_retrieval("Q1", ["c0", "c1"])], _chunks(), top_k=2, out_path=out
    )
    assert result["num_annotated_queries"] == 1
    assert result["mean_recall_at_k"] == 0.5  # found c0 of {c0, c2}
    assert result["hit_rate_at_k"] == 1.0
    assert out.is_file()


def test_document_level_recall(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?", expected_documents=["b.txt"])]
    result = metrics.compute_metrics(
        queries, [_retrieval("Q1", ["c2"])], _chunks(),
        top_k=2, out_path=tmp_path / "m.json",
    )
    assert result["mean_recall_at_k"] == 1.0
    assert result["hit_rate_at_k"] == 1.0


def test_unannotated_query_marked_but_not_counted(tmp_path: Path):
    queries = [
        Query(query_id="Q1", question="?", expected_chunk_ids=["c0"]),
        Query(query_id="Q2", question="?"),  # no annotations
    ]
    result = metrics.compute_metrics(
        queries,
        [_retrieval("Q1", ["c0"]), _retrieval("Q2", ["c1"])],
        _chunks(), top_k=2, out_path=tmp_path / "m.json",
    )
    assert result["num_annotated_queries"] == 1
    by_id = {pq["query_id"]: pq for pq in result["per_query"]}
    assert by_id["Q1"]["annotated"] is True
    assert by_id["Q2"]["annotated"] is False
    assert by_id["Q2"]["recall_at_k"] is None


def test_missing_retrieval_yields_zero_recall(tmp_path: Path):
    queries = [Query(query_id="Q1", question="?", expected_chunk_ids=["c0"])]
    result = metrics.compute_metrics(
        queries, [], _chunks(), top_k=2, out_path=tmp_path / "m.json"
    )
    assert result["mean_recall_at_k"] == 0.0
    assert result["hit_rate_at_k"] == 0.0
