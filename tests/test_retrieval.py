"""BM25 retrieval ranking and determinism."""

from minirag.retrieval import BM25Retriever, retrieve_all, tokenize
from minirag.schemas import Chunk, Query


def _chunks() -> list[Chunk]:
    texts = [
        ("c0", "Event data is retained for 13 months on the standard plan."),
        ("c1", "SCIM provisioning is available on enterprise plans."),
        ("c2", "Reports can be exported as CSV or PDF files."),
    ]
    return [
        Chunk(chunk_id=cid, document_name="doc.txt", start_char=0, end_char=len(t), text=t)
        for cid, t in texts
    ]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World! 13months") == ["hello", "world", "13months"]


def test_bm25_ranks_relevant_chunk_first():
    r = BM25Retriever(_chunks())
    hits = r.search("How long is event data retained on the standard plan?", top_k=3)
    assert hits[0][0].chunk_id == "c0"


def test_bm25_scurve_is_deterministic():
    r = BM25Retriever(_chunks())
    a = r.search("SCIM provisioning", top_k=3)
    b = r.search("SCIM provisioning", top_k=3)
    assert [(c.chunk_id, s) for c, s in a] == [(c.chunk_id, s) for c, s in b]
    assert a[0][0].chunk_id == "c1"


def test_tie_break_by_chunk_id():
    # Two identical-content chunks should rank by chunk_id ascending on a tie.
    chunks = [
        Chunk(chunk_id="z", document_name="d.txt", start_char=0, end_char=4, text="same text"),
        Chunk(chunk_id="a", document_name="d.txt", start_char=0, end_char=4, text="same text"),
    ]
    r = BM25Retriever(chunks)
    hits = r.search("same text", top_k=2)
    assert [c.chunk_id for c, _ in hits] == ["a", "z"]


def test_retrieve_all_assigns_sequential_ranks():
    r = BM25Retriever(_chunks())
    queries = [Query(query_id="Q1", question="export CSV PDF reports")]
    results = retrieve_all(r, queries, top_k=2)
    assert results[0].query_id == "Q1"
    ranks = [rc.rank for rc in results[0].retrieved_chunks]
    assert ranks == [1, 2]
