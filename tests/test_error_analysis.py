"""Tests for retrieval error-analysis classification."""

from pathlib import Path

from minirag import error_analysis
from minirag.schemas import AuditResult, Chunk, DraftAnswer, Query, QueryRetrieval, RetrievedChunk


def _chunk(cid, text, doc="d.txt"):
    return Chunk(chunk_id=cid, document_name=doc, start_char=0, end_char=len(text), text=text)


def _retrieval(query_id, chunk_ids):
    return QueryRetrieval(
        query_id=query_id,
        question="",
        retrieved_chunks=[
            RetrievedChunk(chunk_id=cid, document_name="d.txt", rank=i + 1, retrieval_score=1.0)
            for i, cid in enumerate(chunk_ids)
        ],
    )


def _audit(query_id, *, label="fail", risk="high"):
    return AuditResult(
        query_id=query_id,
        audit_label=label,
        support_assessment="assessment",
        citation_check="check",
        hallucination_risk=risk,
        recommended_fix="fix",
    )


def _draft(query_id, label="unsupported"):
    return DraftAnswer(
        query_id=query_id, answer="a", label=label, citations=[], reasoning_summary="r"
    )


def _run(queries, retrievals, drafts, audits, chunks, tmp_path):
    return error_analysis.analyse(
        queries, retrievals, drafts, audits, chunks,
        out_path=tmp_path / "errors.json",
    )


def test_strong_answers_are_not_flagged(tmp_path: Path):
    chunks = [_chunk("c0", "billing invoice refund process details")]
    q = Query(query_id="Q1", question="billing invoice refund process")
    findings = _run(
        [q], [_retrieval("Q1", ["c0"])], [_draft("Q1", "supported")],
        [_audit("Q1", label="pass", risk="low")], chunks, tmp_path,
    )
    assert findings == []
    assert not (tmp_path / "errors.json").exists()


def test_corpus_gap_when_terms_absent(tmp_path: Path):
    chunks = [_chunk("c0", "unrelated content about something else entirely")]
    q = Query(query_id="Q1", question="kubernetes helm yaml manifests")
    findings = _run(
        [q], [_retrieval("Q1", ["c0"])], [_draft("Q1")], [_audit("Q1")], chunks, tmp_path
    )
    assert findings[0]["failure_type"] == "corpus_gap"


def test_ranking_failure_when_overlap_present_but_fail(tmp_path: Path):
    chunks = [_chunk("c0", "billing invoice refund process workflow steps")]
    q = Query(query_id="Q1", question="billing invoice refund process")
    findings = _run(
        [q], [_retrieval("Q1", ["c0"])], [_draft("Q1", "partially_supported")],
        [_audit("Q1", label="fail", risk="medium")], chunks, tmp_path,
    )
    assert findings[0]["failure_type"] == "ranking"


def test_chunking_failure_when_terms_split_away(tmp_path: Path):
    # Terms exist in the corpus, but the retrieved chunk does not contain them.
    relevant = _chunk("c0", "quantum entanglement physics theory")
    retrieved = _chunk("c1", "completely different unrelated material here")
    q = Query(query_id="Q1", question="quantum entanglement physics")
    findings = _run(
        [q], [_retrieval("Q1", ["c1"])], [_draft("Q1")],
        [_audit("Q1", label="fail", risk="medium")], [relevant, retrieved], tmp_path,
    )
    assert findings[0]["failure_type"] == "chunking"


def test_ambiguity_for_partial_support(tmp_path: Path):
    chunks = [_chunk("c0", "billing invoice refund process workflow")]
    q = Query(query_id="Q1", question="billing invoice refund process")
    findings = _run(
        [q], [_retrieval("Q1", ["c0"])], [_draft("Q1", "partially_supported")],
        [_audit("Q1", label="pass", risk="medium")], chunks, tmp_path,
    )
    assert findings[0]["failure_type"] == "ambiguity"


def test_findings_written_to_disk(tmp_path: Path):
    chunks = [_chunk("c0", "unrelated content")]
    q = Query(query_id="Q1", question="kubernetes helm yaml")
    _run([q], [_retrieval("Q1", ["c0"])], [_draft("Q1")], [_audit("Q1")], chunks, tmp_path)
    assert (tmp_path / "errors.json").is_file()
