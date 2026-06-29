"""Tests for final report generation: required sections + final-context reflection."""

from pathlib import Path

from minirag import report
from minirag.schemas import (
    AuditResult,
    Chunk,
    DraftAnswer,
    Policy,
    Query,
    QueryRetrieval,
    RetrievedChunk,
    ReviewOverride,
    RevisedAnswer,
)

REQUIRED_SECTIONS = [
    "Retrieval Summary",
    "Query-by-Query Results",
    "Reviewed Overrides",
    "Audit Findings",
    "Failure Modes Observed",
    "Recommended Improvements",
]


def _fixture():
    chunks = [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=5, text="alpha"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=5, text="gamma"),
    ]
    queries = [
        Query(query_id="Q1", question="q1?"),
        Query(query_id="Q2", question="q2?"),
    ]
    retrievals = [
        QueryRetrieval(query_id="Q1", question="q1?", retrieved_chunks=[
            RetrievedChunk(chunk_id="a-0000", document_name="a.txt", rank=1, retrieval_score=1.0)]),
        QueryRetrieval(query_id="Q2", question="q2?", retrieved_chunks=[
            RetrievedChunk(chunk_id="a-0000", document_name="a.txt", rank=1, retrieval_score=1.0)]),
    ]
    drafts = [
        DraftAnswer(query_id="Q1", answer="grounded", label="supported", citations=["a-0000"], reasoning_summary="r"),
        DraftAnswer(query_id="Q2", answer="weak", label="unsupported", citations=[], reasoning_summary="r"),
    ]
    overrides = [
        ReviewOverride(query_id="Q1", overridden=False, final_context_chunk_ids=["a-0000"], note=""),
        ReviewOverride(query_id="Q2", overridden=True, final_context_chunk_ids=["b-0000"], note="forced gamma"),
    ]
    audits = [
        AuditResult(query_id="Q1", audit_label="pass", support_assessment="ok", citation_check="ok", hallucination_risk="low", recommended_fix="none"),
        AuditResult(query_id="Q2", audit_label="fail", support_assessment="bad", citation_check="missing", hallucination_risk="high", recommended_fix="be conservative"),
    ]
    return chunks, queries, retrievals, drafts, overrides, audits


def test_report_has_all_sections_and_reflects_override(tmp_path: Path):
    chunks, queries, retrievals, drafts, overrides, audits = _fixture()
    out = tmp_path / "final_report.md"
    report.build_report(
        queries=queries, retrievals=retrievals, drafts=drafts, overrides=overrides,
        audits=audits, revised=[], policy=Policy(),
        index_metadata={"retrieval_mode": "keyword", "num_documents": 2, "num_chunks": 2},
        metrics=None, error_analysis=[], out_path=out,
    )
    text = out.read_text()
    for sec in REQUIRED_SECTIONS:
        assert sec in text
    # The overridden final context (b-0000) must be reflected.
    assert "b-0000" in text
    assert "forced gamma" in text
    # Grounded vs unsupported distinction present.
    assert "GROUNDED" in text
    assert "UNSUPPORTED" in text


def test_report_uses_revised_recommendation(tmp_path: Path):
    chunks, queries, retrievals, drafts, overrides, audits = _fixture()
    revised = [RevisedAnswer(query_id="Q2", answer="more careful", label="unsupported",
                             citations=[], reasoning_summary="r",
                             reason_for_revision="audit_label=fail, hallucination_risk=high")]
    out = tmp_path / "final_report.md"
    report.build_report(
        queries=queries, retrievals=retrievals, drafts=drafts, overrides=overrides,
        audits=audits, revised=revised, policy=Policy(),
        index_metadata={"retrieval_mode": "keyword", "num_documents": 2, "num_chunks": 2},
        metrics=None, error_analysis=[], out_path=out,
    )
    text = out.read_text()
    assert "Use revised answer" in text
    assert "more careful" in text


def test_report_includes_metrics_when_present(tmp_path: Path):
    chunks, queries, retrievals, drafts, overrides, audits = _fixture()
    out = tmp_path / "final_report.md"
    report.build_report(
        queries=queries, retrievals=retrievals, drafts=drafts, overrides=overrides,
        audits=audits, revised=[], policy=Policy(),
        index_metadata={"retrieval_mode": "keyword", "num_documents": 2, "num_chunks": 2},
        metrics={"k": 3, "num_annotated_queries": 2, "mean_recall_at_k": 0.5, "hit_rate_at_k": 1.0},
        error_analysis=[{"query_id": "Q2", "failure_type": "corpus_gap", "description": "no data"}],
        out_path=out,
    )
    text = out.read_text()
    assert "mean recall@3=0.5" in text
    assert "corpus_gap" in text
