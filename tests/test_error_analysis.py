"""Tests for retrieval error-analysis classification."""

from pathlib import Path

from minirag import error_analysis
from minirag.schemas import (
    AuditResult,
    Chunk,
    DraftAnswer,
    Query,
    QueryRetrieval,
    RetrievedChunk,
)


def _corpus():
    return [
        Chunk(chunk_id="ov-0000", document_name="ov.txt", start_char=0, end_char=60,
              text="Event data is retained for 13 months on the standard plan."),
        Chunk(chunk_id="sec-0000", document_name="sec.txt", start_char=0, end_char=40,
              text="SCIM provisioning is available on enterprise plans."),
    ]


def test_corpus_gap_for_out_of_scope_question(tmp_path: Path):
    queries = [Query(query_id="Q4", question="Is the service HIPAA compliant for refunds?")]
    retrievals = [QueryRetrieval(query_id="Q4", question="?", retrieved_chunks=[
        RetrievedChunk(chunk_id="ov-0000", document_name="ov.txt", rank=1, retrieval_score=0.0),
    ])]
    drafts = [DraftAnswer(query_id="Q4", answer="x", label="unsupported", citations=[], reasoning_summary="r")]
    audits = [AuditResult(query_id="Q4", audit_label="fail", support_assessment="unsupported", citation_check="missing", hallucination_risk="high", recommended_fix="fix")]

    findings = error_analysis.analyse(
        queries, retrievals, drafts, audits, _corpus(), out_path=tmp_path / "ea.json"
    )
    assert len(findings) == 1
    assert findings[0]["failure_type"] == "corpus_gap"
    assert (tmp_path / "ea.json").exists()


def test_passing_grounded_query_not_flagged(tmp_path: Path):
    queries = [Query(query_id="Q1", question="How long is event data retained on standard plan?")]
    retrievals = [QueryRetrieval(query_id="Q1", question="?", retrieved_chunks=[
        RetrievedChunk(chunk_id="ov-0000", document_name="ov.txt", rank=1, retrieval_score=2.0),
    ])]
    drafts = [DraftAnswer(query_id="Q1", answer="13 months", label="supported", citations=["ov-0000"], reasoning_summary="r")]
    audits = [AuditResult(query_id="Q1", audit_label="pass", support_assessment="supported", citation_check="ok", hallucination_risk="low", recommended_fix="none")]

    findings = error_analysis.analyse(
        queries, retrievals, drafts, audits, _corpus(), out_path=tmp_path / "ea.json"
    )
    assert findings == []
    assert not (tmp_path / "ea.json").exists()


def test_classification_values_are_known():
    valid = {"ranking", "chunking", "ambiguity", "corpus_gap"}
    queries = [Query(query_id="Q2", question="Does the product support SCIM provisioning today?")]
    retrievals = [QueryRetrieval(query_id="Q2", question="?", retrieved_chunks=[
        RetrievedChunk(chunk_id="sec-0000", document_name="sec.txt", rank=1, retrieval_score=1.0),
    ])]
    drafts = [DraftAnswer(query_id="Q2", answer="x", label="partially_supported", citations=[], reasoning_summary="r")]
    audits = [AuditResult(query_id="Q2", audit_label="fail", support_assessment="weak", citation_check="missing", hallucination_risk="medium", recommended_fix="fix")]
    findings = error_analysis.analyse(queries, retrievals, drafts, audits, _corpus(), out_path=Path("/tmp/_ea_ignore.json"))
    assert all(f["failure_type"] in valid for f in findings)
