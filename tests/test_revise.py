"""Tests for conservative re-generation after audit failure / high risk."""

from pathlib import Path

import minirag.revise as revise
from minirag.schemas import (
    AuditResult,
    Chunk,
    DraftAnswer,
    Policy,
    RevisedAnswerLLM,
)


def _chunks():
    return [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=5, text="alpha"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=5, text="gamma"),
    ]


def test_needs_revision_logic():
    assert revise.needs_revision(AuditResult(query_id="Q1", audit_label="fail", support_assessment="", citation_check="", hallucination_risk="low", recommended_fix=""))
    assert revise.needs_revision(AuditResult(query_id="Q2", audit_label="pass", support_assessment="", citation_check="", hallucination_risk="high", recommended_fix=""))
    assert not revise.needs_revision(AuditResult(query_id="Q3", audit_label="pass", support_assessment="", citation_check="", hallucination_risk="low", recommended_fix=""))


def test_only_failing_queries_revised_and_citations_filtered(tmp_path: Path, monkeypatch):
    drafts = [
        DraftAnswer(query_id="Q1", answer="x", label="supported", citations=["a-0000"], reasoning_summary="r"),
        DraftAnswer(query_id="Q2", answer="y", label="supported", citations=["b-0000"], reasoning_summary="r"),
    ]
    audits = [
        AuditResult(query_id="Q1", audit_label="fail", support_assessment="", citation_check="", hallucination_risk="high", recommended_fix="fix"),
        AuditResult(query_id="Q2", audit_label="pass", support_assessment="", citation_check="", hallucination_risk="low", recommended_fix="none"),
    ]

    def fake_call(**kwargs):
        # Cites one in-context and one out-of-context chunk.
        return RevisedAnswerLLM(
            answer="revised", label="unsupported",
            citations=["a-0000", "out-of-scope"], reasoning_summary="r",
        )

    monkeypatch.setattr(revise, "call_structured", fake_call)

    revised = revise.revise_answers(
        drafts, audits, _chunks(), Policy(),
        {"Q1": ["a-0000"], "Q2": ["b-0000"]},
        {"Q1": "?", "Q2": "?"},
        model="m", out_path=tmp_path / "revised.json",
        llm_log_path=tmp_path / "llm.jsonl", input_artifacts=[],
    )

    assert [r.query_id for r in revised] == ["Q1"]  # only the failing query
    assert revised[0].citations == ["a-0000"]       # out-of-scope dropped
    assert "high" in revised[0].reason_for_revision
    assert (tmp_path / "revised.json").exists()


def test_no_revision_writes_nothing(tmp_path: Path, monkeypatch):
    drafts = [DraftAnswer(query_id="Q1", answer="x", label="supported", citations=[], reasoning_summary="r")]
    audits = [AuditResult(query_id="Q1", audit_label="pass", support_assessment="", citation_check="", hallucination_risk="low", recommended_fix="none")]
    monkeypatch.setattr(revise, "call_structured", lambda **k: None)

    revised = revise.revise_answers(
        drafts, audits, _chunks(), Policy(), {"Q1": ["a-0000"]}, {"Q1": "?"},
        model="m", out_path=tmp_path / "revised.json",
        llm_log_path=tmp_path / "llm.jsonl", input_artifacts=[],
    )
    assert revised == []
    assert not (tmp_path / "revised.json").exists()
