"""Tests for Stage 2 audit: one un-batched call per query on final context."""

from pathlib import Path

import minirag.audit as audit
from minirag.io_utils import read_jsonl
from minirag.schemas import AuditResultLLM, Chunk, DraftAnswer, Policy


def _chunks():
    return [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=5, text="alpha"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=5, text="gamma"),
    ]


def _drafts():
    return [
        DraftAnswer(query_id="Q1", answer="a1", label="supported", citations=["a-0000"], reasoning_summary="r"),
        DraftAnswer(query_id="Q2", answer="a2", label="unsupported", citations=[], reasoning_summary="r"),
    ]


def test_audit_one_call_per_query_and_uses_final_context(tmp_path: Path, monkeypatch):
    seen_prompts = []

    def fake_call(**kwargs):
        seen_prompts.append(kwargs["user_prompt"])
        return AuditResultLLM(
            audit_label="pass",
            support_assessment="ok",
            citation_check="ok",
            hallucination_risk="low",
            recommended_fix="none",
        )

    monkeypatch.setattr(audit, "call_structured", fake_call)

    final_context = {"Q1": ["b-0000"], "Q2": ["a-0000"]}  # overridden contexts
    questions = {"Q1": "q1?", "Q2": "q2?"}

    results = audit.audit_answers(
        _drafts(), _chunks(), Policy(), final_context, questions,
        model="m", out_path=tmp_path / "audit.json",
        llm_log_path=tmp_path / "llm.jsonl",
        input_artifacts=["draft_answers.json"],
    )

    assert len(results) == 2
    assert len(seen_prompts) == 2  # one call per query, not batched
    # Final (overridden) context text must appear in the prompts.
    assert "gamma" in seen_prompts[0]   # Q1 final context = b-0000 (gamma)
    assert "alpha" in seen_prompts[1]   # Q2 final context = a-0000 (alpha)


def test_audit_logs_stage2_records(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        audit, "call_structured",
        lambda **k: AuditResultLLM(
            audit_label="fail", support_assessment="s", citation_check="c",
            hallucination_risk="high", recommended_fix="fix",
        ),
    )
    audit.audit_answers(
        _drafts(), _chunks(), Policy(),
        {"Q1": ["a-0000"], "Q2": ["b-0000"]},
        {"Q1": "?", "Q2": "?"},
        model="m", out_path=tmp_path / "audit.json",
        llm_log_path=tmp_path / "llm.jsonl", input_artifacts=[],
    )
    # call_structured is patched so it doesn't log; verify the audit file written.
    import json
    data = json.load(open(tmp_path / "audit.json"))
    assert {d["query_id"] for d in data} == {"Q1", "Q2"}
    assert all(d["audit_label"] == "fail" for d in data)
