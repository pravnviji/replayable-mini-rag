"""Tests for pydantic schema defaults, validation, and parsing."""

import pytest
from pydantic import ValidationError

from minirag.schemas import (
    AuditResultLLM,
    DraftAnswerLLM,
    Policy,
    Query,
    QuerySet,
    RetrievalPolicy,
)


def test_policy_defaults():
    p = Policy()
    assert p.retrieval.mode == "keyword"
    assert "supported" in p.allowed_labels
    assert p.citation_required is True


def test_policy_from_dict():
    p = Policy.model_validate(
        {
            "retrieval": {"mode": "embedding", "top_k": 5},
            "allowed_labels": ["yes", "no"],
            "citation_required": False,
            "forbidden_behaviours": ["x"],
            "generation": {"provider": "ollama", "model": "m"},
        }
    )
    assert p.retrieval.mode == "embedding"
    assert p.retrieval.top_k == 5
    assert p.allowed_labels == ["yes", "no"]


def test_retrieval_policy_rejects_bad_top_k():
    with pytest.raises(ValidationError):
        RetrievalPolicy(top_k=0)


def test_query_optional_annotations():
    q = Query(query_id="Q1", question="?")
    assert q.expected_chunk_ids is None
    q2 = Query(query_id="Q2", question="?", expected_chunk_ids=["c1"])
    assert q2.expected_chunk_ids == ["c1"]


def test_queryset_parsing():
    qs = QuerySet.model_validate({"queries": [{"query_id": "Q1", "question": "?"}]})
    assert len(qs.queries) == 1


def test_audit_label_literal_enforced():
    with pytest.raises(ValidationError):
        AuditResultLLM(
            audit_label="maybe",
            support_assessment="s",
            citation_check="c",
            hallucination_risk="low",
            recommended_fix="none",
        )


def test_draft_answer_llm_parses_json():
    obj = DraftAnswerLLM.model_validate_json(
        '{"answer":"a","label":"supported","citations":["c1"],"reasoning_summary":"r"}'
    )
    assert obj.citations == ["c1"]
