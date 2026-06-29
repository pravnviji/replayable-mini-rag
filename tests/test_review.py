"""Human-review override loading, validation, and final-context application."""

import json
from pathlib import Path

import pytest

import minirag.review as review
from minirag.schemas import Chunk, DraftAnswer, QueryRetrieval, RetrievedChunk


def _fixture():
    chunks = [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=3, text="alpha"),
        Chunk(chunk_id="a-0001", document_name="a.txt", start_char=3, end_char=6, text="beta"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=3, text="gamma"),
    ]
    retrievals = [
        QueryRetrieval(
            query_id="Q1", question="q1",
            retrieved_chunks=[
                RetrievedChunk(chunk_id="a-0000", document_name="a.txt", rank=1, retrieval_score=1.0),
            ],
        ),
        QueryRetrieval(
            query_id="Q2", question="q2",
            retrieved_chunks=[
                RetrievedChunk(chunk_id="a-0001", document_name="a.txt", rank=1, retrieval_score=1.0),
            ],
        ),
    ]
    drafts = [
        DraftAnswer(query_id="Q1", answer="x", label="supported", citations=["a-0000"], reasoning_summary="r"),
        DraftAnswer(query_id="Q2", answer="y", label="unsupported", citations=[], reasoning_summary="r"),
    ]
    return chunks, retrievals, drafts


def test_auto_continue_uses_original_context(tmp_path: Path):
    chunks, retrievals, drafts = _fixture()
    overrides = review.run_review(
        retrievals, drafts, chunks,
        out_path=tmp_path / "ov.json", auto_continue=True,
    )
    ctx = review.final_context_map(overrides)
    assert ctx == {"Q1": ["a-0000"], "Q2": ["a-0001"]}
    assert all(o.overridden is False for o in overrides)


def test_review_input_file_applies_override(tmp_path: Path):
    chunks, retrievals, drafts = _fixture()
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps({"overrides": [
        {"query_id": "Q2", "chunk_ids": ["b-0000"], "note": "force gamma"}
    ]}))

    overrides = review.run_review(
        retrievals, drafts, chunks,
        out_path=tmp_path / "ov.json", review_input=review_file,
    )
    ctx = review.final_context_map(overrides)
    assert ctx["Q2"] == ["b-0000"]  # overridden, not original a-0001
    q2 = next(o for o in overrides if o.query_id == "Q2")
    assert q2.overridden is True


def test_review_input_rejects_unknown_chunk(tmp_path: Path):
    chunks, retrievals, drafts = _fixture()
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps({"Q1": ["does-not-exist"]}))
    with pytest.raises(ValueError):
        review.run_review(
            retrievals, drafts, chunks,
            out_path=tmp_path / "ov.json", review_input=review_file,
        )
