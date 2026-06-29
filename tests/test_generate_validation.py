"""Citation-subset and label-coercion rules in Stage 1 generation.

These exercise the post-processing logic without invoking the LLM by monkeypatching
``generate.call_structured``.
"""

from pathlib import Path

import minirag.generate as generate
from minirag.schemas import (
    Chunk,
    DraftAnswerLLM,
    Policy,
    QueryRetrieval,
    RetrievedChunk,
)


def _setup():
    chunks = [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=3, text="alpha"),
        Chunk(chunk_id="a-0001", document_name="a.txt", start_char=3, end_char=6, text="beta"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=3, text="gamma"),
    ]
    retrievals = [
        QueryRetrieval(
            query_id="Q1",
            question="q?",
            retrieved_chunks=[
                RetrievedChunk(chunk_id="a-0000", document_name="a.txt", rank=1, retrieval_score=1.0),
                RetrievedChunk(chunk_id="a-0001", document_name="a.txt", rank=2, retrieval_score=0.5),
            ],
        )
    ]
    policy = Policy()
    return chunks, retrievals, policy


def test_citations_filtered_to_retrieved(tmp_path: Path, monkeypatch):
    chunks, retrievals, policy = _setup()

    def fake_call(**kwargs):
        # Cites one retrieved chunk and one out-of-scope chunk.
        return DraftAnswerLLM(
            answer="x",
            label="supported",
            citations=["a-0000", "b-0000"],
            reasoning_summary="r",
        )

    monkeypatch.setattr(generate, "call_structured", fake_call)
    drafts = generate.generate_drafts(
        retrievals, chunks, policy,
        model="m", out_path=tmp_path / "draft.json",
        llm_log_path=tmp_path / "llm.jsonl", input_artifacts=[],
    )
    assert drafts[0].citations == ["a-0000"]  # b-0000 dropped


def test_invalid_label_coerced_to_conservative(tmp_path: Path, monkeypatch):
    chunks, retrievals, policy = _setup()

    def fake_call(**kwargs):
        return DraftAnswerLLM(
            answer="x", label="totally_made_up", citations=[], reasoning_summary="r"
        )

    monkeypatch.setattr(generate, "call_structured", fake_call)
    drafts = generate.generate_drafts(
        retrievals, chunks, policy,
        model="m", out_path=tmp_path / "draft.json",
        llm_log_path=tmp_path / "llm.jsonl", input_artifacts=[],
    )
    assert drafts[0].label == "unsupported"
