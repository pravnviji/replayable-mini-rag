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


def _fake_input(responses):
    it = iter(responses)

    def _inner(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            return ""

    return _inner


def test_interactive_override_applied(tmp_path: Path, monkeypatch):
    chunks, retrievals, drafts = _fixture()
    # Force interactive path: pretend we have a TTY and feed input lines.
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        _fake_input(["Q1 b-0000", ""]),  # override Q1 then continue
    )
    overrides = review.run_review(
        retrievals, drafts, chunks, out_path=tmp_path / "ov.json"
    )
    ctx = review.final_context_map(overrides)
    assert ctx["Q1"] == ["b-0000"]
    q1 = next(o for o in overrides if o.query_id == "Q1")
    assert q1.overridden is True


def test_interactive_handles_invalid_then_valid(tmp_path: Path, monkeypatch, capsys):
    chunks, retrievals, drafts = _fixture()
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        _fake_input([
            "garbage",            # invalid format
            "ZZ a-0000",          # unknown query id
            "Q2 nope-9999",       # unknown chunk id
            "Q2 a-0000",          # valid override
            "",                   # continue
        ]),
    )
    overrides = review.run_review(
        retrievals, drafts, chunks, out_path=tmp_path / "ov.json"
    )
    out = capsys.readouterr().out
    assert "Invalid format" in out
    assert "Unknown query_id" in out
    ctx = review.final_context_map(overrides)
    assert ctx["Q2"] == ["a-0000"]


def test_non_tty_continues_without_override(tmp_path: Path, monkeypatch):
    chunks, retrievals, drafts = _fixture()
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: False)
    overrides = review.run_review(
        retrievals, drafts, chunks, out_path=tmp_path / "ov.json"
    )
    assert all(o.overridden is False for o in overrides)
