"""Tests for the Ollama wrapper: structured parse, logging, embeddings, errors."""

from pathlib import Path

import pytest

import minirag.llm as llm
from minirag.io_utils import read_jsonl
from minirag.schemas import DraftAnswerLLM


def test_call_structured_parses_and_logs(tmp_path: Path, fake_ollama):
    log = tmp_path / "llm.jsonl"
    result = llm.call_structured(
        stage="stage1_draft",
        query_id="Q1",
        system_prompt="sys",
        user_prompt="user",
        schema=DraftAnswerLLM,
        model="test-model",
        log_path=log,
        input_artifacts=["a.json"],
        output_artifact="out.json",
    )
    assert isinstance(result, DraftAnswerLLM)
    assert result.label == "supported"

    records = read_jsonl(log)
    assert len(records) == 1
    rec = records[0]
    assert rec["stage"] == "stage1_draft"
    assert rec["query_id"] == "Q1"
    assert rec["provider"] == "ollama"
    assert rec["model"] == "test-model"
    assert rec["input_artifacts"] == ["a.json"]
    assert rec["output_artifact"] == "out.json"
    assert len(rec["prompt_hash"]) == 64  # sha256 hex


def test_call_structured_logs_before_failure(tmp_path: Path, monkeypatch):
    log = tmp_path / "llm.jsonl"

    class BadClient:
        def chat(self, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(llm, "_client", lambda host=None: BadClient())
    with pytest.raises(llm.LLMError):
        llm.call_structured(
            stage="stage1_draft",
            query_id="Q1",
            system_prompt="s",
            user_prompt="u",
            schema=DraftAnswerLLM,
            model="m",
            log_path=log,
            input_artifacts=[],
            output_artifact="o.json",
        )
    # The call was logged before the failure (audit trail preserved).
    assert len(read_jsonl(log)) == 1


def test_call_structured_invalid_json_raises(tmp_path: Path, monkeypatch):
    class JunkClient:
        def chat(self, **kwargs):
            return {"message": {"content": "not json"}}

    monkeypatch.setattr(llm, "_client", lambda host=None: JunkClient())
    with pytest.raises(llm.LLMError):
        llm.call_structured(
            stage="stage1_draft",
            query_id="Q1",
            system_prompt="s",
            user_prompt="u",
            schema=DraftAnswerLLM,
            model="m",
            log_path=tmp_path / "llm.jsonl",
            input_artifacts=[],
            output_artifact="o.json",
        )


def test_embed_texts_returns_vectors(fake_ollama):
    vectors = llm.embed_texts(["hello", "world"], model="nomic")
    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


def test_default_model_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom:tag")
    assert llm.default_model() == "custom:tag"
    monkeypatch.setenv("EMBED_MODEL", "embed:tag")
    assert llm.default_embed_model() == "embed:tag"


def test_extract_content_object_form():
    class Msg:
        content = '{"x": 1}'

    class Resp:
        message = Msg()

    assert llm._extract_content(Resp()) == '{"x": 1}'
