"""Tests for the Ollama wrapper: structured calls, call logging, and errors.

These exercise ``minirag.llm`` against the ``fake_ollama`` fixture (defined in
conftest.py) so no real Ollama server is required.
"""

import json

import pytest

import minirag.llm as llm
from minirag.io_utils import read_jsonl
from minirag.schemas import DraftAnswerLLM


def _call(llm_log_path, **overrides):
    kwargs = dict(
        stage="stage1_draft",
        query_id="Q1",
        system_prompt="system",
        user_prompt="user",
        schema=DraftAnswerLLM,
        model="test-model",
        log_path=llm_log_path,
        input_artifacts=["a.json", "b.json"],
        output_artifact="draft_answers.json",
    )
    kwargs.update(overrides)
    return llm.call_structured(**kwargs)


def test_call_structured_returns_validated_object(tmp_path, fake_ollama):
    result = _call(tmp_path / "llm_calls.jsonl")
    assert isinstance(result, DraftAnswerLLM)
    assert result.label == "supported"
    assert len(fake_ollama.chat_calls) == 1


def test_call_structured_appends_one_log_record(tmp_path, fake_ollama):
    log = tmp_path / "llm_calls.jsonl"
    _call(log)
    records = read_jsonl(log)
    assert len(records) == 1
    rec = records[0]
    for field in (
        "stage", "query_id", "timestamp", "provider",
        "model", "prompt_hash", "input_artifacts", "output_artifact",
    ):
        assert field in rec, f"missing required log field: {field}"
    assert rec["provider"] == "ollama"
    assert rec["model"] == "test-model"
    assert rec["input_artifacts"] == ["a.json", "b.json"]


def test_prompt_hash_is_deterministic_and_input_sensitive(tmp_path, fake_ollama):
    log = tmp_path / "llm_calls.jsonl"
    _call(log, user_prompt="same")
    _call(log, user_prompt="same")
    _call(log, user_prompt="different")
    hashes = [r["prompt_hash"] for r in read_jsonl(log)]
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_call_is_logged_even_when_parsing_fails(tmp_path, monkeypatch):
    """A record is written before the call, so a failure still leaves a trace."""
    log = tmp_path / "llm_calls.jsonl"

    class BadClient:
        def chat(self, *, model, messages, format=None, options=None):
            return {"message": {"content": "not valid json"}}

    monkeypatch.setattr(llm, "_client", lambda host=None: BadClient())
    with pytest.raises(llm.LLMError):
        _call(log)
    # The pre-send log record must still be present.
    assert len(read_jsonl(log)) == 1


def test_call_raises_llmerror_on_transport_failure(tmp_path, monkeypatch):
    log = tmp_path / "llm_calls.jsonl"

    class BoomClient:
        def chat(self, *, model, messages, format=None, options=None):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(llm, "_client", lambda host=None: BoomClient())
    with pytest.raises(llm.LLMError) as exc:
        _call(log)
    assert "connection refused" in str(exc.value)


def test_embed_texts_returns_vectors(fake_ollama):
    vectors = llm.embed_texts(["alpha", "beta"], model="embed-model")
    assert len(vectors) == 2
    assert all(len(v) == fake_ollama.embed_dim for v in vectors)
    assert len(fake_ollama.embed_calls) == 2


def test_default_model_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom:latest")
    assert llm.default_model() == "custom:latest"
    monkeypatch.delenv("LLM_MODEL", raising=False)


def test_extract_content_supports_object_response():
    class Msg:
        content = json.dumps({"x": 1})

    class Resp:
        message = Msg()

    assert llm._extract_content(Resp()) == '{"x": 1}'
