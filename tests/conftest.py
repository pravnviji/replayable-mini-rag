"""Make the ``src`` package and repo root importable in tests."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (SRC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# A JSON payload that is a superset of every structured LLM schema
# (DraftAnswerLLM, AuditResultLLM, RevisedAnswerLLM). Pydantic ignores extra
# fields, so this single response validates against any of them.
_SUPERSET_RESPONSE = {
    "answer": "Event data is retained for 13 months on the standard plan.",
    "label": "supported",
    "citations": ["product_overview-0000"],
    "reasoning_summary": "Grounded in the retained-for-13-months sentence.",
    "audit_label": "pass",
    "support_assessment": "supported",
    "citation_check": "citations present and relevant",
    "hallucination_risk": "low",
    "recommended_fix": "none",
}


class FakeOllamaClient:
    """Stand-in for the ollama client/module used by ``minirag.llm``.

    ``response_builder`` may be supplied to customise the returned JSON content
    per call (receives the messages list); otherwise the superset payload is
    returned so any schema validates.
    """

    def __init__(self, response_builder=None, embed_dim: int = 8):
        self.response_builder = response_builder
        self.embed_dim = embed_dim
        self.chat_calls = []
        self.embed_calls = []

    def chat(self, *, model, messages, format=None, options=None):
        self.chat_calls.append({"model": model, "messages": messages})
        if self.response_builder is not None:
            payload = self.response_builder(messages)
        else:
            payload = dict(_SUPERSET_RESPONSE)
        return {"message": {"content": json.dumps(payload)}}

    def embeddings(self, *, model, prompt):
        self.embed_calls.append({"model": model, "prompt": prompt})
        # Deterministic pseudo-embedding derived from the text.
        vec = [0.0] * self.embed_dim
        for i, ch in enumerate(prompt):
            vec[i % self.embed_dim] += (ord(ch) % 17) / 17.0
        return {"embedding": vec}


@pytest.fixture
def fake_ollama(monkeypatch):
    """Patch ``minirag.llm._client`` to return a FakeOllamaClient instance."""
    import minirag.llm as llm

    client = FakeOllamaClient()
    monkeypatch.setattr(llm, "_client", lambda host=None: client)
    return client
