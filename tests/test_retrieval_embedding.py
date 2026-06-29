"""Tests for the embedding retriever (Ollama embeddings mocked)."""

import minirag.llm as llm
from minirag.retrieval import EmbeddingRetriever, build_retriever
from minirag.schemas import Chunk


def _chunks():
    return [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=10, text="retention months data"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=10, text="scim provisioning enterprise"),
    ]


def _fake_embedder(monkeypatch):
    """Deterministic embedding: vector keyed by presence of marker words."""
    def embed_texts(texts, *, model, host=None):
        vecs = []
        for t in texts:
            tl = t.lower()
            vecs.append([
                1.0 if "retention" in tl or "months" in tl else 0.0,
                1.0 if "scim" in tl or "provisioning" in tl else 0.0,
            ])
        return vecs
    monkeypatch.setattr(llm, "embed_texts", embed_texts)


def test_embedding_retriever_ranks_by_cosine(monkeypatch):
    _fake_embedder(monkeypatch)
    r = EmbeddingRetriever(_chunks(), embed_model="fake")
    hits = r.search("how many months is data retention", top_k=2)
    assert hits[0][0].chunk_id == "a-0000"


def test_build_retriever_embedding_mode(monkeypatch):
    _fake_embedder(monkeypatch)
    r = build_retriever(_chunks(), mode="embedding", embed_model="fake")
    hits = r.search("scim provisioning", top_k=1)
    assert hits[0][0].chunk_id == "b-0000"


def test_build_retriever_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        build_retriever(_chunks(), mode="bogus")


def test_build_retriever_embedding_requires_model():
    import pytest
    with pytest.raises(ValueError):
        build_retriever(_chunks(), mode="embedding", embed_model=None)
