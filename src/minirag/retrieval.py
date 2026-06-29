"""Reproducible retrieval: in-code BM25 (default) and Ollama embeddings.

Both modes are reproducible for the same inputs and config. Scores are rounded
to a fixed precision and ties are broken deterministically by ``chunk_id`` so
the ranked output never depends on dict/iteration order.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from .schemas import Chunk, Query, QueryRetrieval, RetrievedChunk

_SCORE_PRECISION = 6

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer (deterministic)."""
    return _TOKEN_RE.findall(text.lower())


class Retriever(Protocol):
    def search(self, query_text: str, top_k: int) -> list[tuple[Chunk, float]]:
        ...


# --------------------------------------------------------------------------- #
# BM25 (keyword) retriever
# --------------------------------------------------------------------------- #
class BM25Retriever:
    """A small, transparent BM25 implementation over the chunk corpus."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [tokenize(c.text) for c in chunks]
        self.doc_len: list[int] = [len(t) for t in self.doc_tokens]
        self.N = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        # Document frequency per term.
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # Precompute term frequencies per document.
        self.tf: list[dict[str, int]] = []
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for term in tokens:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

        # Precompute IDF once per term. Previously this (and its ``math.log``)
        # was recomputed for every (document, query-term) pair at score time,
        # i.e. O(N_docs * query_terms) redundant log() calls per query.
        self.idf: dict[str, float] = {
            term: self._compute_idf(df) for term, df in self.df.items()
        }

    def _compute_idf(self, n_qi: int) -> float:
        # BM25 idf with +1 smoothing to keep values non-negative.
        return math.log(1 + (self.N - n_qi + 0.5) / (n_qi + 0.5))

    def _idf(self, term: str) -> float:
        return self.idf.get(term, self._compute_idf(self.df.get(term, 0)))

    def score(self, query_terms: list[str], doc_index: int) -> float:
        if self.avgdl == 0:
            return 0.0
        tf = self.tf[doc_index]
        dl = self.doc_len[doc_index]
        len_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        s = 0.0
        for term in query_terms:
            freq = tf.get(term)
            if not freq:
                continue
            idf = self.idf[term]  # term is in tf => guaranteed present
            denom = freq + len_norm
            s += idf * (freq * (self.k1 + 1)) / denom
        return s

    def search(self, query_text: str, top_k: int) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query_text)
        scored = [
            (chunk, round(self.score(query_terms, i), _SCORE_PRECISION))
            for i, chunk in enumerate(self.chunks)
        ]
        # Deterministic: highest score first, then chunk_id ascending.
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# Embedding retriever (Ollama)
# --------------------------------------------------------------------------- #
class EmbeddingRetriever:
    """Cosine-similarity retrieval over Ollama embeddings.

    Embeddings are requested via the Ollama client. Vectors are deterministic
    for a fixed model, so retrieval is reproducible. Imports of ``ollama`` /
    ``numpy`` are local so the keyword path has no hard dependency on them.
    """

    def __init__(self, chunks: list[Chunk], *, embed_model: str, host: str | None = None):
        import numpy as np  # local import

        from .llm import embed_texts  # local import to avoid cycles

        self.chunks = chunks
        self.embed_model = embed_model
        self._np = np
        vectors = embed_texts([c.text for c in chunks], model=embed_model, host=host)
        self.matrix = np.array(vectors, dtype="float64")
        # Pre-normalise for cosine similarity.
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = self.matrix / norms
        self._host = host

    def search(self, query_text: str, top_k: int) -> list[tuple[Chunk, float]]:
        from .llm import embed_texts  # local import

        np = self._np
        qvec = np.array(
            embed_texts([query_text], model=self.embed_model, host=self._host)[0],
            dtype="float64",
        )
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            qnorm = 1.0
        qvec = qvec / qnorm
        sims = self.matrix @ qvec
        scored = [
            (chunk, round(float(sims[i]), _SCORE_PRECISION))
            for i, chunk in enumerate(self.chunks)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:top_k]


def build_retriever(
    chunks: list[Chunk],
    *,
    mode: str,
    embed_model: str | None = None,
    host: str | None = None,
) -> Retriever:
    if mode == "keyword":
        return BM25Retriever(chunks)
    if mode == "embedding":
        if not embed_model:
            raise ValueError("embedding mode requires an embed_model")
        return EmbeddingRetriever(chunks, embed_model=embed_model, host=host)
    raise ValueError(f"Unknown retrieval mode: {mode!r}")


def retrieve_all(
    retriever: Retriever,
    queries: list[Query],
    *,
    top_k: int,
) -> list[QueryRetrieval]:
    """Run retrieval for every query, producing per-query ranked results."""
    results: list[QueryRetrieval] = []
    for q in queries:
        hits = retriever.search(q.question, top_k)
        retrieved = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                rank=rank,
                retrieval_score=score,
            )
            for rank, (chunk, score) in enumerate(hits, start=1)
        ]
        results.append(
            QueryRetrieval(
                query_id=q.query_id,
                question=q.question,
                retrieved_chunks=retrieved,
            )
        )
    return results
