"""Pydantic models for inputs, artifacts, and structured LLM outputs.

The LLM-output models (``DraftAnswerLLM``, ``AuditResultLLM``,
``RevisedAnswerLLM``) are passed to Ollama via ``model_json_schema()`` so the
model is constrained to emit schema-conformant JSON.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class RetrievalPolicy(BaseModel):
    mode: Literal["keyword", "embedding"] = "keyword"
    top_k: int = Field(default=3, ge=1)
    chunk_size_chars: int = Field(default=480, ge=1)
    chunk_overlap_chars: int = Field(default=80, ge=0)


class GenerationPolicy(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1:8b"


class Policy(BaseModel):
    retrieval: RetrievalPolicy = Field(default_factory=RetrievalPolicy)
    allowed_labels: list[str] = Field(
        default_factory=lambda: ["supported", "partially_supported", "unsupported"]
    )
    citation_required: bool = True
    forbidden_behaviours: list[str] = Field(default_factory=list)
    generation: GenerationPolicy = Field(default_factory=GenerationPolicy)


class Query(BaseModel):
    query_id: str
    question: str
    # Optional expected-evidence annotations (enable retrieval metrics).
    expected_chunk_ids: Optional[list[str]] = None
    expected_documents: Optional[list[str]] = None


class QuerySet(BaseModel):
    queries: list[Query]


# --------------------------------------------------------------------------- #
# Chunking / retrieval artifacts
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    chunk_id: str
    document_name: str
    start_char: int
    end_char: int
    text: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_name: str
    rank: int
    retrieval_score: float


class QueryRetrieval(BaseModel):
    query_id: str
    question: str
    retrieved_chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------- #
# Stage 1: draft answers
# --------------------------------------------------------------------------- #
class DraftAnswerLLM(BaseModel):
    """Schema the LLM must conform to for a Stage 1 draft answer."""

    answer: str
    label: str
    citations: list[str]
    reasoning_summary: str


class DraftAnswer(BaseModel):
    query_id: str
    answer: str
    label: str
    citations: list[str]
    reasoning_summary: str


# --------------------------------------------------------------------------- #
# Human review overrides
# --------------------------------------------------------------------------- #
class ReviewOverride(BaseModel):
    query_id: str
    overridden: bool
    final_context_chunk_ids: list[str]
    note: str = ""


# --------------------------------------------------------------------------- #
# Stage 2: audit
# --------------------------------------------------------------------------- #
class AuditResultLLM(BaseModel):
    """Schema the LLM must conform to for a Stage 2 audit."""

    audit_label: Literal["pass", "fail"]
    support_assessment: str
    citation_check: str
    hallucination_risk: Literal["low", "medium", "high"]
    recommended_fix: str


class AuditResult(BaseModel):
    query_id: str
    audit_label: Literal["pass", "fail"]
    support_assessment: str
    citation_check: str
    hallucination_risk: Literal["low", "medium", "high"]
    recommended_fix: str


# --------------------------------------------------------------------------- #
# Item 8: revised answers
# --------------------------------------------------------------------------- #
class RevisedAnswerLLM(BaseModel):
    answer: str
    label: str
    citations: list[str]
    reasoning_summary: str


class RevisedAnswer(BaseModel):
    query_id: str
    answer: str
    label: str
    citations: list[str]
    reasoning_summary: str
    reason_for_revision: str
