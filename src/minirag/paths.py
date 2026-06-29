"""Canonical artifact filenames and path resolution.

Shared by the pipeline and the validator so both agree on where artifacts live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Artifact filenames (inside the output directory).
CHUNKS = "chunks.json"
INDEX_METADATA = "index_metadata.json"
RETRIEVAL_RESULTS = "retrieval_results.json"
DRAFT_ANSWERS = "draft_answers.json"
REVIEW_OVERRIDES = "review_overrides.json"
ANSWER_AUDIT = "answer_audit.json"
FINAL_REPORT = "final_report.md"
RETRIEVAL_METRICS = "retrieval_metrics.json"
REVISED_ANSWERS = "revised_answers.json"
RETRIEVAL_ERROR_ANALYSIS = "retrieval_error_analysis.json"
LLM_CALLS = "llm_calls.jsonl"
PIPELINE_STATE = "pipeline_state.json"

# Always-required artifacts (the optional/stretch ones are conditionally present).
REQUIRED_ARTIFACTS = (
    CHUNKS,
    INDEX_METADATA,
    RETRIEVAL_RESULTS,
    DRAFT_ANSWERS,
    REVIEW_OVERRIDES,
    ANSWER_AUDIT,
    FINAL_REPORT,
    LLM_CALLS,
    PIPELINE_STATE,
)


@dataclass
class Paths:
    """Resolved input and output paths for a single pipeline run."""

    documents: Path
    queries: Path
    policy: Path
    out_dir: Path

    def out(self, name: str) -> Path:
        return self.out_dir / name

    @property
    def chunks(self) -> Path:
        return self.out(CHUNKS)

    @property
    def index_metadata(self) -> Path:
        return self.out(INDEX_METADATA)

    @property
    def retrieval_results(self) -> Path:
        return self.out(RETRIEVAL_RESULTS)

    @property
    def draft_answers(self) -> Path:
        return self.out(DRAFT_ANSWERS)

    @property
    def review_overrides(self) -> Path:
        return self.out(REVIEW_OVERRIDES)

    @property
    def answer_audit(self) -> Path:
        return self.out(ANSWER_AUDIT)

    @property
    def final_report(self) -> Path:
        return self.out(FINAL_REPORT)

    @property
    def retrieval_metrics(self) -> Path:
        return self.out(RETRIEVAL_METRICS)

    @property
    def revised_answers(self) -> Path:
        return self.out(REVISED_ANSWERS)

    @property
    def retrieval_error_analysis(self) -> Path:
        return self.out(RETRIEVAL_ERROR_ANALYSIS)

    @property
    def llm_calls(self) -> Path:
        return self.out(LLM_CALLS)

    @property
    def pipeline_state(self) -> Path:
        return self.out(PIPELINE_STATE)
