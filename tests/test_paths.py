"""Tests for path resolution and artifact-name constants."""

from pathlib import Path

from minirag import paths as P
from minirag.paths import Paths


def test_required_artifacts_listed():
    for name in (
        P.CHUNKS, P.INDEX_METADATA, P.RETRIEVAL_RESULTS, P.DRAFT_ANSWERS,
        P.REVIEW_OVERRIDES, P.ANSWER_AUDIT, P.FINAL_REPORT, P.LLM_CALLS,
        P.PIPELINE_STATE,
    ):
        assert name in P.REQUIRED_ARTIFACTS


def test_paths_properties_resolve_under_out_dir():
    paths = Paths(
        documents=Path("documents"),
        queries=Path("queries.json"),
        policy=Path("policy.json"),
        out_dir=Path("artifacts"),
    )
    assert paths.chunks == Path("artifacts/chunks.json")
    assert paths.final_report == Path("artifacts/final_report.md")
    assert paths.llm_calls == Path("artifacts/llm_calls.jsonl")
    assert paths.out("custom.json") == Path("artifacts/custom.json")
