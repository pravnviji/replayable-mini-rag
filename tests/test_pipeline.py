"""End-to-end pipeline integration test with the Ollama client mocked.

Exercises the full staged run (INIT -> RESULTS_FINALISED) against temporary
fixtures, then asserts every required artifact is produced and the stage order
is correct. Uses the ``fake_ollama`` fixture so no real model is needed.
"""

import json
from pathlib import Path

from minirag import paths as P
from minirag.io_utils import read_jsonl
from minirag.paths import Paths
from minirag.pipeline import RunConfig, run_pipeline


def _make_fixtures(tmp_path: Path) -> Paths:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "overview.txt").write_text(
        "Event data is retained for 13 months on the standard plan. "
        "SCIM provisioning is available on enterprise plans."
    )
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"queries": [
        {"query_id": "Q1", "question": "How long is event data retained?"},
        {"query_id": "Q2", "question": "Is SCIM supported?"},
    ]}))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "retrieval": {"mode": "keyword", "top_k": 2, "chunk_size_chars": 200, "chunk_overlap_chars": 40},
        "allowed_labels": ["supported", "partially_supported", "unsupported"],
        "citation_required": True,
        "forbidden_behaviours": ["no outside knowledge"],
        "generation": {"provider": "ollama", "model": "test-model"},
    }))
    return Paths(documents=docs, queries=queries, policy=policy, out_dir=tmp_path / "artifacts")


def test_full_pipeline_produces_all_artifacts(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    state = run_pipeline(RunConfig(paths=paths, auto_continue=True, model="test-model"))

    assert state.current == "RESULTS_FINALISED"

    # All required artifacts exist.
    for name in P.REQUIRED_ARTIFACTS:
        assert (paths.out_dir / name).is_file(), f"missing {name}"

    # Stage order recorded correctly.
    order = [t["stage"] for t in state.transitions]
    assert order.index("DOCUMENTS_CHUNKED") < order.index("DRAFT_ANSWERS_GENERATED")
    assert order.index("HUMAN_REVIEW_COMPLETE") < order.index("ANSWERS_AUDITED")

    # One Stage-1 and one Stage-2 LLM call per query.
    records = read_jsonl(paths.llm_calls)
    stage1 = [r for r in records if r["stage"] == "stage1_draft"]
    stage2 = [r for r in records if r["stage"] == "stage2_audit"]
    assert len(stage1) == 2
    assert len(stage2) == 2

    # Chunking happened before the first LLM call (timestamp ordering).
    chunk_ts = next(t["timestamp"] for t in state.transitions if t["stage"] == "DOCUMENTS_CHUNKED")
    earliest_llm = min(r["timestamp"] for r in records)
    assert chunk_ts <= earliest_llm


def test_pipeline_metrics_skipped_without_annotations(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    run_pipeline(RunConfig(paths=paths, auto_continue=True, model="test-model"))
    # No expected-evidence annotations -> metrics file absent.
    assert not paths.retrieval_metrics.is_file()


def test_pipeline_with_override_file(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    review = tmp_path / "ov.json"
    # Force Q1's final context to a chunk id that exists after chunking.
    # The single short doc yields 'overview-0000'.
    review.write_text(json.dumps({"overrides": [
        {"query_id": "Q1", "chunk_ids": ["overview-0000"], "note": "forced"}
    ]}))
    run_pipeline(RunConfig(paths=paths, review_input=review, model="test-model"))

    overrides = json.load(open(paths.review_overrides))
    q1 = next(o for o in overrides if o["query_id"] == "Q1")
    assert q1["overridden"] is True
    assert q1["final_context_chunk_ids"] == ["overview-0000"]
