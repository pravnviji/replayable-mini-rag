"""Tests for validate.py: passes on a good run, fails on a corrupted artifact."""

import json
from pathlib import Path

import pytest

from minirag.paths import Paths
from minirag.pipeline import RunConfig, run_pipeline

import validate as validate_mod


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
    ]}))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "retrieval": {"mode": "keyword", "top_k": 2, "chunk_size_chars": 200, "chunk_overlap_chars": 40},
        "allowed_labels": ["supported", "partially_supported", "unsupported"],
        "citation_required": True,
        "forbidden_behaviours": [],
        "generation": {"provider": "ollama", "model": "test-model"},
    }))
    return Paths(documents=docs, queries=queries, policy=policy, out_dir=tmp_path / "artifacts")


def _validator(paths: Paths) -> validate_mod.Validator:
    return validate_mod.Validator(
        documents=paths.documents,
        queries=paths.queries,
        policy=paths.policy,
        out=paths.out_dir,
    )


def test_validation_passes_on_clean_run(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    run_pipeline(RunConfig(paths=paths, auto_continue=True, model="test-model"))
    v = _validator(paths)
    ok = v.run()
    assert ok, f"unexpected failures: {v.failures}"
    assert v.failures == []


def test_validation_fails_when_artifact_missing(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    run_pipeline(RunConfig(paths=paths, auto_continue=True, model="test-model"))
    # Corrupt the run: delete the audit artifact.
    paths.answer_audit.unlink()
    v = _validator(paths)
    ok = v.run()
    assert not ok
    assert any("answer_audit.json" in f for f in v.failures)


def test_validation_detects_bad_label(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    run_pipeline(RunConfig(paths=paths, auto_continue=True, model="test-model"))
    # Inject a disallowed label into draft answers.
    drafts = json.load(open(paths.draft_answers))
    drafts[0]["label"] = "totally_invalid"
    paths.draft_answers.write_text(json.dumps(drafts))
    v = _validator(paths)
    ok = v.run()
    assert not ok
    assert any("disallowed label" in f for f in v.failures)


def test_validation_main_entrypoint(tmp_path: Path, fake_ollama):
    paths = _make_fixtures(tmp_path)
    run_pipeline(RunConfig(paths=paths, auto_continue=True, model="test-model"))
    code = validate_mod.main([
        "--documents", str(paths.documents),
        "--queries", str(paths.queries),
        "--policy", str(paths.policy),
        "--out", str(paths.out_dir),
    ])
    assert code == 0
