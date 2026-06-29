"""Tests for the run.py CLI argument parsing and error handling."""

import run as run_mod
from minirag.llm import LLMError
from minirag.state import StageError


def test_parse_args_defaults():
    args = run_mod.parse_args([])
    assert args.documents == "documents"
    assert args.queries == "queries.json"
    assert args.policy == "policy.json"
    assert args.out == "artifacts"
    assert args.auto_continue is False
    assert args.mode is None


def test_parse_args_overrides():
    args = run_mod.parse_args([
        "--documents", "d", "--queries", "q.json", "--policy", "p.json",
        "--out", "o", "--mode", "embedding", "--auto-continue",
        "--model", "m", "--embed-model", "e", "--host", "http://x",
    ])
    assert args.mode == "embedding"
    assert args.auto_continue is True
    assert args.model == "m"
    assert args.host == "http://x"


def test_main_success(monkeypatch):
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return None

    monkeypatch.setattr(run_mod, "run_pipeline", fake_run)
    rc = run_mod.main(["--auto-continue", "--model", "test"])
    assert rc == 0
    assert captured["cfg"].auto_continue is True
    assert captured["cfg"].model == "test"


def test_main_handles_file_not_found(monkeypatch):
    def boom(cfg):
        raise FileNotFoundError("documents missing")

    monkeypatch.setattr(run_mod, "run_pipeline", boom)
    assert run_mod.main([]) == 2


def test_main_handles_stage_error(monkeypatch):
    def boom(cfg):
        raise StageError("bad transition")

    monkeypatch.setattr(run_mod, "run_pipeline", boom)
    assert run_mod.main([]) == 3


def test_main_handles_llm_error(monkeypatch):
    def boom(cfg):
        raise LLMError("ollama down")

    monkeypatch.setattr(run_mod, "run_pipeline", boom)
    assert run_mod.main([]) == 4
