"""Stage state machine ordering enforcement."""

import pytest

from minirag.state import PipelineState, StageError


def test_advance_in_order(tmp_path):
    state = PipelineState.initialise(tmp_path / "state.json")
    state.advance("INPUTS_LOADED")
    state.advance("DOCUMENTS_CHUNKED")
    assert state.current == "DOCUMENTS_CHUNKED"
    assert state.timestamp_of("INPUTS_LOADED") is not None


def test_skipping_stage_raises(tmp_path):
    state = PipelineState.initialise(tmp_path / "state.json")
    with pytest.raises(StageError):
        state.advance("RETRIEVAL_COMPLETE")  # skips several stages


def test_require_enforces_minimum_stage(tmp_path):
    state = PipelineState.initialise(tmp_path / "state.json")
    state.advance("INPUTS_LOADED")
    with pytest.raises(StageError):
        state.require("INDEX_BUILT")


def test_state_persists_and_reloads(tmp_path):
    path = tmp_path / "state.json"
    state = PipelineState.initialise(path)
    state.advance("INPUTS_LOADED")
    reloaded = PipelineState.load(path)
    assert reloaded.current == "INPUTS_LOADED"
    assert reloaded.timestamp_of("INIT") is not None
