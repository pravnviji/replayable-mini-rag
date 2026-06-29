"""Tests for prompt-construction helpers."""

from minirag.prompts import (
    chunk_lookup,
    format_context,
    policy_block,
)
from minirag.schemas import Chunk, Policy


def _chunks():
    return [
        Chunk(chunk_id="a-0000", document_name="a.txt", start_char=0, end_char=5, text="alpha"),
        Chunk(chunk_id="b-0000", document_name="b.txt", start_char=0, end_char=5, text="gamma"),
    ]


def test_chunk_lookup_indexes_by_id():
    lookup = chunk_lookup(_chunks())
    assert lookup["a-0000"].text == "alpha"


def test_format_context_includes_ids_and_text():
    lookup = chunk_lookup(_chunks())
    out = format_context(["a-0000"], lookup)
    assert "a-0000" in out
    assert "alpha" in out


def test_format_context_empty():
    assert "no context" in format_context([], {}).lower()


def test_format_context_skips_unknown_ids():
    lookup = chunk_lookup(_chunks())
    out = format_context(["does-not-exist"], lookup)
    assert "no context" in out.lower()


def test_policy_block_lists_labels_and_forbidden():
    p = Policy(allowed_labels=["supported", "unsupported"], forbidden_behaviours=["no lies"])
    block = policy_block(p)
    assert "supported" in block
    assert "no lies" in block
    assert "Citation required: yes" in block
