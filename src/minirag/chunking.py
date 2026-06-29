"""Deterministic document chunking.

Chunking is pure code with no LLM involvement. Each document is split into
fixed-size character windows with a fixed overlap. Given the same documents and
the same ``chunk_size_chars`` / ``chunk_overlap_chars``, the output (including
chunk IDs and ordering) is byte-for-byte reproducible.

Chunk IDs are stable and human-readable: ``{document_stem}-{index:04d}``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .io_utils import list_txt_files, read_text, write_json
from .schemas import Chunk

# Per-document size cap (bytes) to bound memory use when ingesting the corpus.
# Generous default so normal corpora are unaffected; override with the
# ``MINIRAG_MAX_DOC_BYTES`` environment variable (0 disables the check).
DEFAULT_MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 MiB


def _max_doc_bytes() -> int:
    raw = os.environ.get("MINIRAG_MAX_DOC_BYTES")
    if raw is None:
        return DEFAULT_MAX_DOC_BYTES
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_MAX_DOC_BYTES


def chunk_text(
    text: str,
    *,
    size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split ``text`` into ``(start_char, end_char, chunk_text)`` windows.

    Windows advance by ``size - overlap`` characters. The final window is
    clamped to the end of the text. Empty/whitespace-only windows are dropped.
    """
    if size <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap_chars must satisfy 0 <= overlap < size")

    step = size - overlap
    windows: list[tuple[int, int, str]] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        piece = text[start:end]
        if piece.strip():
            windows.append((start, end, piece))
        if end >= n:
            break
        start += step
    return windows


def build_chunks(
    documents_dir: Path,
    *,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Read all ``.txt`` files (sorted) and produce a deterministic chunk list."""
    chunks: list[Chunk] = []
    max_bytes = _max_doc_bytes()
    for doc_path in list_txt_files(documents_dir):
        document_name = doc_path.name
        stem = doc_path.stem
        if max_bytes > 0:
            size = doc_path.stat().st_size
            if size > max_bytes:
                raise ValueError(
                    f"document {document_name!r} is {size} bytes, exceeding the "
                    f"{max_bytes}-byte limit; raise or disable it via "
                    f"MINIRAG_MAX_DOC_BYTES (0 disables)."
                )
        text = read_text(doc_path)
        for i, (start, end, piece) in enumerate(
            chunk_text(text, size=size, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}-{i:04d}",
                    document_name=document_name,
                    start_char=start,
                    end_char=end,
                    text=piece,
                )
            )
    return chunks


def write_chunks(chunks: list[Chunk], out_path: Path) -> Path:
    """Persist chunks to ``chunks.json``."""
    return write_json(out_path, [c.model_dump() for c in chunks])
