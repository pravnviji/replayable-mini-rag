"""Deterministic document chunking.

Chunking is pure code with no LLM involvement. Each document is split into
fixed-size character windows with a fixed overlap. Given the same documents and
the same ``chunk_size_chars`` / ``chunk_overlap_chars``, the output (including
chunk IDs and ordering) is byte-for-byte reproducible.

Chunk IDs are stable and human-readable: ``{document_stem}-{index:04d}``.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import list_txt_files, read_text, write_json
from .schemas import Chunk


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
    for doc_path in list_txt_files(documents_dir):
        document_name = doc_path.name
        stem = doc_path.stem
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
