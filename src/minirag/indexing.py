"""Index metadata construction.

For the keyword (BM25) mode the "index" is computed in-memory at retrieval
time; for the embedding mode it is the set of chunk vectors. In both cases we
record reproducible metadata describing how the index was built so the run can
be audited and the retrieval mode validated.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io_utils import now_iso, stable_hash_obj, write_json
from .schemas import Chunk, Policy


def build_index_metadata(
    chunks: list[Chunk],
    policy: Policy,
    *,
    mode: str,
    embed_model: str | None = None,
) -> dict:
    """Assemble deterministic index metadata recording the retrieval mode."""
    per_document = Counter(c.document_name for c in chunks)

    config = {
        "mode": mode,
        "top_k": policy.retrieval.top_k,
        "chunk_size_chars": policy.retrieval.chunk_size_chars,
        "chunk_overlap_chars": policy.retrieval.chunk_overlap_chars,
        "embed_model": embed_model if mode == "embedding" else None,
    }

    metadata = {
        "retrieval_mode": mode,
        "embed_model": embed_model if mode == "embedding" else None,
        "num_chunks": len(chunks),
        "num_documents": len(per_document),
        "chunks_per_document": dict(sorted(per_document.items())),
        "chunk_ids": [c.chunk_id for c in chunks],
        "config": config,
        "config_hash": stable_hash_obj(config),
        "built_at": now_iso(),
    }
    return metadata


def write_index_metadata(metadata: dict, out_path: Path) -> Path:
    return write_json(out_path, metadata)
