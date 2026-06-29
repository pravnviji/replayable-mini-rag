"""Item 7: deterministic retrieval metrics (recall@k / hit@k).

Computed in code only when queries carry expected-evidence annotations
(``expected_chunk_ids`` or ``expected_documents``). If no query has annotations
the stage is skipped gracefully and no file is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .io_utils import write_json
from .schemas import Chunk, Query, QueryRetrieval


def has_annotations(queries: list[Query]) -> bool:
    return any(q.expected_chunk_ids or q.expected_documents for q in queries)


def compute_metrics(
    queries: list[Query],
    retrievals: list[QueryRetrieval],
    chunks: list[Chunk],
    *,
    top_k: int,
    out_path: Path,
) -> Optional[dict]:
    """Compute recall@k and hit@k; return ``None`` (and write nothing) if no annotations."""
    if not has_annotations(queries):
        return None

    chunk_doc = {c.chunk_id: c.document_name for c in chunks}
    retrieval_by_id = {r.query_id: r for r in retrievals}

    per_query = []
    recall_sum = 0.0
    hit_sum = 0
    counted = 0

    for q in queries:
        expected_chunks = set(q.expected_chunk_ids or [])
        expected_docs = set(q.expected_documents or [])
        if not expected_chunks and not expected_docs:
            per_query.append(
                {
                    "query_id": q.query_id,
                    "annotated": False,
                    "recall_at_k": None,
                    "hit_at_k": None,
                }
            )
            continue

        r = retrieval_by_id.get(q.query_id)
        retrieved_ids = [rc.chunk_id for rc in r.retrieved_chunks] if r else []
        retrieved_docs = {chunk_doc.get(cid) for cid in retrieved_ids}

        if expected_chunks:
            found = expected_chunks & set(retrieved_ids)
            recall = len(found) / len(expected_chunks)
            hit = 1 if found else 0
        else:  # document-level annotations
            found_docs = expected_docs & retrieved_docs
            recall = len(found_docs) / len(expected_docs)
            hit = 1 if found_docs else 0

        recall_sum += recall
        hit_sum += hit
        counted += 1
        per_query.append(
            {
                "query_id": q.query_id,
                "annotated": True,
                "recall_at_k": round(recall, 6),
                "hit_at_k": hit,
            }
        )

    metrics = {
        "k": top_k,
        "num_annotated_queries": counted,
        "mean_recall_at_k": round(recall_sum / counted, 6) if counted else None,
        "hit_rate_at_k": round(hit_sum / counted, 6) if counted else None,
        "per_query": per_query,
    }
    write_json(out_path, metrics)
    return metrics
