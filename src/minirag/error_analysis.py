"""Item 9: retrieval error analysis.

Classifies likely failure causes for queries that did not yield a confidently
grounded, passing answer. Classification is driven by observable evidence from
the retrieval and audit artifacts (not freeform opinion):

  * ``corpus_gap``    - answer unsupported and no retrieved chunk shares meaningful
                        query terms (the corpus likely lacks the information).
  * ``ranking``       - relevant terms appear in the corpus, and at least one
                        retrieved chunk overlaps query terms, yet the audit found
                        the answer unsupported (right doc maybe mis-ranked / weak).
  * ``chunking``      - query terms are present in the corpus but split such that
                        no single retrieved chunk covers them well.
  * ``ambiguity``     - the audit flagged partial/medium support or the draft was
                        partially_supported (question under-specified vs context).

Only queries with a fail/high-risk/weak signal are emitted.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_json
from .retrieval import tokenize
from .schemas import AuditResult, Chunk, DraftAnswer, Query, QueryRetrieval

STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "can", "for", "of", "on",
    "in", "to", "how", "long", "what", "which", "and", "or", "with", "be",
    "service", "product", "customers", "get", "support", "supported",
}


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def _corpus_terms(chunks: list[Chunk]) -> set[str]:
    terms: set[str] = set()
    for c in chunks:
        terms |= set(tokenize(c.text))
    return terms


def analyse(
    queries: list[Query],
    retrievals: list[QueryRetrieval],
    drafts: list[DraftAnswer],
    audits: list[AuditResult],
    chunks: list[Chunk],
    *,
    out_path: Path,
) -> list[dict]:
    by_chunk = {c.chunk_id: c for c in chunks}
    corpus_terms = _corpus_terms(chunks)
    retrieval_by_id = {r.query_id: r for r in retrievals}
    draft_by_id = {d.query_id: d for d in drafts}
    audit_by_id = {a.query_id: a for a in audits}

    findings: list[dict] = []
    for q in queries:
        audit = audit_by_id.get(q.query_id)
        draft = draft_by_id.get(q.query_id)
        if audit is None:
            continue

        weak = (
            audit.audit_label == "fail"
            or audit.hallucination_risk in ("medium", "high")
            or (draft is not None and draft.label != "supported")
        )
        if not weak:
            continue

        q_terms = _content_terms(q.question)
        terms_in_corpus = q_terms & corpus_terms
        coverage = (len(terms_in_corpus) / len(q_terms)) if q_terms else 0.0

        r = retrieval_by_id.get(q.query_id)
        retrieved_chunks = (
            [by_chunk[rc.chunk_id] for rc in r.retrieved_chunks if rc.chunk_id in by_chunk]
            if r
            else []
        )
        # Best single-chunk overlap with query content terms.
        best_overlap = 0
        for ch in retrieved_chunks:
            overlap = len(q_terms & set(tokenize(ch.text)))
            best_overlap = max(best_overlap, overlap)

        failure_type, description = _classify(
            coverage=coverage,
            terms_in_corpus=terms_in_corpus,
            q_terms=q_terms,
            best_overlap=best_overlap,
            audit=audit,
            draft=draft,
        )

        findings.append(
            {
                "query_id": q.query_id,
                "failure_type": failure_type,
                "description": description,
            }
        )

    if findings:
        write_json(out_path, findings)
    return findings


def _classify(
    *,
    coverage: float,
    terms_in_corpus: set[str],
    q_terms: set[str],
    best_overlap: int,
    audit: AuditResult,
    draft: DraftAnswer | None,
) -> tuple[str, str]:
    label = draft.label if draft else "unknown"

    # Almost no query content appears anywhere in the corpus -> genuine gap.
    if coverage < 0.34:
        return (
            "corpus_gap",
            f"Only {len(terms_in_corpus)}/{len(q_terms)} content terms from the "
            f"question appear anywhere in the corpus; the information is likely "
            f"absent. Audit support: {audit.support_assessment[:160]}",
        )

    # Terms exist in the corpus and a retrieved chunk overlaps, but the answer
    # was still judged unsupported -> ranking/grounding problem.
    if best_overlap >= 2 and audit.audit_label == "fail":
        return (
            "ranking",
            "Relevant terms are present and at least one retrieved chunk overlaps "
            "the query, yet the audit judged the answer unsupported - the most "
            "relevant evidence may be mis-ranked or too weak in context.",
        )

    # Terms exist in the corpus but no retrieved chunk covers them well ->
    # likely a chunk-boundary problem.
    if best_overlap <= 1 and len(terms_in_corpus) >= max(1, len(q_terms) // 2):
        return (
            "chunking",
            "Query terms exist in the corpus but no single retrieved chunk covers "
            "them well; chunk boundaries may have split the relevant evidence.",
        )

    # Otherwise treat partial/medium support as ambiguity.
    return (
        "ambiguity",
        f"Partial or uncertain support (draft label '{label}', risk "
        f"'{audit.hallucination_risk}'); the question may be under-specified "
        "relative to the available context.",
    )
