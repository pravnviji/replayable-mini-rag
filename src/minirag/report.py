"""Final evaluation report (``final_report.md``).

Required sections:
  * Retrieval Summary
  * Query-by-Query Results
  * Reviewed Overrides
  * Audit Findings
  * Failure Modes Observed
  * Recommended Improvements

For each query the report shows the question, final-context chunk IDs (after any
override), the draft label, the audit label, and a final recommendation. Grounded
answers are clearly distinguished from unsupported / weakly-supported ones.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_text
from .schemas import (
    AuditResult,
    DraftAnswer,
    Policy,
    Query,
    QueryRetrieval,
    ReviewOverride,
    RevisedAnswer,
)

GROUNDED = "GROUNDED"
WEAK = "WEAK / PARTIAL"
UNSUPPORTED = "UNSUPPORTED"


def _grounding_status(draft: DraftAnswer, audit: AuditResult) -> str:
    if audit.audit_label == "pass" and draft.label == "supported" and audit.hallucination_risk == "low":
        return GROUNDED
    if draft.label == "unsupported" or audit.audit_label == "fail" or audit.hallucination_risk == "high":
        return UNSUPPORTED
    return WEAK


def _final_recommendation(draft: DraftAnswer, audit: AuditResult, revised: RevisedAnswer | None) -> str:
    if revised is not None:
        return (
            f"Answer regenerated more conservatively after audit "
            f"({revised.reason_for_revision}). Use revised answer."
        )
    if audit.audit_label == "pass" and audit.hallucination_risk == "low":
        return "Accept answer as grounded."
    if audit.recommended_fix and audit.recommended_fix.lower() not in ("none", ""):
        return f"Apply audit fix: {audit.recommended_fix}"
    return "Treat as unsupported / weakly supported; do not present as grounded fact."


def build_report(
    *,
    queries: list[Query],
    retrievals: list[QueryRetrieval],
    drafts: list[DraftAnswer],
    overrides: list[ReviewOverride],
    audits: list[AuditResult],
    revised: list[RevisedAnswer],
    policy: Policy,
    index_metadata: dict,
    metrics: dict | None,
    error_analysis: list[dict],
    out_path: Path,
) -> Path:
    draft_by_id = {d.query_id: d for d in drafts}
    audit_by_id = {a.query_id: a for a in audits}
    override_by_id = {o.query_id: o for o in overrides}
    revised_by_id = {r.query_id: r for r in revised}
    retrieval_by_id = {r.query_id: r for r in retrievals}

    lines: list[str] = []
    lines.append("# Final Evaluation Report")
    lines.append("")
    lines.append(
        "This report separates **grounded** answers from **unsupported / weakly "
        "supported** ones. Status legend: "
        f"`{GROUNDED}`, `{WEAK}`, `{UNSUPPORTED}`."
    )
    lines.append("")

    # ---------------- Retrieval Summary ----------------
    lines.append("## Retrieval Summary")
    lines.append("")
    lines.append(f"- Retrieval mode: `{index_metadata.get('retrieval_mode')}`")
    lines.append(f"- Top-k: {policy.retrieval.top_k}")
    lines.append(
        f"- Chunking: {policy.retrieval.chunk_size_chars} chars, "
        f"{policy.retrieval.chunk_overlap_chars} overlap"
    )
    lines.append(f"- Documents: {index_metadata.get('num_documents')}")
    lines.append(f"- Chunks: {index_metadata.get('num_chunks')}")
    lines.append(f"- Queries: {len(queries)}")
    if metrics:
        lines.append(
            f"- Retrieval metrics (annotated queries={metrics.get('num_annotated_queries')}): "
            f"mean recall@{metrics.get('k')}={metrics.get('mean_recall_at_k')}, "
            f"hit@{metrics.get('k')}={metrics.get('hit_rate_at_k')}"
        )
    else:
        lines.append("- Retrieval metrics: skipped (no expected-evidence annotations)")
    lines.append("")

    # ---------------- Query-by-Query Results ----------------
    lines.append("## Query-by-Query Results")
    lines.append("")
    for q in queries:
        draft = draft_by_id.get(q.query_id)
        audit = audit_by_id.get(q.query_id)
        override = override_by_id.get(q.query_id)
        revised_ans = revised_by_id.get(q.query_id)
        final_ids = override.final_context_chunk_ids if override else []
        status = _grounding_status(draft, audit) if draft and audit else "UNKNOWN"

        lines.append(f"### {q.query_id}: {q.question}")
        lines.append("")
        lines.append(f"- Grounding status: **{status}**")
        final_ctx = ", ".join(f"`{c}`" for c in final_ids) if final_ids else "(none)"
        lines.append(f"- Final context chunk IDs: {final_ctx}")
        if draft:
            cites = ", ".join(f"`{c}`" for c in draft.citations) if draft.citations else "(none)"
            lines.append(f"- Draft label: `{draft.label}`")
            lines.append(f"- Draft answer: {draft.answer}")
            lines.append(f"- Draft citations: {cites}")
        if audit:
            lines.append(f"- Audit label: `{audit.audit_label}`")
            lines.append(f"- Hallucination risk: `{audit.hallucination_risk}`")
        if revised_ans:
            rcites = ", ".join(f"`{c}`" for c in revised_ans.citations) if revised_ans.citations else "(none)"
            lines.append(f"- Revised answer (`{revised_ans.label}`): {revised_ans.answer}")
            lines.append(f"- Revised citations: {rcites}")
        if draft and audit:
            lines.append(
                f"- Final recommendation: {_final_recommendation(draft, audit, revised_ans)}"
            )
        lines.append("")

    # ---------------- Reviewed Overrides ----------------
    lines.append("## Reviewed Overrides")
    lines.append("")
    overridden = [o for o in overrides if o.overridden]
    if not overridden:
        lines.append("No retrieval overrides were applied during human review.")
    else:
        for o in overridden:
            ctx = ", ".join(f"`{c}`" for c in o.final_context_chunk_ids)
            lines.append(f"- **{o.query_id}**: forced final context {ctx} ({o.note})")
    lines.append("")

    # ---------------- Audit Findings ----------------
    lines.append("## Audit Findings")
    lines.append("")
    for q in queries:
        audit = audit_by_id.get(q.query_id)
        if not audit:
            continue
        lines.append(
            f"- **{q.query_id}** -> audit `{audit.audit_label}`, risk "
            f"`{audit.hallucination_risk}`"
        )
        lines.append(f"  - Support: {audit.support_assessment}")
        lines.append(f"  - Citations: {audit.citation_check}")
        if audit.recommended_fix and audit.recommended_fix.lower() not in ("none", ""):
            lines.append(f"  - Recommended fix: {audit.recommended_fix}")
    lines.append("")

    # ---------------- Failure Modes Observed ----------------
    lines.append("## Failure Modes Observed")
    lines.append("")
    fail_count = sum(1 for a in audits if a.audit_label == "fail")
    high_risk = sum(1 for a in audits if a.hallucination_risk == "high")
    unsupported = sum(1 for d in drafts if d.label == "unsupported")
    lines.append(f"- Draft answers labelled `unsupported`: {unsupported}")
    lines.append(f"- Audits failed: {fail_count}")
    lines.append(f"- High hallucination risk: {high_risk}")
    if error_analysis:
        lines.append("- Retrieval error analysis:")
        for f in error_analysis:
            lines.append(
                f"  - **{f['query_id']}** -> `{f['failure_type']}`: {f['description']}"
            )
    else:
        lines.append("- Retrieval error analysis: no notable retrieval failures detected.")
    lines.append("")

    # ---------------- Recommended Improvements ----------------
    lines.append("## Recommended Improvements")
    lines.append("")
    recs = _recommendations(error_analysis, fail_count, high_risk, unsupported, policy)
    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")

    return write_text(out_path, "\n".join(lines))


def _recommendations(
    error_analysis: list[dict],
    fail_count: int,
    high_risk: int,
    unsupported: int,
    policy: Policy,
) -> list[str]:
    recs: list[str] = []
    types = {f["failure_type"] for f in error_analysis}

    if "corpus_gap" in types:
        recs.append(
            "Several questions cannot be answered from the corpus (corpus_gap). "
            "Expand the document set or explicitly return 'unsupported' for "
            "out-of-scope questions (already enforced here)."
        )
    if "ranking" in types:
        recs.append(
            "Ranking failures observed: consider a hybrid keyword+embedding "
            "retriever or a reranking step to surface the most relevant chunk."
        )
    if "chunking" in types:
        recs.append(
            "Chunk boundaries split relevant evidence: try smaller chunks with "
            "larger overlap, or sentence-aware chunking."
        )
    if "ambiguity" in types:
        recs.append(
            "Ambiguous questions led to partial support: add query clarification "
            "or expand top-k to provide more context."
        )
    if high_risk or fail_count:
        recs.append(
            "Keep the Stage-2 audit and conservative regeneration in place; they "
            "successfully flagged overclaiming answers."
        )
    if not recs:
        recs.append(
            "All answers were adequately grounded; maintain the current "
            "deterministic chunking, retrieval, and audit configuration."
        )
    return recs
