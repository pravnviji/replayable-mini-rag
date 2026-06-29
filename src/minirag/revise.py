"""Item 8: regenerated answers after audit failure.

For any query whose audit returned ``audit_label == 'fail'`` or
``hallucination_risk == 'high'``, make one more conservative LLM call using the
audited FINAL context. The revised answer must preserve citation discipline
(citations restricted to the final context chunk IDs).

If no query qualifies, nothing is written and ``[]`` is returned.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_json
from .llm import call_structured
from .prompts import REVISION_SYSTEM, chunk_lookup, format_context, policy_block
from .schemas import (
    AuditResult,
    Chunk,
    DraftAnswer,
    Policy,
    RevisedAnswer,
    RevisedAnswerLLM,
)

STAGE = "revised_answer"


def needs_revision(audit: AuditResult) -> bool:
    return audit.audit_label == "fail" or audit.hallucination_risk == "high"


def _conservative_label(policy: Policy) -> str:
    for preferred in ("unsupported", "partially_supported"):
        if preferred in policy.allowed_labels:
            return preferred
    return policy.allowed_labels[-1] if policy.allowed_labels else "unsupported"


def build_prompt(
    question: str,
    draft: DraftAnswer,
    audit: AuditResult,
    final_context_ids: list[str],
    lookup: dict[str, Chunk],
    policy: Policy,
) -> str:
    context = format_context(final_context_ids, lookup)
    return (
        f"Question:\n{question}\n\n"
        f"Previous answer (failed audit):\n{draft.answer}\n\n"
        f"Audit findings:\n"
        f"- support_assessment: {audit.support_assessment}\n"
        f"- citation_check: {audit.citation_check}\n"
        f"- hallucination_risk: {audit.hallucination_risk}\n"
        f"- recommended_fix: {audit.recommended_fix}\n\n"
        f"FINAL context (authoritative; cite only these chunk_ids):\n{context}\n\n"
        f"Answer policy:\n{policy_block(policy)}\n\n"
        "Produce a revised answer that:\n"
        "- stays strictly within the final context,\n"
        "- explicitly states if the corpus does not support an answer,\n"
        "- cites only chunk_ids from the final context,\n"
        "- chooses the most conservative accurate label.\n"
        "Return JSON matching the required schema."
    )


def revise_answers(
    drafts: list[DraftAnswer],
    audits: list[AuditResult],
    chunks: list[Chunk],
    policy: Policy,
    final_context: dict[str, list[str]],
    questions: dict[str, str],
    *,
    model: str,
    out_path: Path,
    llm_log_path: Path,
    input_artifacts: list[str],
    host: str | None = None,
) -> list[RevisedAnswer]:
    lookup = chunk_lookup(chunks)
    allowed = set(policy.allowed_labels)
    fallback_label = _conservative_label(policy)
    drafts_by_id = {d.query_id: d for d in drafts}

    revised: list[RevisedAnswer] = []
    for audit in audits:
        if not needs_revision(audit):
            continue
        qid = audit.query_id
        draft = drafts_by_id.get(qid)
        if draft is None:
            continue
        question = questions.get(qid, "")
        final_ids = final_context.get(qid, [])

        prompt = build_prompt(question, draft, audit, final_ids, lookup, policy)
        result: RevisedAnswerLLM = call_structured(
            stage=STAGE,
            query_id=qid,
            system_prompt=REVISION_SYSTEM,
            user_prompt=prompt,
            schema=RevisedAnswerLLM,
            model=model,
            log_path=llm_log_path,
            input_artifacts=input_artifacts,
            output_artifact=str(out_path),
            host=host,
        )

        label = result.label if result.label in allowed else fallback_label
        final_set = set(final_ids)
        citations = [c for c in result.citations if c in final_set]

        reason = (
            f"audit_label={audit.audit_label}, "
            f"hallucination_risk={audit.hallucination_risk}"
        )
        revised.append(
            RevisedAnswer(
                query_id=qid,
                answer=result.answer,
                label=label,
                citations=citations,
                reasoning_summary=result.reasoning_summary,
                reason_for_revision=reason,
            )
        )

    if revised:
        write_json(out_path, [r.model_dump() for r in revised])
    return revised
