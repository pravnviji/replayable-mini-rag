"""Stage 2: per-query answer audit.

Exactly one LLM call per query (never batched), made *after* the human review
checkpoint. Each audit call receives the original question, the draft answer,
the cited chunk IDs, the FINAL context after any overrides, and the answer
policy / forbidden behaviours. The audit judges whether the answer is actually
supported by the final context, whether citations are appropriate, and whether
the answer overclaims beyond the corpus.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_json
from .llm import call_structured
from .prompts import AUDIT_SYSTEM, chunk_lookup, format_context, policy_block
from .schemas import (
    AuditResult,
    AuditResultLLM,
    Chunk,
    DraftAnswer,
    Policy,
)

STAGE = "stage2_audit"


def build_prompt(
    question: str,
    draft: DraftAnswer,
    final_context_ids: list[str],
    lookup: dict[str, Chunk],
    policy: Policy,
) -> str:
    context = format_context(final_context_ids, lookup)
    citations = ", ".join(draft.citations) if draft.citations else "(none)"
    return (
        f"Original question:\n{question}\n\n"
        f"Draft answer:\n{draft.answer}\n\n"
        f"Draft label: {draft.label}\n"
        f"Cited chunk_ids: {citations}\n\n"
        f"FINAL context after any human override (authoritative for this audit):\n"
        f"{context}\n\n"
        f"Answer policy:\n{policy_block(policy)}\n\n"
        "Audit tasks:\n"
        "- support_assessment: Is the draft answer actually supported by the "
        "FINAL context? Quote or reference the supporting text, or state that it "
        "is unsupported / only partially supported.\n"
        "- citation_check: Are the cited chunk_ids present in the final context "
        "and do they actually support the claims? Note missing or irrelevant "
        "citations.\n"
        "- hallucination_risk: 'high' if the answer asserts facts not in the "
        "final context, 'medium' if it overreaches or is vague, 'low' if every "
        "claim is grounded.\n"
        "- audit_label: 'pass' only if the answer is adequately grounded and "
        "cited; otherwise 'fail'.\n"
        "- recommended_fix: concrete fix, or 'none' if it passes.\n"
        "Return JSON matching the required schema."
    )


def audit_answers(
    drafts: list[DraftAnswer],
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
) -> list[AuditResult]:
    lookup = chunk_lookup(chunks)
    results: list[AuditResult] = []

    # One call per query, in deterministic query order. Never batched.
    for draft in drafts:
        qid = draft.query_id
        question = questions.get(qid, "")
        final_ids = final_context.get(qid, [])

        prompt = build_prompt(question, draft, final_ids, lookup, policy)
        result: AuditResultLLM = call_structured(
            stage=STAGE,
            query_id=qid,
            system_prompt=AUDIT_SYSTEM,
            user_prompt=prompt,
            schema=AuditResultLLM,
            model=model,
            log_path=llm_log_path,
            input_artifacts=input_artifacts,
            output_artifact=str(out_path),
            host=host,
        )

        results.append(
            AuditResult(
                query_id=qid,
                audit_label=result.audit_label,
                support_assessment=result.support_assessment,
                citation_check=result.citation_check,
                hallucination_risk=result.hallucination_risk,
                recommended_fix=result.recommended_fix,
            )
        )

    write_json(out_path, [r.model_dump() for r in results])
    return results
