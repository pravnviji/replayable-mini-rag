"""Stage 1: draft answer generation.

Exactly one LLM call per query, made *after* retrieval. Each call is grounded
only in that query's retrieved chunks and is given the answer policy, the
allowed labels, and the citation requirement. Outputs are validated so that:

  * ``label`` is one of ``policy.allowed_labels`` (coerced to the most
    conservative allowed label otherwise), and
  * ``citations`` reference only that query's retrieved chunk IDs.
"""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_json
from .llm import call_structured
from .prompts import GENERATION_SYSTEM, chunk_lookup, format_context, policy_block
from .schemas import Chunk, DraftAnswer, DraftAnswerLLM, Policy, QueryRetrieval


STAGE = "stage1_draft"


def _conservative_label(policy: Policy) -> str:
    """Pick the safest allowed label ('unsupported' if available, else the last)."""
    for preferred in ("unsupported", "partially_supported"):
        if preferred in policy.allowed_labels:
            return preferred
    return policy.allowed_labels[-1] if policy.allowed_labels else "unsupported"


def build_prompt(
    question: str,
    retrieved_ids: list[str],
    lookup: dict[str, Chunk],
    policy: Policy,
) -> str:
    context = format_context(retrieved_ids, lookup)
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context chunks (cite by chunk_id):\n{context}\n\n"
        f"Answer policy:\n{policy_block(policy)}\n\n"
        "Instructions:\n"
        "- Answer the question using ONLY the context chunks above.\n"
        "- Choose exactly one label from the allowed labels.\n"
        "- Use 'supported' only if the context fully answers the question.\n"
        "- Use 'partially_supported' if the context is relevant but incomplete.\n"
        "- Use 'unsupported' if the context does not answer the question; in that "
        "case say explicitly that the corpus does not contain the answer.\n"
        "- 'citations' must list only chunk_id values taken from the context above.\n"
        "- If evidence is weak or missing, state that clearly in the answer.\n"
        "Return JSON matching the required schema."
    )


def generate_drafts(
    retrievals: list[QueryRetrieval],
    chunks: list[Chunk],
    policy: Policy,
    *,
    model: str,
    out_path: Path,
    llm_log_path: Path,
    input_artifacts: list[str],
    host: str | None = None,
) -> list[DraftAnswer]:
    lookup = chunk_lookup(chunks)
    allowed = set(policy.allowed_labels)
    fallback_label = _conservative_label(policy)

    drafts: list[DraftAnswer] = []
    for r in retrievals:
        retrieved_ids = [rc.chunk_id for rc in r.retrieved_chunks]
        prompt = build_prompt(r.question, retrieved_ids, lookup, policy)

        result: DraftAnswerLLM = call_structured(
            stage=STAGE,
            query_id=r.query_id,
            system_prompt=GENERATION_SYSTEM,
            user_prompt=prompt,
            schema=DraftAnswerLLM,
            model=model,
            log_path=llm_log_path,
            input_artifacts=input_artifacts,
            output_artifact=str(out_path),
            host=host,
        )

        # Enforce label is an allowed value.
        label = result.label if result.label in allowed else fallback_label

        # Enforce citations are a subset of this query's retrieved chunk IDs,
        # preserving order and dropping anything fabricated/out-of-scope.
        retrieved_set = set(retrieved_ids)
        citations = [c for c in result.citations if c in retrieved_set]

        drafts.append(
            DraftAnswer(
                query_id=r.query_id,
                answer=result.answer,
                label=label,
                citations=citations,
                reasoning_summary=result.reasoning_summary,
            )
        )

    write_json(out_path, [d.model_dump() for d in drafts])
    return drafts
