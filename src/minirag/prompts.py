"""Prompt construction helpers shared by the generation and audit stages."""

from __future__ import annotations

from .schemas import Chunk, Policy


def chunk_lookup(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def format_context(chunk_ids: list[str], lookup: dict[str, Chunk]) -> str:
    """Render the provided chunks as a numbered, clearly-delimited context block."""
    if not chunk_ids:
        return "(no context chunks were provided)"
    blocks = []
    for cid in chunk_ids:
        chunk = lookup.get(cid)
        if chunk is None:
            continue
        blocks.append(
            f"[chunk_id: {chunk.chunk_id} | document: {chunk.document_name}]\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(blocks) if blocks else "(no context chunks were provided)"


def policy_block(policy: Policy) -> str:
    forbidden = "\n".join(f"- {f}" for f in policy.forbidden_behaviours) or "- (none)"
    return (
        f"Allowed labels: {', '.join(policy.allowed_labels)}\n"
        f"Citation required: {'yes' if policy.citation_required else 'no'}\n"
        f"Forbidden behaviours:\n{forbidden}"
    )


GENERATION_SYSTEM = (
    "You are a careful retrieval-augmented answering assistant. You answer "
    "ONLY using the provided context chunks. You never use outside knowledge as "
    "if it were grounded in the corpus. If the context does not contain the "
    "answer, you say so explicitly and choose the most conservative label. "
    "Every factual claim you make must be supported by a cited chunk."
)

AUDIT_SYSTEM = (
    "You are a strict answer auditor for a retrieval-augmented system. You "
    "judge whether a draft answer is actually supported by the FINAL context "
    "provided, whether its citations are appropriate, and whether it overclaims "
    "beyond the corpus. You are skeptical: if support is weak, partial, or "
    "absent, you say so and raise the hallucination risk accordingly."
)

REVISION_SYSTEM = (
    "You are revising an answer that failed audit or carried high hallucination "
    "risk. Produce a more conservative answer that stays strictly within the "
    "final context, preserves citation discipline, and clearly states when the "
    "corpus does not support a claim."
)
