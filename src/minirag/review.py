"""Human review checkpoint for retrieval overrides.

Before the audit stage, the reviewer is shown each query's retrieval results
and draft-answer label, then prompted to override the retrieved chunks for any
query. Overrides become the *final context* used downstream by the audit (and
revision) stages.

Three input modes:
  * interactive (default): prompt on the terminal,
  * ``--review-input <file>``: read overrides from a JSON file (non-interactive),
  * ``--auto-continue``: proceed with no overrides (CI / automation).

Override rules enforced here:
  * every override chunk_id must exist in ``chunks.json``,
  * the final context for an overridden query is saved even if it differs from
    the original top-k retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .io_utils import read_json, write_json
from .schemas import Chunk, DraftAnswer, QueryRetrieval, ReviewOverride

PROMPT = (
    "Do you want to override retrieved chunks for any query before audit?\n"
    "Enter query_id and comma-separated chunk_ids to force as final context,\n"
    "or press Enter to continue.\n"
    "Format: <query_id> <chunk_id,chunk_id,...>   (e.g. 'Q1 product_overview-0000,product_overview-0001')\n"
    "> "
)


def _print_summary(
    retrievals: list[QueryRetrieval],
    drafts_by_id: dict[str, DraftAnswer],
) -> None:
    print("\n================ HUMAN REVIEW CHECKPOINT ================")
    for r in retrievals:
        draft = drafts_by_id.get(r.query_id)
        label = draft.label if draft else "(no draft)"
        print(f"\nQuery {r.query_id}: {r.question}")
        print(f"  Draft label: {label}")
        print("  Retrieved chunks (rank: chunk_id [document] score):")
        for rc in r.retrieved_chunks:
            print(
                f"    {rc.rank}: {rc.chunk_id} [{rc.document_name}] "
                f"{rc.retrieval_score:.4f}"
            )
    print("\n========================================================")


def _default_context(retrievals: list[QueryRetrieval]) -> dict[str, list[str]]:
    """Original top-k context (chunk IDs) per query."""
    return {
        r.query_id: [rc.chunk_id for rc in r.retrieved_chunks] for r in retrievals
    }


def _validate_override_ids(chunk_ids: list[str], valid_ids: set[str]) -> list[str]:
    bad = [cid for cid in chunk_ids if cid not in valid_ids]
    if bad:
        raise ValueError(f"override references unknown chunk_ids: {bad}")
    return chunk_ids


def _build_overrides(
    retrievals: list[QueryRetrieval],
    forced: dict[str, list[str]],
    notes: dict[str, str] | None = None,
) -> list[ReviewOverride]:
    notes = notes or {}
    default_ctx = _default_context(retrievals)
    overrides: list[ReviewOverride] = []
    for r in retrievals:
        qid = r.query_id
        if qid in forced:
            overrides.append(
                ReviewOverride(
                    query_id=qid,
                    overridden=True,
                    final_context_chunk_ids=forced[qid],
                    note=notes.get(qid, "manual override"),
                )
            )
        else:
            overrides.append(
                ReviewOverride(
                    query_id=qid,
                    overridden=False,
                    final_context_chunk_ids=default_ctx[qid],
                    note="",
                )
            )
    return overrides


def run_review(
    retrievals: list[QueryRetrieval],
    drafts: list[DraftAnswer],
    chunks: list[Chunk],
    *,
    out_path: Path,
    auto_continue: bool = False,
    review_input: Path | None = None,
) -> list[ReviewOverride]:
    """Run the checkpoint and persist ``review_overrides.json``."""
    valid_ids = {c.chunk_id for c in chunks}
    valid_qids = {r.query_id for r in retrievals}
    drafts_by_id = {d.query_id: d for d in drafts}

    forced: dict[str, list[str]] = {}
    notes: dict[str, str] = {}

    if review_input is not None:
        forced, notes = _load_review_file(review_input, valid_qids, valid_ids)
    elif auto_continue or not sys.stdin.isatty():
        # Non-interactive: proceed with original retrieval as final context.
        if not auto_continue:
            print(
                "[review] No TTY detected; continuing without overrides. "
                "Use --review-input to supply overrides non-interactively."
            )
    else:
        _print_summary(retrievals, drafts_by_id)
        forced, notes = _interactive_loop(valid_qids, valid_ids)

    overrides = _build_overrides(retrievals, forced, notes)
    write_json(out_path, [o.model_dump() for o in overrides])
    return overrides


def _interactive_loop(
    valid_qids: set[str],
    valid_ids: set[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    forced: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    while True:
        try:
            line = input(PROMPT).strip()
        except EOFError:
            break
        if not line:
            break
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            print("  Invalid format. Use: <query_id> <chunk_id,chunk_id,...>")
            continue
        qid, chunk_csv = parts[0].strip(), parts[1].strip()
        if qid not in valid_qids:
            print(f"  Unknown query_id: {qid}. Known: {sorted(valid_qids)}")
            continue
        chunk_ids = [c.strip() for c in chunk_csv.split(",") if c.strip()]
        try:
            _validate_override_ids(chunk_ids, valid_ids)
        except ValueError as exc:
            print(f"  {exc}")
            continue
        forced[qid] = chunk_ids
        notes[qid] = "manual override"
        print(f"  Recorded override for {qid}: {chunk_ids}")
    return forced, notes


def _load_review_file(
    path: Path,
    valid_qids: set[str],
    valid_ids: set[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load overrides from a JSON file.

    Accepted shapes::

        {"overrides": [{"query_id": "Q1", "chunk_ids": ["..."], "note": "..."}]}
        {"Q1": ["chunk_a", "chunk_b"]}
    """
    data = read_json(path)
    forced: dict[str, list[str]] = {}
    notes: dict[str, str] = {}

    if isinstance(data, dict) and "overrides" in data:
        entries = data["overrides"]
    elif isinstance(data, dict):
        entries = [{"query_id": k, "chunk_ids": v} for k, v in data.items()]
    else:
        raise ValueError("review-input file must be a JSON object")

    for entry in entries:
        qid = entry["query_id"]
        chunk_ids = list(entry.get("chunk_ids", []))
        if qid not in valid_qids:
            raise ValueError(f"review-input references unknown query_id: {qid}")
        _validate_override_ids(chunk_ids, valid_ids)
        forced[qid] = chunk_ids
        notes[qid] = entry.get("note", "override from file")
    return forced, notes


def final_context_map(overrides: list[ReviewOverride]) -> dict[str, list[str]]:
    """Map query_id -> final context chunk IDs after overrides."""
    return {o.query_id: list(o.final_context_chunk_ids) for o in overrides}
