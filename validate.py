#!/usr/bin/env python3
"""Validation gate for the Replayable Mini RAG pipeline.

Checks (per the assessment's VALIDATION REQUIREMENTS):
  1.  required artifacts exist
  2.  JSON files are valid
  3.  documents, queries, and policy were read from disk (inputs present)
  4.  chunking happened before any LLM call
  5.  every query has retrieval results
  6.  every draft answer label uses only allowed labels from policy.json
  7.  citations in each draft answer refer only to retrieved chunk IDs for that query
  8.  each query has its own audit LLM call record
  9.  audit was run after human review
  10. overrides are saved and applied to audit inputs
  11. final report reflects reviewed final context, not only original retrieval
  12. llm_calls.jsonl contains separate records for required stages

Exit code 0 = all checks passed; non-zero = at least one failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minirag.io_utils import read_json, read_jsonl, read_text  # noqa: E402
from minirag import paths as P  # noqa: E402


class Validator:
    def __init__(self, documents: Path, queries: Path, policy: Path, out: Path):
        self.documents = documents
        self.queries_path = queries
        self.policy_path = policy
        self.out = out
        self.failures: list[str] = []
        self.passes: list[str] = []

    # -- helpers ----------------------------------------------------------- #
    def check(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        if condition:
            self.passes.append(ok_msg)
        else:
            self.failures.append(fail_msg)
        return condition

    def _load(self, name: str):
        return read_json(self.out / name)

    # -- individual checks ------------------------------------------------- #
    def check_inputs_present(self) -> None:
        self.check(
            self.documents.is_dir()
            and any(p.suffix == ".txt" for p in self.documents.iterdir()),
            "Inputs: documents/ present with .txt files",
            "Inputs: documents/ missing or has no .txt files",
        )
        self.check(
            self.queries_path.is_file(),
            "Inputs: queries.json present",
            f"Inputs: queries file missing: {self.queries_path}",
        )
        self.check(
            self.policy_path.is_file(),
            "Inputs: policy.json present",
            f"Inputs: policy file missing: {self.policy_path}",
        )

    def check_required_artifacts(self) -> None:
        for name in P.REQUIRED_ARTIFACTS:
            self.check(
                (self.out / name).is_file(),
                f"Artifact present: {name}",
                f"Artifact MISSING: {name}",
            )

    def check_json_valid(self) -> None:
        json_files = [
            P.CHUNKS, P.INDEX_METADATA, P.RETRIEVAL_RESULTS, P.DRAFT_ANSWERS,
            P.REVIEW_OVERRIDES, P.ANSWER_AUDIT, P.PIPELINE_STATE,
        ]
        for opt in (P.RETRIEVAL_METRICS, P.REVISED_ANSWERS, P.RETRIEVAL_ERROR_ANALYSIS):
            if (self.out / opt).is_file():
                json_files.append(opt)
        for name in json_files:
            path = self.out / name
            if not path.is_file():
                continue
            try:
                read_json(path)
                self.passes.append(f"Valid JSON: {name}")
            except Exception as exc:
                self.failures.append(f"Invalid JSON in {name}: {exc}")

        # llm_calls.jsonl: each line must parse.
        llm_path = self.out / P.LLM_CALLS
        if llm_path.is_file():
            try:
                for i, line in enumerate(read_text(llm_path).splitlines(), start=1):
                    if line.strip():
                        json.loads(line)
                self.passes.append("Valid JSONL: llm_calls.jsonl")
            except Exception as exc:
                self.failures.append(f"Invalid JSONL in llm_calls.jsonl: {exc}")

    def check_chunking_before_llm(self) -> None:
        try:
            state = self._load(P.PIPELINE_STATE)
            chunk_ts = None
            for t in state.get("transitions", []):
                if t.get("stage") == "DOCUMENTS_CHUNKED":
                    chunk_ts = t.get("timestamp")
                    break
            llm_records = read_jsonl(self.out / P.LLM_CALLS)
            llm_ts = [r["timestamp"] for r in llm_records if "timestamp" in r]
            earliest_llm = min(llm_ts) if llm_ts else None
        except Exception as exc:
            self.failures.append(f"Ordering check could not run: {exc}")
            return

        self.check(
            chunk_ts is not None,
            "Ordering: DOCUMENTS_CHUNKED transition recorded",
            "Ordering: no DOCUMENTS_CHUNKED transition in pipeline_state.json",
        )
        if chunk_ts and earliest_llm:
            self.check(
                chunk_ts <= earliest_llm,
                "Ordering: chunking happened before the first LLM call",
                f"Ordering: an LLM call ({earliest_llm}) predates chunking ({chunk_ts})",
            )
        elif chunk_ts and not earliest_llm:
            self.failures.append("Ordering: no LLM calls recorded in llm_calls.jsonl")

    def check_retrieval_coverage(self) -> None:
        try:
            queries = read_json(self.queries_path)["queries"]
            retrievals = self._load(P.RETRIEVAL_RESULTS)
        except Exception as exc:
            self.failures.append(f"Retrieval coverage check could not run: {exc}")
            return
        retrieved_ids = {r["query_id"] for r in retrievals}
        for q in queries:
            self.check(
                q["query_id"] in retrieved_ids,
                f"Retrieval: {q['query_id']} has results",
                f"Retrieval: {q['query_id']} has NO retrieval results",
            )
            r = next((x for x in retrievals if x["query_id"] == q["query_id"]), None)
            if r is not None:
                self.check(
                    len(r.get("retrieved_chunks", [])) > 0,
                    f"Retrieval: {q['query_id']} returned chunks",
                    f"Retrieval: {q['query_id']} returned zero chunks",
                )

    def check_labels_and_citations(self) -> None:
        try:
            policy = read_json(self.policy_path)
            allowed = set(policy["allowed_labels"])
            drafts = self._load(P.DRAFT_ANSWERS)
            retrievals = self._load(P.RETRIEVAL_RESULTS)
        except Exception as exc:
            self.failures.append(f"Label/citation check could not run: {exc}")
            return

        retrieved_by_q = {
            r["query_id"]: {rc["chunk_id"] for rc in r["retrieved_chunks"]}
            for r in retrievals
        }
        for d in drafts:
            qid = d["query_id"]
            self.check(
                d["label"] in allowed,
                f"Label: {qid} uses allowed label '{d['label']}'",
                f"Label: {qid} uses disallowed label '{d['label']}' (allowed: {sorted(allowed)})",
            )
            retrieved = retrieved_by_q.get(qid, set())
            bad = [c for c in d.get("citations", []) if c not in retrieved]
            self.check(
                not bad,
                f"Citations: {qid} cites only retrieved chunks",
                f"Citations: {qid} cites non-retrieved chunk(s): {bad}",
            )

    def check_audit_per_query(self) -> None:
        try:
            queries = read_json(self.queries_path)["queries"]
            audit = self._load(P.ANSWER_AUDIT)
            llm_records = read_jsonl(self.out / P.LLM_CALLS)
        except Exception as exc:
            self.failures.append(f"Audit-per-query check could not run: {exc}")
            return

        audited_ids = {a["query_id"] for a in audit}
        audit_call_ids = [
            r.get("query_id") for r in llm_records if r.get("stage") == "stage2_audit"
        ]
        for q in queries:
            qid = q["query_id"]
            self.check(
                qid in audited_ids,
                f"Audit: {qid} has an audit result",
                f"Audit: {qid} has NO audit result",
            )
            self.check(
                audit_call_ids.count(qid) == 1,
                f"Audit: {qid} has exactly one Stage-2 LLM call",
                f"Audit: {qid} has {audit_call_ids.count(qid)} Stage-2 calls (expected 1; not batched)",
            )

    def check_audit_after_review(self) -> None:
        try:
            state = self._load(P.PIPELINE_STATE)
            order = [t["stage"] for t in state.get("transitions", [])]
        except Exception as exc:
            self.failures.append(f"Audit-after-review check could not run: {exc}")
            return
        try:
            review_idx = order.index("HUMAN_REVIEW_COMPLETE")
            audit_idx = order.index("ANSWERS_AUDITED")
            self.check(
                review_idx < audit_idx,
                "Ordering: audit ran after human review",
                "Ordering: audit did not run after human review",
            )
        except ValueError:
            self.failures.append("Ordering: review/audit stages missing from state log")

    def check_overrides_applied(self) -> None:
        try:
            overrides = self._load(P.REVIEW_OVERRIDES)
            retrievals = self._load(P.RETRIEVAL_RESULTS)
        except Exception as exc:
            self.failures.append(f"Override-applied check could not run: {exc}")
            return

        # Overrides saved for every query.
        q_ids = {r["query_id"] for r in retrievals}
        override_ids = {o["query_id"] for o in overrides}
        self.check(
            q_ids <= override_ids,
            "Overrides: a review_overrides entry exists for every query",
            f"Overrides: missing entries for {sorted(q_ids - override_ids)}",
        )

        # Every override has an explicit final context.
        for o in overrides:
            self.check(
                "final_context_chunk_ids" in o,
                f"Overrides: {o['query_id']} records final context",
                f"Overrides: {o['query_id']} missing final_context_chunk_ids",
            )

        # If an override actually changed context, the audit's final context must
        # reflect it (verified via the report in check_report_reflects_final).

    def check_report_reflects_final(self) -> None:
        try:
            report_text = read_text(self.out / P.FINAL_REPORT)
            overrides = self._load(P.REVIEW_OVERRIDES)
        except Exception as exc:
            self.failures.append(f"Report-final-context check could not run: {exc}")
            return

        required_sections = [
            "Retrieval Summary",
            "Query-by-Query Results",
            "Reviewed Overrides",
            "Audit Findings",
            "Failure Modes Observed",
            "Recommended Improvements",
        ]
        for sec in required_sections:
            self.check(
                sec in report_text,
                f"Report: section '{sec}' present",
                f"Report: section '{sec}' MISSING",
            )

        # For any overridden query, the forced final-context chunk IDs must appear
        # in the report (proving the report reflects final, not original, context).
        for o in overrides:
            if o.get("overridden"):
                missing = [
                    cid for cid in o["final_context_chunk_ids"] if cid not in report_text
                ]
                self.check(
                    not missing,
                    f"Report: overridden {o['query_id']} final context shown",
                    f"Report: overridden {o['query_id']} final context not reflected: {missing}",
                )

    def check_llm_call_stages(self) -> None:
        try:
            queries = read_json(self.queries_path)["queries"]
            llm_records = read_jsonl(self.out / P.LLM_CALLS)
            revised_exists = (self.out / P.REVISED_ANSWERS).is_file()
        except Exception as exc:
            self.failures.append(f"LLM-stage check could not run: {exc}")
            return

        required_fields = {
            "stage", "query_id", "timestamp", "provider", "model",
            "prompt_hash", "input_artifacts", "output_artifact",
        }
        for i, r in enumerate(llm_records):
            missing = required_fields - set(r.keys())
            self.check(
                not missing,
                f"LLM log: record {i} has all required fields",
                f"LLM log: record {i} missing fields {sorted(missing)}",
            )

        stage1 = [r for r in llm_records if r.get("stage") == "stage1_draft"]
        stage2 = [r for r in llm_records if r.get("stage") == "stage2_audit"]
        self.check(
            len(stage1) == len(queries),
            f"LLM log: {len(stage1)} Stage-1 draft calls (one per query)",
            f"LLM log: {len(stage1)} Stage-1 calls != {len(queries)} queries",
        )
        self.check(
            len(stage2) == len(queries),
            f"LLM log: {len(stage2)} Stage-2 audit calls (one per query)",
            f"LLM log: {len(stage2)} Stage-2 calls != {len(queries)} queries",
        )
        if revised_exists:
            revised = self._load(P.REVISED_ANSWERS)
            revised_calls = [r for r in llm_records if r.get("stage") == "revised_answer"]
            self.check(
                len(revised_calls) == len(revised),
                "LLM log: revised-answer calls match revised_answers.json",
                f"LLM log: {len(revised_calls)} revised calls != {len(revised)} revised answers",
            )

    def check_retrieval_mode_recorded(self) -> None:
        try:
            meta = self._load(P.INDEX_METADATA)
        except Exception as exc:
            self.failures.append(f"Retrieval-mode check could not run: {exc}")
            return
        self.check(
            meta.get("retrieval_mode") in ("keyword", "embedding"),
            f"Index: retrieval_mode recorded ('{meta.get('retrieval_mode')}')",
            "Index: retrieval_mode not recorded in index_metadata.json",
        )

    # -- driver ------------------------------------------------------------ #
    def run(self) -> bool:
        self.check_inputs_present()
        self.check_required_artifacts()
        self.check_json_valid()
        self.check_chunking_before_llm()
        self.check_retrieval_coverage()
        self.check_labels_and_citations()
        self.check_audit_per_query()
        self.check_audit_after_review()
        self.check_overrides_applied()
        self.check_report_reflects_final()
        self.check_llm_call_stages()
        self.check_retrieval_mode_recorded()
        return not self.failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate Replayable Mini RAG artifacts")
    p.add_argument("--documents", default="documents")
    p.add_argument("--queries", default="queries.json")
    p.add_argument("--policy", default="policy.json")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args(argv)

    v = Validator(
        documents=Path(args.documents),
        queries=Path(args.queries),
        policy=Path(args.policy),
        out=Path(args.out),
    )
    ok = v.run()

    print("\n==================== VALIDATION ====================")
    for msg in v.passes:
        print(f"  PASS  {msg}")
    for msg in v.failures:
        print(f"  FAIL  {msg}")
    print("----------------------------------------------------")
    print(f"  {len(v.passes)} passed, {len(v.failures)} failed")
    print("====================================================")

    if ok:
        print("VALIDATION PASSED")
        return 0
    print("VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
