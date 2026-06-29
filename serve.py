#!/usr/bin/env python3
"""Tiny local web UI for the Replayable Mini RAG pipeline.

Dependency-free (uses only the standard library ``http.server``). It exposes:

  * ``GET  /``               -> the single-page UI (``web/index.html``)
  * ``GET  /api/artifacts``  -> JSON view of the artifacts from the last run
  * ``POST /api/query``      -> live BM25 retrieval for an ad-hoc question, plus
                                an LLM draft answer when Ollama is reachable
                                (degrades gracefully to retrieval-only otherwise)

Run:
    python serve.py            # then open http://localhost:8000
    python serve.py --port 9000 --out artifacts

This is a *test harness* for poking at the system in a browser; the canonical
pipeline entry point is still ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minirag import chunking, generate, retrieval  # noqa: E402
from minirag.io_utils import read_json  # noqa: E402
from minirag.llm import LLMError, default_embed_model, default_model  # noqa: E402
from minirag.paths import Paths  # noqa: E402
from minirag.schemas import Policy, QueryRetrieval, RetrievedChunk  # noqa: E402

WEB_DIR = ROOT / "web"

# Populated in main() from CLI args.
PATHS: Paths
POLICY: Policy


@lru_cache(maxsize=1)
def _chunks():
    """Build (and cache) chunks from the configured documents directory."""
    return chunking.build_chunks(
        PATHS.documents,
        size=POLICY.retrieval.chunk_size_chars,
        overlap=POLICY.retrieval.chunk_overlap_chars,
    )


@lru_cache(maxsize=4)
def _retriever(mode: str):
    return retrieval.build_retriever(
        _chunks(), mode=mode, embed_model=default_embed_model()
    )


def _run_query(question: str, mode: str, top_k: int) -> dict:
    """Retrieve for a single ad-hoc question; add an LLM draft when available."""
    chunks = _chunks()
    retriever = _retriever(mode)
    hits = retriever.search(question, top_k)
    by_id = {c.chunk_id: c for c in chunks}
    retrieved = [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "document_name": chunk.document_name,
            "retrieval_score": score,
            "text": chunk.text.strip(),
        }
        for rank, (chunk, score) in enumerate(hits, start=1)
    ]

    result: dict = {"question": question, "mode": mode, "top_k": top_k, "retrieved": retrieved}

    # Best-effort draft generation. Retrieval-only is still a useful UI test.
    qr = QueryRetrieval(
        query_id="adhoc",
        question=question,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_name=r["document_name"],
                rank=r["rank"],
                retrieval_score=r["retrieval_score"],
            )
            for r in retrieved
        ],
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            drafts = generate.generate_drafts(
                [qr],
                chunks,
                POLICY,
                model=POLICY.generation.model or default_model(),
                out_path=tmp_dir / "draft.json",
                llm_log_path=tmp_dir / "llm.jsonl",
                input_artifacts=["(ad-hoc UI query)"],
            )
            d = drafts[0]
            result["draft"] = {
                "answer": d.answer,
                "label": d.label,
                "citations": d.citations,
                "reasoning_summary": d.reasoning_summary,
            }
        except LLMError as exc:
            result["draft"] = None
            result["llm_note"] = (
                f"LLM draft unavailable (showing retrieval only): {exc}"
            )
    return result


def _load_artifacts() -> dict:
    """Load whatever artifacts exist from the configured output directory."""
    out: dict = {"out_dir": str(PATHS.out_dir), "present": []}

    def _maybe(key: str, path: Path, *, kind: str = "json"):
        if not path.exists():
            return
        out["present"].append(path.name)
        if kind == "text":
            out[key] = path.read_text(encoding="utf-8")
        else:
            out[key] = read_json(path)

    _maybe("index_metadata", PATHS.index_metadata)
    _maybe("retrieval_results", PATHS.retrieval_results)
    _maybe("draft_answers", PATHS.draft_answers)
    _maybe("review_overrides", PATHS.review_overrides)
    _maybe("answer_audit", PATHS.answer_audit)
    _maybe("revised_answers", PATHS.revised_answers)
    _maybe("retrieval_metrics", PATHS.retrieval_metrics)
    _maybe("retrieval_error_analysis", PATHS.retrieval_error_analysis)
    _maybe("pipeline_state", PATHS.pipeline_state)
    _maybe("final_report", PATHS.final_report, kind="text")
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniRAGUI/0.1"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            index = WEB_DIR / "index.html"
            if not index.exists():
                self._send_json(500, {"error": "web/index.html not found"})
                return
            self._send(200, index.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/artifacts":
            try:
                self._send_json(200, _load_artifacts())
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                self._send_json(500, {"error": str(exc)})
            return
        self._send_json(404, {"error": f"not found: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/query":
            self._send_json(404, {"error": f"not found: {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "question is required"})
                return
            mode = body.get("mode") or POLICY.retrieval.mode
            top_k = int(body.get("top_k") or POLICY.retrieval.top_k)
            self._send_json(200, _run_query(question, mode, max(1, top_k)))
        except LLMError as exc:
            self._send_json(200, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def main(argv: list[str] | None = None) -> int:
    global PATHS, POLICY
    parser = argparse.ArgumentParser(description="Local web UI for Mini RAG")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--documents", default="documents")
    parser.add_argument("--queries", default="queries.json")
    parser.add_argument("--policy", default="policy.json")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)

    PATHS = Paths(
        documents=Path(args.documents),
        queries=Path(args.queries),
        policy=Path(args.policy),
        out_dir=Path(args.out),
    )
    POLICY = Policy.model_validate(read_json(PATHS.policy)) if PATHS.policy.exists() else Policy()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Mini RAG UI serving at {url}  (Ctrl-C to stop)")
    print(f"  documents={PATHS.documents}  artifacts={PATHS.out_dir}  mode={POLICY.retrieval.mode}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
