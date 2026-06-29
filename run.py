#!/usr/bin/env python3
"""Entry point for the Replayable Mini RAG pipeline.

Usage:
    python run.py [--documents DIR] [--queries FILE] [--policy FILE]
                  [--out DIR] [--mode keyword|embedding]
                  [--auto-continue] [--review-input FILE]
                  [--model NAME] [--embed-model NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run from a clean checkout.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minirag.llm import LLMError  # noqa: E402
from minirag.paths import Paths  # noqa: E402
from minirag.pipeline import RunConfig, run_pipeline  # noqa: E402
from minirag.state import StageError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replayable Mini RAG pipeline")
    p.add_argument("--documents", default="documents", help="documents directory")
    p.add_argument("--queries", default="queries.json", help="queries JSON file")
    p.add_argument("--policy", default="policy.json", help="policy JSON file")
    p.add_argument("--out", default="artifacts", help="output artifacts directory")
    p.add_argument(
        "--mode",
        choices=["keyword", "embedding"],
        default=None,
        help="retrieval mode (overrides policy.json)",
    )
    p.add_argument(
        "--auto-continue",
        action="store_true",
        help="skip interactive review (no overrides); for CI/automation",
    )
    p.add_argument(
        "--review-input",
        default=None,
        help="JSON file of overrides for non-interactive review",
    )
    p.add_argument("--model", default=None, help="Ollama generation model")
    p.add_argument("--embed-model", default=None, help="Ollama embedding model")
    p.add_argument("--host", default=None, help="Ollama host (e.g. http://localhost:11434)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    paths = Paths(
        documents=Path(args.documents),
        queries=Path(args.queries),
        policy=Path(args.policy),
        out_dir=Path(args.out),
    )
    cfg = RunConfig(
        paths=paths,
        mode=args.mode,
        auto_continue=args.auto_continue,
        review_input=Path(args.review_input) if args.review_input else None,
        model=args.model,
        embed_model=args.embed_model,
        host=args.host,
    )

    try:
        run_pipeline(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: input not found: {exc}", file=sys.stderr)
        return 2
    except StageError as exc:
        print(f"ERROR: stage machine violation: {exc}", file=sys.stderr)
        return 3
    except LLMError as exc:
        print(f"ERROR: LLM call failed: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
