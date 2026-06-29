"""Replayable Mini RAG Pipeline.

A staged, replayable retrieval-augmented-generation pipeline:
deterministic chunking + retrieval, Ollama-backed draft generation,
an interactive human-review override checkpoint, per-query audit, and a
final evaluation report. Every stage persists intermediate artifacts and
every LLM call is logged.
"""

__version__ = "0.1.0"
