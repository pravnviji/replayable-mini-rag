"""Ollama client wrapper with structured output, determinism, and call logging.

Every chat call:
  * is sent with deterministic options (temperature 0, fixed seed, top_k/top_p),
  * is constrained to a pydantic JSON schema via the ``format`` parameter,
  * is appended as one record to ``llm_calls.jsonl`` with the required fields.

The ``ollama`` import is local so importing this module never fails when Ollama
is absent (only *calling* it requires a running server).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from .io_utils import append_jsonl, now_iso, sha256_text

T = TypeVar("T", bound=BaseModel)

PROVIDER = "ollama"

# Deterministic generation options (best-effort reproducibility).
DETERMINISTIC_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "top_k": 1,
    "top_p": 1,
    "num_ctx": 8192,
}


class LLMError(RuntimeError):
    """Raised when an LLM call cannot be completed."""


def _client(host: str | None = None):
    """Return an Ollama client, raising a clear error if the lib is missing."""
    try:
        import ollama  # local import
    except ImportError as exc:  # pragma: no cover - import guard
        raise LLMError(
            "The 'ollama' Python package is not installed. Run 'make setup' or "
            "'pip install -r requirements.txt'."
        ) from exc

    host = host or os.environ.get("OLLAMA_HOST")
    if host:
        return ollama.Client(host=host)
    return ollama


def default_model() -> str:
    return os.environ.get("LLM_MODEL", "llama3.1:8b")


def default_embed_model() -> str:
    return os.environ.get("EMBED_MODEL", "nomic-embed-text")


def call_structured(
    *,
    stage: str,
    query_id: str | None,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    model: str,
    log_path: Path,
    input_artifacts: list[str],
    output_artifact: str,
    host: str | None = None,
) -> T:
    """Make one structured chat call and append a record to ``llm_calls.jsonl``.

    Returns the validated pydantic object. Raises ``LLMError`` on connection or
    parsing failure with an actionable message.
    """
    client = _client(host)
    model = model or default_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # The prompt hash covers everything that influences the output so the call
    # is auditable and (config-)reproducible.
    prompt_hash = sha256_text(
        "\n".join(
            [
                f"model={model}",
                f"options={sorted(DETERMINISTIC_OPTIONS.items())}",
                f"schema={schema.__name__}",
                system_prompt,
                user_prompt,
            ]
        )
    )

    # Log the call *before* sending so a crash still leaves a trace, then we
    # rely on the returned object for downstream artifacts.
    record = {
        "stage": stage,
        "query_id": query_id,
        "timestamp": now_iso(),
        "provider": PROVIDER,
        "model": model,
        "prompt_hash": prompt_hash,
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
    }
    append_jsonl(log_path, record)

    try:
        response = client.chat(
            model=model,
            messages=messages,
            format=schema.model_json_schema(),
            options=DETERMINISTIC_OPTIONS,
        )
    except LLMError:
        raise
    except Exception as exc:  # connection errors, model-not-found, etc.
        raise LLMError(
            f"Ollama chat call failed for stage={stage} query_id={query_id}: {exc}. "
            "Is 'ollama serve' running and the model pulled "
            f"(try: ollama pull {model})?"
        ) from exc

    content = _extract_content(response)
    try:
        return schema.model_validate_json(content)
    except Exception as exc:
        raise LLMError(
            f"Failed to parse structured output for stage={stage} "
            f"query_id={query_id}: {exc}. Raw content: {content[:500]}"
        ) from exc


def embed_texts(
    texts: list[str],
    *,
    model: str,
    host: str | None = None,
) -> list[list[float]]:
    """Return embedding vectors for ``texts`` using Ollama."""
    client = _client(host)
    model = model or default_embed_model()
    vectors: list[list[float]] = []
    for text in texts:
        try:
            resp = client.embeddings(model=model, prompt=text)
        except Exception as exc:
            raise LLMError(
                f"Ollama embeddings call failed: {exc}. Is 'ollama serve' running "
                f"and the model pulled (try: ollama pull {model})?"
            ) from exc
        vectors.append(list(_extract_embedding(resp)))
    return vectors


def _extract_content(response) -> str:
    """Pull the message content from an Ollama chat response (dict or object)."""
    if isinstance(response, dict):
        return response["message"]["content"]
    # ollama>=0.4 returns an object with a ``.message.content`` attribute.
    return response.message.content


def _extract_embedding(response):
    if isinstance(response, dict):
        return response["embedding"]
    return response.embedding
