"""Disk IO helpers: atomic JSON/JSONL writes, loaders, hashing, timestamps.

All writes are atomic (write to a temp file in the same directory, then
``os.replace``) so a crashed run never leaves a half-written artifact that
would fool the validator.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (a directory) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> Path:
    """Atomically write ``data`` as pretty-printed, sorted JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    _atomic_write_text(path, text + "\n")
    return path


def read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: dict) -> Path:
    """Append a single JSON object as one line to a JSONL file.

    Appends are not atomic across processes, but each line is written in one
    ``write`` call which is sufficient for the single-process pipeline.
    """
    path = Path(path)
    ensure_dir(path.parent)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file (skips blank lines)."""
    records: list[dict] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_text(path: Path, text: str) -> Path:
    """Atomically write a text file."""
    path = Path(path)
    ensure_dir(path.parent)
    _atomic_write_text(path, text)
    return path


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def list_txt_files(documents_dir: Path) -> list[Path]:
    """Return ``.txt`` files in ``documents_dir`` sorted by name (deterministic)."""
    documents_dir = Path(documents_dir)
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"documents directory not found: {documents_dir}")
    return sorted(
        (p for p in documents_dir.iterdir() if p.suffix == ".txt" and p.is_file()),
        key=lambda p: p.name,
    )


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a string (used for deterministic prompt hashes / config hashes)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash_obj(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def _atomic_write_text(path: Path, text: str) -> None:
    directory = str(path.parent)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            # Flush user-space and OS buffers to disk before the rename so a
            # crash/power-loss right after os.replace cannot leave the target
            # as a zero-length or partially written file.
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _fsync_dir(directory: str) -> None:
    """fsync a directory so a rename into it is durable (best-effort)."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        # Some platforms/filesystems (e.g. certain network mounts) disallow
        # fsync on a directory; the rename itself is still atomic.
        pass
    finally:
        os.close(dir_fd)


def iter_lines(items: Iterable[str]) -> str:
    """Join an iterable of strings with newlines (small helper for prompts)."""
    return "\n".join(items)
