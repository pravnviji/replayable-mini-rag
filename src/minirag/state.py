"""Pipeline stage state machine.

Enforces the required stage ordering in code and persists every transition
(with a UTC timestamp) to ``pipeline_state.json``. The persisted log lets the
validator prove ordering invariants, e.g. that ``DOCUMENTS_CHUNKED`` happened
before any LLM call recorded in ``llm_calls.jsonl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .io_utils import now_iso, read_json, write_json

# Canonical, ordered list of stages. The pipeline must advance through these
# strictly in order; skipping ahead raises ``StageError``.
STAGES: tuple[str, ...] = (
    "INIT",
    "INPUTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "DRAFT_ANSWERS_GENERATED",
    "HUMAN_REVIEW_COMPLETE",
    "ANSWERS_AUDITED",
    "FINAL_REPORT_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
)

_STAGE_INDEX = {name: i for i, name in enumerate(STAGES)}


class StageError(RuntimeError):
    """Raised when an illegal stage transition is attempted."""


@dataclass
class PipelineState:
    """Tracks and persists the pipeline's current stage.

    The state file records the current stage plus an append-only list of
    transitions, each with the stage name and an ISO-8601 UTC timestamp.
    """

    path: Path
    current: str = "INIT"
    transitions: list[dict] = field(default_factory=list)

    @classmethod
    def initialise(cls, path: Path) -> "PipelineState":
        """Create a fresh state at ``INIT`` and persist it."""
        state = cls(path=Path(path), current="INIT", transitions=[])
        state.transitions.append({"stage": "INIT", "timestamp": now_iso()})
        state.save()
        return state

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load an existing state file."""
        data = read_json(path)
        return cls(
            path=Path(path),
            current=data.get("current", "INIT"),
            transitions=list(data.get("transitions", [])),
        )

    def save(self) -> None:
        write_json(
            self.path,
            {"current": self.current, "transitions": self.transitions},
        )

    def advance(self, stage: str) -> None:
        """Advance to ``stage``, which must be the immediate next stage.

        Re-entering the current stage is allowed (idempotent re-runs of a
        stage), but jumping ahead or moving backwards raises ``StageError``.
        """
        if stage not in _STAGE_INDEX:
            raise StageError(f"Unknown stage: {stage!r}")

        target = _STAGE_INDEX[stage]
        cur = _STAGE_INDEX[self.current]

        if target == cur:
            # Idempotent: record the repeated transition timestamp.
            self.transitions.append({"stage": stage, "timestamp": now_iso()})
            self.save()
            return

        if target != cur + 1:
            raise StageError(
                f"Illegal stage transition {self.current!r} -> {stage!r}; "
                f"expected next stage {STAGES[cur + 1]!r}"
                if cur + 1 < len(STAGES)
                else f"Illegal stage transition {self.current!r} -> {stage!r}; "
                "pipeline already at final stage"
            )

        self.current = stage
        self.transitions.append({"stage": stage, "timestamp": now_iso()})
        self.save()

    def require(self, stage: str) -> None:
        """Assert that the pipeline has reached at least ``stage``."""
        if _STAGE_INDEX[self.current] < _STAGE_INDEX[stage]:
            raise StageError(
                f"Stage {stage!r} required, but pipeline is at {self.current!r}"
            )

    def timestamp_of(self, stage: str) -> Optional[str]:
        """Return the timestamp of the first transition into ``stage``."""
        for t in self.transitions:
            if t.get("stage") == stage:
                return t.get("timestamp")
        return None
