"""Data types for context switch snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now

if TYPE_CHECKING:
    from silicon_memory.security.types import PrivacyMetadata


@dataclass
class ContextSnapshot:
    """A snapshot of the current working state for task switching.

    Captures working memory, recent experiences, and a summary
    so that context can be restored when resuming a task.
    """

    id: UUID = field(default_factory=uuid4)
    task_context: str = ""  # e.g. "project-alpha/auth-module"
    summary: str = ""  # LLM-generated or rule-based summary
    working_memory: dict[str, Any] = field(default_factory=dict)
    recent_experiences: list[UUID] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    session_id: str | None = None

    # Multi-user security fields
    user_id: str | None = None
    tenant_id: str | None = None
    privacy: "PrivacyMetadata | None" = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/LLM consumption."""
        return {
            "id": str(self.id),
            "task_context": self.task_context,
            "summary": self.summary,
            "working_memory": self.working_memory,
            "recent_experiences": [str(eid) for eid in self.recent_experiences],
            "next_steps": self.next_steps,
            "open_questions": self.open_questions,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
        }


@dataclass
class SnapshotConfig:
    """Configuration for snapshot creation."""

    max_recent_experiences: int = 20
    recent_hours: int = 24
    llm_temperature: float = 0.3
    fallback_summary_max_chars: int = 500
