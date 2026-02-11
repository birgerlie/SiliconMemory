"""Decision record types for SM-1: Decision Records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now


class DecisionStatus(str, Enum):
    """Status of a decision in its lifecycle."""

    ACTIVE = "active"
    REVISIT_SUGGESTED = "revisit_suggested"
    REVISED = "revised"
    SUPERSEDED = "superseded"


@dataclass
class Assumption:
    """An assumption underlying a decision."""

    belief_id: UUID
    description: str
    confidence_at_decision: float
    is_critical: bool = False


@dataclass
class Alternative:
    """An alternative considered but not chosen."""

    title: str
    description: str
    rejection_reason: str = ""
    beliefs_supporting: list[UUID] = field(default_factory=list)
    beliefs_against: list[UUID] = field(default_factory=list)


@dataclass
class Decision:
    """A decision record capturing what was decided, why, and based on what assumptions."""

    id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: str = ""
    decided_at: datetime = field(default_factory=utc_now)
    decided_by: str | None = None
    session_id: str | None = None
    belief_snapshot_id: str | None = None

    assumptions: list[Assumption] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)

    status: DecisionStatus = DecisionStatus.ACTIVE

    outcome: str | None = None
    outcome_recorded_at: datetime | None = None

    revision_of: UUID | None = None

    node_type: str = "decision"

    # Multi-user security fields
    user_id: str | None = None
    tenant_id: str | None = None

    # Metadata
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "decided_at": self.decided_at.isoformat(),
            "decided_by": self.decided_by,
            "session_id": self.session_id,
            "belief_snapshot_id": self.belief_snapshot_id,
            "assumptions": [
                {
                    "belief_id": str(a.belief_id),
                    "description": a.description,
                    "confidence_at_decision": a.confidence_at_decision,
                    "is_critical": a.is_critical,
                }
                for a in self.assumptions
            ],
            "alternatives": [
                {
                    "title": alt.title,
                    "description": alt.description,
                    "rejection_reason": alt.rejection_reason,
                    "beliefs_supporting": [str(b) for b in alt.beliefs_supporting],
                    "beliefs_against": [str(b) for b in alt.beliefs_against],
                }
                for alt in self.alternatives
            ],
            "status": self.status.value,
            "outcome": self.outcome,
            "outcome_recorded_at": self.outcome_recorded_at.isoformat() if self.outcome_recorded_at else None,
            "revision_of": str(self.revision_of) if self.revision_of else None,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        """Deserialize from dictionary."""
        from datetime import datetime as dt

        assumptions = [
            Assumption(
                belief_id=UUID(a["belief_id"]),
                description=a["description"],
                confidence_at_decision=a["confidence_at_decision"],
                is_critical=a.get("is_critical", False),
            )
            for a in data.get("assumptions", [])
        ]

        alternatives = [
            Alternative(
                title=alt["title"],
                description=alt["description"],
                rejection_reason=alt.get("rejection_reason", ""),
                beliefs_supporting=[UUID(b) for b in alt.get("beliefs_supporting", [])],
                beliefs_against=[UUID(b) for b in alt.get("beliefs_against", [])],
            )
            for alt in data.get("alternatives", [])
        ]

        decided_at = data.get("decided_at")
        if isinstance(decided_at, str):
            decided_at = dt.fromisoformat(decided_at)
        elif decided_at is None:
            decided_at = utc_now()

        outcome_at = data.get("outcome_recorded_at")
        if isinstance(outcome_at, str):
            outcome_at = dt.fromisoformat(outcome_at)

        revision_of = data.get("revision_of")
        if isinstance(revision_of, str):
            revision_of = UUID(revision_of)

        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            title=data.get("title", ""),
            description=data.get("description", ""),
            decided_at=decided_at,
            decided_by=data.get("decided_by"),
            session_id=data.get("session_id"),
            belief_snapshot_id=data.get("belief_snapshot_id"),
            assumptions=assumptions,
            alternatives=alternatives,
            status=DecisionStatus(data.get("status", "active")),
            outcome=data.get("outcome"),
            outcome_recorded_at=outcome_at,
            revision_of=revision_of,
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
            user_id=data.get("user_id"),
            tenant_id=data.get("tenant_id"),
        )
