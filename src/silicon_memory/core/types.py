"""Core data types for Silicon Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now

if TYPE_CHECKING:
    from silicon_memory.security.types import PrivacyMetadata


class SourceType(Enum):
    """Type of knowledge source."""

    OBSERVATION = "observation"  # Direct experience
    REFLECTION = "reflection"  # Derived by reflection agent
    EXTERNAL = "external"  # From external source (API, KB)
    HUMAN = "human"  # Human provided/verified
    BOOTSTRAP = "bootstrap"  # Initial ingestion


class BeliefStatus(Enum):
    """Status of a belief in its lifecycle."""

    PROVISIONAL = "provisional"  # Needs validation
    VALIDATED = "validated"  # Confirmed by evidence
    CONTESTED = "contested"  # Conflicting evidence exists
    REJECTED = "rejected"  # Disproven
    EXPIRED = "expired"  # TTL passed without reconfirmation


@dataclass(frozen=True)
class Triplet:
    """Immutable subject-predicate-object triplet."""

    subject: str
    predicate: str
    object: str

    def as_text(self) -> str:
        """Return human-readable representation."""
        return f"({self.subject}, {self.predicate}, {self.object})"

    def as_dict(self) -> dict[str, str]:
        """Return as dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass
class Source:
    """Provenance information for a piece of knowledge."""

    id: str
    type: SourceType
    reliability: float = 0.5  # 0.0 - 1.0
    retrieved_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Reliability must be between 0.0 and 1.0")


@dataclass
class TemporalContext:
    """Temporal metadata for time-aware knowledge."""

    observed_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    last_verified: datetime | None = None
    ttl_seconds: int | None = None

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if the knowledge is valid at a given time."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True

    def is_expired(self, current_time: datetime) -> bool:
        """Check if TTL has expired."""
        if self.ttl_seconds is None:
            return False
        if self.last_verified is None:
            reference = self.observed_at
        else:
            reference = self.last_verified
        elapsed = (current_time - reference).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass
class Belief:
    """A belief with confidence, provenance, and temporal context."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    triplet: Triplet | None = None
    confidence: float = 0.5
    source: Source | None = None
    status: BeliefStatus = BeliefStatus.PROVISIONAL
    temporal: TemporalContext | None = None

    # Evidence tracking
    evidence_for: list[UUID] = field(default_factory=list)
    evidence_against: list[UUID] = field(default_factory=list)
    falsifiable_by: str | None = None

    # Metadata
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Multi-user security fields
    user_id: str | None = None
    tenant_id: str | None = None
    privacy: "PrivacyMetadata | None" = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def evidence_count(self) -> int:
        """Total evidence count (for and against)."""
        return len(self.evidence_for) + len(self.evidence_against)

    @property
    def evidence_ratio(self) -> float:
        """Ratio of supporting evidence (0.0 to 1.0)."""
        total = self.evidence_count
        if total == 0:
            return 0.5
        return len(self.evidence_for) / total


@dataclass
class Experience:
    """An episodic memory - a recorded experience/event."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    occurred_at: datetime = field(default_factory=utc_now)
    outcome: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    # Processing state
    processed: bool = False  # Has reflection agent processed this?

    # Sequence information
    sequence_id: int | None = None  # Order within session
    session_id: str | None = None
    causal_parent: UUID | None = None  # What caused this experience

    # Metadata
    tags: set[str] = field(default_factory=set)

    # Multi-user security fields
    user_id: str | None = None
    tenant_id: str | None = None
    privacy: "PrivacyMetadata | None" = None


@dataclass
class Procedure:
    """Procedural knowledge - how to do something."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    trigger: str = ""  # When to apply this procedure
    steps: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source: Source | None = None

    # Outcome tracking
    success_count: int = 0
    failure_count: int = 0

    # Metadata
    tags: set[str] = field(default_factory=set)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    # Multi-user security fields
    user_id: str | None = None
    tenant_id: str | None = None
    privacy: "PrivacyMetadata | None" = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def success_rate(self) -> float:
        """Calculate success rate from tracked outcomes."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown
        return self.success_count / total

    def record_outcome(self, success: bool) -> None:
        """Record an outcome and update confidence."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        # Update confidence based on outcomes
        self.confidence = 0.3 + (0.6 * self.success_rate)


@dataclass
class RecallResult:
    """Result from memory recall with relevance scoring."""

    content: str
    confidence: float
    source: Source | None
    memory_type: str  # "semantic" | "episodic" | "procedural" | "working"
    relevance_score: float
    temporal: TemporalContext | None = None

    # Reference information for "what do you know" queries
    belief_id: UUID | None = None
    triplet: Triplet | None = None
    evidence_count: int = 0
    entropy: float = 0.0

    def as_citation(self) -> str:
        """Format as a citation string."""
        source_str = f"[{self.source.id}]" if self.source else "[unknown]"
        conf_str = f"{self.confidence:.0%}"
        return f"{self.content} ({conf_str} confidence, {source_str})"


@dataclass
class KnowledgeProof:
    """Proof of knowledge for 'what do you know' queries."""

    query: str
    beliefs: list[Belief]
    total_confidence: float
    sources: list[Source]
    contradictions: list[tuple[Belief, Belief]]
    temporal_validity: dict[UUID, bool]  # belief_id -> is_valid_now
    evidence_summary: dict[UUID, dict[str, int]]  # belief_id -> {for: n, against: m}

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def as_report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            f"Knowledge Report: {self.query}",
            f"=" * 50,
            f"Total beliefs found: {len(self.beliefs)}",
            f"Overall confidence: {self.total_confidence:.0%}",
            f"Sources: {self.source_count}",
            f"Contradictions: {len(self.contradictions)}",
            "",
            "BELIEFS:",
        ]

        for belief in sorted(self.beliefs, key=lambda b: b.confidence, reverse=True):
            status = "✓" if belief.status == BeliefStatus.VALIDATED else "?"
            lines.append(
                f"  {status} [{belief.confidence:.0%}] "
                f"{belief.content or (belief.triplet.as_text() if belief.triplet else 'N/A')}"
            )
            if belief.source:
                lines.append(f"      Source: {belief.source.id} ({belief.source.type.value})")
            evidence = self.evidence_summary.get(belief.id, {})
            if evidence:
                lines.append(
                    f"      Evidence: {evidence.get('for', 0)} for, "
                    f"{evidence.get('against', 0)} against"
                )

        if self.contradictions:
            lines.append("")
            lines.append("CONTRADICTIONS:")
            for b1, b2 in self.contradictions:
                lines.append(f"  ⚠ {b1.content} vs {b2.content}")

        return "\n".join(lines)
