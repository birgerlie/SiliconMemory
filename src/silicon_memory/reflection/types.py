"""Types for the reflection engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now


class PatternType(str, Enum):
    """Types of patterns extracted from experiences."""

    CAUSAL = "causal"  # A causes B
    TEMPORAL = "temporal"  # A followed by B
    CORRELATION = "correlation"  # A often occurs with B
    GENERALIZATION = "generalization"  # All X have property Y
    PREFERENCE = "preference"  # User prefers X over Y
    PROCEDURE = "procedure"  # Steps to achieve X
    FACT = "fact"  # Simple factual observation


@dataclass
class Pattern:
    """A pattern extracted from experiences.

    Patterns represent recurring structures or relationships
    observed across multiple experiences.
    """

    id: UUID = field(default_factory=uuid4)
    type: PatternType = PatternType.FACT
    description: str = ""
    evidence: list[UUID] = field(default_factory=list)  # Experience IDs
    confidence: float = 0.5
    occurrences: int = 1
    first_observed: datetime = field(default_factory=utc_now)
    last_observed: datetime = field(default_factory=utc_now)

    # Pattern-specific data
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def strength(self) -> float:
        """Calculate pattern strength based on occurrences and recency."""
        # More occurrences = stronger pattern
        occurrence_factor = min(1.0, self.occurrences / 10)
        # More confident = stronger
        return self.confidence * (0.5 + 0.5 * occurrence_factor)

    def as_triplet_dict(self) -> dict[str, str] | None:
        """Return as triplet dict if applicable."""
        if self.subject and self.predicate and self.object:
            return {
                "subject": self.subject,
                "predicate": self.predicate,
                "object": self.object,
            }
        return None


@dataclass
class BeliefCandidate:
    """A candidate belief generated from patterns.

    Before becoming a full Belief, candidates are validated
    against existing knowledge for contradictions.
    """

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = 0.5
    source_patterns: list[UUID] = field(default_factory=list)
    source_experiences: list[UUID] = field(default_factory=list)
    reasoning: str = ""  # Why this belief was generated

    # Validation results (filled during validation)
    contradicts: list[UUID] = field(default_factory=list)  # Contradicting belief IDs
    supports: list[UUID] = field(default_factory=list)  # Supporting belief IDs
    is_novel: bool = True  # No existing similar belief

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def has_triplet(self) -> bool:
        """Check if this candidate has triplet form."""
        return bool(self.subject and self.predicate and self.object)

    @property
    def is_contested(self) -> bool:
        """Check if this candidate has contradictions."""
        return len(self.contradicts) > 0


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""

    new_beliefs: list[BeliefCandidate] = field(default_factory=list)
    updated_beliefs: list[tuple[UUID, float]] = field(default_factory=list)  # (id, new_confidence)
    contradictions: list[tuple[UUID, UUID]] = field(default_factory=list)
    patterns_found: list[Pattern] = field(default_factory=list)
    experiences_processed: int = 0
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def total_changes(self) -> int:
        """Total number of changes made."""
        return len(self.new_beliefs) + len(self.updated_beliefs)

    def summary(self) -> str:
        """Generate a summary of the reflection."""
        return (
            f"Reflection completed at {self.timestamp.isoformat()}\n"
            f"  Experiences processed: {self.experiences_processed}\n"
            f"  Patterns found: {len(self.patterns_found)}\n"
            f"  New beliefs: {len(self.new_beliefs)}\n"
            f"  Updated beliefs: {len(self.updated_beliefs)}\n"
            f"  Contradictions: {len(self.contradictions)}"
        )


@dataclass
class ReflectionConfig:
    """Configuration for the reflection engine."""

    # Processing limits
    max_experiences_per_batch: int = 100
    min_pattern_occurrences: int = 2
    min_confidence_threshold: float = 0.5

    # Pattern extraction
    enable_causal_patterns: bool = True
    enable_temporal_patterns: bool = True
    enable_correlation_patterns: bool = True
    enable_generalizations: bool = True

    # Belief generation
    auto_commit_beliefs: bool = False  # Require manual approval
    max_beliefs_per_cycle: int = 50
    require_multiple_sources: bool = True  # Need 2+ experiences for belief

    # LLM settings (for advanced pattern/belief extraction)
    use_llm: bool = False
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.3


@dataclass
class ExperienceGroup:
    """A group of related experiences for pattern extraction."""

    id: UUID = field(default_factory=uuid4)
    experiences: list[UUID] = field(default_factory=list)
    common_entities: set[str] = field(default_factory=set)
    common_tags: set[str] = field(default_factory=set)
    session_id: str | None = None
    time_span: tuple[datetime, datetime] | None = None

    @property
    def size(self) -> int:
        """Number of experiences in the group."""
        return len(self.experiences)
