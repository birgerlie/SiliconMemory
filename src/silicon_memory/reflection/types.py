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

    FACT = "fact"  # Factual triplet (subject-predicate-object)
    RELATIONSHIP = "relationship"  # Person-to-person or person-to-institution link
    ARGUMENT = "argument"  # Legal/logical argument with rhetoric classification
    TIMELINE_EVENT = "timeline_event"  # Dated event with actors and significance
    CAUSAL = "causal"  # A causes B
    TEMPORAL = "temporal"  # A followed by B
    CORRELATION = "correlation"  # A often occurs with B
    GENERALIZATION = "generalization"  # All X have property Y
    PREFERENCE = "preference"  # User prefers X over Y
    PROCEDURE = "procedure"  # Steps to achieve X


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
    source_context: dict[str, Any] = field(default_factory=dict)  # document provenance
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
class ConsolidationResult:
    """Result of Monte Carlo belief consolidation."""

    cooccurrence_links: int = 0
    mc_updates: int = 0  # beliefs updated by Monte Carlo
    propagations: int = 0  # beliefs affected by propagation
    mc_contradictions: int = 0  # MC-detected contradictions
    triple_contradictions: int = 0  # structural triple contradictions
    uncertain_beliefs: int = 0  # high-entropy beliefs found


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""

    new_beliefs: list[BeliefCandidate] = field(default_factory=list)
    updated_beliefs: list[tuple[UUID, float]] = field(default_factory=list)  # (id, new_confidence)
    contradictions: list[tuple[UUID, UUID]] = field(default_factory=list)
    patterns_found: list[Pattern] = field(default_factory=list)
    experiences_processed: int = 0
    consolidation: ConsolidationResult = field(default_factory=ConsolidationResult)
    timings: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def total_changes(self) -> int:
        """Total number of changes made."""
        return len(self.new_beliefs) + len(self.updated_beliefs)

    def summary(self) -> str:
        """Generate a summary of the reflection."""
        lines = [
            f"Reflection completed at {self.timestamp.isoformat()}",
            f"  Experiences processed: {self.experiences_processed}",
            f"  Patterns found: {len(self.patterns_found)}",
            f"  New beliefs: {len(self.new_beliefs)}",
            f"  Updated beliefs: {len(self.updated_beliefs)}",
            f"  Contradictions: {len(self.contradictions)}",
        ]
        c = self.consolidation
        if c.mc_updates or c.propagations or c.mc_contradictions:
            lines.append("  --- Monte Carlo Consolidation ---")
            lines.append(f"  Co-occurrence links: {c.cooccurrence_links}")
            lines.append(f"  MC probability updates: {c.mc_updates}")
            lines.append(f"  Beliefs propagated: {c.propagations}")
            lines.append(f"  MC contradictions: {c.mc_contradictions}")
            lines.append(f"  Triple contradictions: {c.triple_contradictions}")
            lines.append(f"  Uncertain beliefs: {c.uncertain_beliefs}")
        if self.timings:
            lines.append("  --- Timings (seconds) ---")
            for key in sorted(self.timings.keys()):
                lines.append(f"  {key}: {self.timings[key]:.3f}")
        return "\n".join(lines)


@dataclass
class ReflectionConfig:
    """Configuration for the reflection engine."""

    # Processing limits
    max_experiences_per_batch: int = 0  # 0 = unlimited
    min_confidence_threshold: float = 0.5

    # Belief generation
    auto_commit_beliefs: bool = False  # Require manual approval
    max_beliefs_per_cycle: int = 0  # 0 = unlimited; keep all extracted beliefs
    require_multiple_sources: bool = False  # Single LLM extraction is sufficient

    # Consolidation bounds
    max_consolidation_nodes: int = 0  # 0 = unlimited

    # LLM extraction settings
    llm_temperature: float = 0.3
    extraction_chunk_size: int = 30  # experiences per LLM call (pack many docs)
    extraction_max_chars: int = 48000  # char limit per LLM call (~26-35 docs)
    extraction_max_items: int = 12  # hard cap per dimension to control token spend
    extraction_max_tokens: int = 2500  # cap extraction output size
    allow_weak_extraction_model: bool = False  # diagnostics-only override for qwen3-4b class
    empty_extracted_cooldown_s: float = 0.5  # skip repeated empty scans briefly
    enable_observation_consolidation: bool = False  # expensive; keep opt-in for dev speed

    # Dream-mode controls (token spend + runtime)
    dream_max_communities: int = 4
    dream_min_community_size: int = 3
    dream_enable_hypothesis_generation: bool = True
    dream_enable_procedure_detection: bool = False
    dream_enable_question_generation: bool = False
    dream_enable_entity_consolidation: bool = False
    dream_enable_predicate_consolidation: bool = False
    dream_enable_hypothesis_validation: bool = True
    dream_max_hypothesis_validations: int = 50
    dream_graph_call_timeout_s: float = 10.0


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
