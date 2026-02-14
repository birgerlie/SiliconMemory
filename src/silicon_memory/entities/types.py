"""Data types for the entity resolution system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Candidate:
    """A potential entity reference found by a detector (pass 1)."""

    text: str
    span: tuple[int, int]
    context_text: str
    detector_id: str


@dataclass
class EntityReference:
    """A resolved entity reference (pass 2+3 output)."""

    text: str
    canonical_id: str
    entity_type: str
    confidence: float
    span: tuple[int, int]
    context_text: str
    rule_id: str | None = None


@dataclass
class ResolveResult:
    """Result of entity resolution — matches adapter interface."""

    resolved: list[EntityReference] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


@dataclass
class DetectorRule:
    """Broad regex pattern for pass 1 (candidate detection)."""

    id: str
    pattern: str
    description: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExtractorRule:
    """Precise regex pattern for pass 2 (type-specific extraction)."""

    id: str
    entity_type: str
    detector_ids: list[str]
    pattern: str
    normalize_template: str
    examples: list[str] = field(default_factory=list)
    context_examples: list[str] = field(default_factory=list)
    context_embedding: list[float] | None = None
    confidence: float = 1.0
    context_threshold: float = 0.6
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EntityEntry:
    """A canonical entity with its aliases."""

    canonical_id: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
