"""Types for the graph query layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class TraversalDirection(str, Enum):
    """Direction for graph traversal."""

    OUTGOING = "outgoing"  # Follow edges going out
    INCOMING = "incoming"  # Follow edges coming in
    BOTH = "both"  # Follow edges in both directions


class NodeType(str, Enum):
    """Type of graph node."""

    ENTITY = "entity"
    BELIEF = "belief"
    EXPERIENCE = "experience"
    PROCEDURE = "procedure"


class EdgeType(str, Enum):
    """Common edge types in the knowledge graph."""

    # Semantic relationships
    IS_A = "is_a"
    HAS = "has"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"

    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONCURRENT_WITH = "concurrent_with"

    # Belief relationships
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"

    # Experience relationships
    TRIGGERED_BY = "triggered_by"
    RESULTED_IN = "resulted_in"

    # Generic
    CUSTOM = "custom"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    label: str
    type: NodeType = NodeType.ENTITY
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional references to memory objects
    belief_id: UUID | None = None
    experience_id: UUID | None = None
    procedure_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source: str  # Node ID
    target: str  # Node ID
    type: str  # Edge type (can be EdgeType.value or custom)
    weight: float = 1.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
            "confidence": self.confidence,
        }


@dataclass
class GraphPath:
    """A path through the knowledge graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of edges in the path."""
        return len(self.edges)

    @property
    def total_confidence(self) -> float:
        """Product of all edge confidences."""
        if not self.edges:
            return 1.0
        conf = 1.0
        for edge in self.edges:
            conf *= edge.confidence
        return conf

    @property
    def start_node(self) -> GraphNode | None:
        """First node in the path."""
        return self.nodes[0] if self.nodes else None

    @property
    def end_node(self) -> GraphNode | None:
        """Last node in the path."""
        return self.nodes[-1] if self.nodes else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "length": self.length,
            "total_confidence": self.total_confidence,
        }

    def as_text(self) -> str:
        """Format as human-readable text."""
        if not self.nodes:
            return "(empty path)"

        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node.label)
            if i < len(self.edges):
                parts.append(f" --[{self.edges[i].type}]--> ")

        return "".join(parts)


@dataclass
class EntityProfile:
    """Comprehensive profile of an entity in the knowledge graph."""

    entity: str
    node: GraphNode | None = None

    # Direct relationships
    direct_beliefs: list[dict[str, Any]] = field(default_factory=list)
    related_entities: list[tuple[str, str, float]] = field(default_factory=list)  # (entity, relation, confidence)

    # Graph structure
    incoming_edges: list[GraphEdge] = field(default_factory=list)
    outgoing_edges: list[GraphEdge] = field(default_factory=list)

    # Paths to important entities
    paths: list[GraphPath] = field(default_factory=list)

    # Contradictions involving this entity
    contradictions: list[tuple[str, str]] = field(default_factory=list)  # (belief1_content, belief2_content)

    # Statistics
    total_mentions: int = 0
    first_mentioned: datetime | None = None
    last_mentioned: datetime | None = None
    average_confidence: float = 0.0

    def summary(self) -> str:
        """Generate a summary of the entity profile."""
        lines = [
            f"Entity Profile: {self.entity}",
            "=" * 50,
            f"Total mentions: {self.total_mentions}",
            f"Average confidence: {self.average_confidence:.0%}",
            f"Direct beliefs: {len(self.direct_beliefs)}",
            f"Related entities: {len(self.related_entities)}",
            f"Contradictions: {len(self.contradictions)}",
        ]

        if self.first_mentioned:
            lines.append(f"First mentioned: {self.first_mentioned.isoformat()}")
        if self.last_mentioned:
            lines.append(f"Last mentioned: {self.last_mentioned.isoformat()}")

        if self.direct_beliefs:
            lines.append("")
            lines.append("Key beliefs:")
            for belief in self.direct_beliefs[:5]:
                content = belief.get("content", "")[:60]
                conf = belief.get("confidence", 0)
                lines.append(f"  - [{conf:.0%}] {content}")

        if self.related_entities:
            lines.append("")
            lines.append("Related entities:")
            for entity, relation, conf in self.related_entities[:5]:
                lines.append(f"  - {entity} ({relation}, {conf:.0%})")

        if self.contradictions:
            lines.append("")
            lines.append("Contradictions:")
            for b1, b2 in self.contradictions[:3]:
                lines.append(f"  - '{b1[:30]}' vs '{b2[:30]}'")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity": self.entity,
            "total_mentions": self.total_mentions,
            "average_confidence": self.average_confidence,
            "direct_beliefs": self.direct_beliefs,
            "related_entities": [
                {"entity": e, "relation": r, "confidence": c}
                for e, r, c in self.related_entities
            ],
            "incoming_edges": [e.to_dict() for e in self.incoming_edges],
            "outgoing_edges": [e.to_dict() for e in self.outgoing_edges],
            "paths": [p.to_dict() for p in self.paths],
            "contradictions": [
                {"belief1": b1, "belief2": b2}
                for b1, b2 in self.contradictions
            ],
        }


@dataclass
class QueryResult:
    """Result from a graph query."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "paths": [p.to_dict() for p in self.paths],
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms,
        }
