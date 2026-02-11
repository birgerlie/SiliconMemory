"""Query tool for knowledge queries with proofs and references."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.core.types import Belief, Source

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class QueryFormat(str, Enum):
    """Output format for query responses."""

    FULL = "full"
    SUMMARY = "summary"
    REPORT = "report"
    CITATIONS = "citations"


@dataclass
class BeliefSummary:
    """Summary of a belief for query responses."""

    id: str
    content: str
    confidence: float
    source: str | None
    observed_at: datetime | None
    is_valid: bool
    evidence_for: int
    evidence_against: int
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "source": self.source,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "is_valid": self.is_valid,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "tags": self.tags,
        }


@dataclass
class ContradictionSummary:
    """Summary of a contradiction between beliefs."""

    belief1_id: str
    belief1_content: str
    belief1_confidence: float
    belief2_id: str
    belief2_content: str
    belief2_confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "belief1": {
                "id": self.belief1_id,
                "content": self.belief1_content,
                "confidence": self.belief1_confidence,
            },
            "belief2": {
                "id": self.belief2_id,
                "content": self.belief2_content,
                "confidence": self.belief2_confidence,
            },
        }


@dataclass
class SourceSummary:
    """Summary of a source."""

    id: str
    name: str
    type: str
    url: str | None
    reliability: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "reliability": self.reliability,
        }


@dataclass
class QueryResponse:
    """Response from a knowledge query with proofs."""

    query: str
    timestamp: datetime = field(default_factory=utc_now)
    total_confidence: float = 0.0
    beliefs: list[BeliefSummary] = field(default_factory=list)
    sources: list[SourceSummary] = field(default_factory=list)
    contradictions: list[ContradictionSummary] = field(default_factory=list)
    report: str = ""

    @property
    def has_contradictions(self) -> bool:
        """Check if there are contradictions."""
        return len(self.contradictions) > 0

    @property
    def belief_count(self) -> int:
        """Get number of beliefs."""
        return len(self.beliefs)

    @property
    def high_confidence_beliefs(self) -> list[BeliefSummary]:
        """Get beliefs with confidence >= 0.7."""
        return [b for b in self.beliefs if b.confidence >= 0.7]

    def to_dict(self, format: QueryFormat = QueryFormat.FULL) -> dict[str, Any]:
        """Convert to dictionary in specified format."""
        if format == QueryFormat.REPORT:
            return {"query": self.query, "report": self.report}

        if format == QueryFormat.CITATIONS:
            return {
                "query": self.query,
                "sources": [s.to_dict() for s in self.sources],
                "belief_count": self.belief_count,
            }

        if format == QueryFormat.SUMMARY:
            return {
                "query": self.query,
                "total_confidence": self.total_confidence,
                "belief_count": self.belief_count,
                "source_count": len(self.sources),
                "has_contradictions": self.has_contradictions,
                "high_confidence_count": len(self.high_confidence_beliefs),
            }

        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "total_confidence": self.total_confidence,
            "beliefs": [b.to_dict() for b in self.beliefs],
            "sources": [s.to_dict() for s in self.sources],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "report": self.report,
        }

    def as_context_string(self) -> str:
        """Format as context string for LLM prompts."""
        lines = [f"Knowledge about: {self.query}"]
        lines.append(f"Confidence: {self.total_confidence:.0%}")
        lines.append("")

        if self.beliefs:
            lines.append("Facts:")
            for b in self.beliefs[:10]:
                confidence_bar = "●" * int(b.confidence * 5) + "○" * (5 - int(b.confidence * 5))
                source_note = f" [{b.source}]" if b.source else ""
                lines.append(f"  [{confidence_bar}] {b.content}{source_note}")

        if self.contradictions:
            lines.append("")
            lines.append("⚠️ Contradictions detected:")
            for c in self.contradictions[:3]:
                lines.append(f"  - {c.belief1_content} vs {c.belief2_content}")

        return "\n".join(lines)


class QueryTool:
    """Tool for querying knowledge with proofs and references.

    Example:
        >>> from silicon_memory import SiliconMemory, QueryTool
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     tool = QueryTool(memory)
        ...     response = await tool.query("Python")
        ...     print(response.report)
    """

    def __init__(self, memory: "SiliconMemory") -> None:
        self._memory = memory

    async def query(
        self,
        query: str,
        min_confidence: float = 0.3,
        include_low_confidence: bool = False,
        max_beliefs: int = 50,
    ) -> QueryResponse:
        """Query knowledge with full proofs."""
        proof = await self._memory.what_do_you_know(
            query,
            min_confidence=min_confidence if not include_low_confidence else 0.0,
        )

        response = QueryResponse(
            query=query,
            total_confidence=proof.total_confidence,
            report=proof.as_report(),
        )

        for belief in proof.beliefs[:max_beliefs]:
            is_valid = proof.temporal_validity.get(belief.id, True)
            evidence = proof.evidence_summary.get(belief.id, {"for": 0, "against": 0})

            summary = BeliefSummary(
                id=str(belief.id),
                content=belief.content or (
                    belief.triplet.as_text() if belief.triplet else ""
                ),
                confidence=belief.confidence,
                source=belief.source.name if belief.source else None,
                observed_at=belief.temporal.observed_at if belief.temporal else None,
                is_valid=is_valid,
                evidence_for=evidence["for"],
                evidence_against=evidence["against"],
                tags=belief.tags,
            )
            response.beliefs.append(summary)

        for source in proof.sources:
            summary = SourceSummary(
                id=source.id,
                name=source.name,
                type=source.type,
                url=source.url,
                reliability=source.reliability,
            )
            response.sources.append(summary)

        for b1, b2 in proof.contradictions:
            summary = ContradictionSummary(
                belief1_id=str(b1.id),
                belief1_content=b1.content or (
                    b1.triplet.as_text() if b1.triplet else ""
                ),
                belief1_confidence=b1.confidence,
                belief2_id=str(b2.id),
                belief2_content=b2.content or (
                    b2.triplet.as_text() if b2.triplet else ""
                ),
                belief2_confidence=b2.confidence,
            )
            response.contradictions.append(summary)

        return response

    async def query_entity(
        self,
        entity: str,
        min_confidence: float = 0.3,
    ) -> QueryResponse:
        """Query all knowledge about a specific entity."""
        # Use the backend directly to query by entity
        beliefs = await self._memory._backend.get_beliefs_by_entity(entity)
        filtered = [b for b in beliefs if b.confidence >= min_confidence]

        response = QueryResponse(
            query=f"Entity: {entity}",
            total_confidence=(
                sum(b.confidence for b in filtered) / len(filtered)
                if filtered else 0.0
            ),
        )

        for belief in filtered:
            summary = BeliefSummary(
                id=str(belief.id),
                content=belief.content or (
                    belief.triplet.as_text() if belief.triplet else ""
                ),
                confidence=belief.confidence,
                source=belief.source.name if belief.source else None,
                observed_at=belief.temporal.observed_at if belief.temporal else None,
                is_valid=True,
                evidence_for=len(belief.evidence_for),
                evidence_against=len(belief.evidence_against),
                tags=belief.tags,
            )
            response.beliefs.append(summary)

            if belief.source and not any(
                s.id == belief.source.id for s in response.sources
            ):
                response.sources.append(SourceSummary(
                    id=belief.source.id,
                    name=belief.source.name,
                    type=belief.source.type,
                    url=belief.source.url,
                    reliability=belief.source.reliability,
                ))

        response.report = self._build_entity_report(entity, response)
        return response

    async def verify_claim(
        self,
        claim: str,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Verify a claim against stored knowledge."""
        response = await self.query(claim, min_confidence=0.0)

        supporting = [b for b in response.beliefs if b.confidence >= min_confidence]
        weak = [b for b in response.beliefs if 0.0 < b.confidence < min_confidence]

        if not response.beliefs:
            verification_score = 0.0
            status = "unknown"
        elif response.has_contradictions:
            verification_score = 0.5
            status = "contested"
        elif supporting:
            verification_score = sum(b.confidence for b in supporting) / len(supporting)
            status = "supported" if verification_score >= 0.7 else "weakly_supported"
        else:
            verification_score = 0.0
            status = "unsupported"

        return {
            "claim": claim,
            "status": status,
            "verification_score": verification_score,
            "supporting_beliefs": len(supporting),
            "weak_beliefs": len(weak),
            "contradictions": len(response.contradictions),
            "sources": [s.to_dict() for s in response.sources],
        }

    def _build_entity_report(
        self,
        entity: str,
        response: QueryResponse,
    ) -> str:
        """Build a report for entity query."""
        lines = [f"# Knowledge about: {entity}"]
        lines.append(f"Total beliefs: {len(response.beliefs)}")
        lines.append(f"Average confidence: {response.total_confidence:.0%}")
        lines.append("")

        if response.beliefs:
            lines.append("## Facts")
            for b in response.beliefs:
                lines.append(f"- [{b.confidence:.0%}] {b.content}")
                if b.source:
                    lines.append(f"  Source: {b.source}")

        if response.sources:
            lines.append("")
            lines.append("## Sources")
            for s in response.sources:
                lines.append(f"- {s.name} ({s.type})")

        return "\n".join(lines)
