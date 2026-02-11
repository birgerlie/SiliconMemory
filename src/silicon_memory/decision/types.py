"""Types for decision synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class EvidencedClaim:
    """A claim backed by a specific belief."""

    claim: str
    belief_id: UUID
    confidence: float
    evidence_count: int = 0
    source_description: str = ""


@dataclass
class Option:
    """An option in a decision."""

    title: str
    description: str = ""
    supporting_evidence: list[EvidencedClaim] = field(default_factory=list)
    opposing_evidence: list[EvidencedClaim] = field(default_factory=list)
    risks: list["Risk"] = field(default_factory=list)
    estimated_confidence: float = 0.5


@dataclass
class Risk:
    """A risk associated with a decision option."""

    description: str
    severity: str = "medium"  # low, medium, high, critical
    likelihood: str = "medium"  # low, medium, high
    related_beliefs: list[UUID] = field(default_factory=list)


@dataclass
class Uncertainty:
    """An area of uncertainty relevant to the decision."""

    description: str
    belief_id: UUID | None = None
    entropy: float = 0.0
    impact: str = "medium"  # low, medium, high


@dataclass
class Precedent:
    """A past decision relevant as precedent."""

    decision_id: UUID
    title: str
    outcome: str | None = None
    relevance_score: float = 0.0


@dataclass
class DecisionBrief:
    """Structured decision support brief.

    All claims are linked back to specific beliefs for traceability.
    """

    question: str
    summary: str = ""
    options: list[Option] = field(default_factory=list)
    key_beliefs: list[EvidencedClaim] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)
    uncertainties: list[Uncertainty] = field(default_factory=list)
    past_precedents: list[Precedent] = field(default_factory=list)
    recommendation: str = ""
    confidence_in_recommendation: float = 0.0

    @property
    def has_contradictions(self) -> bool:
        """Whether any options have opposing evidence."""
        return any(opt.opposing_evidence for opt in self.options)

    @property
    def total_evidence(self) -> int:
        """Total number of evidence items across all options."""
        total = len(self.key_beliefs)
        for opt in self.options:
            total += len(opt.supporting_evidence) + len(opt.opposing_evidence)
        return total

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "summary": self.summary,
            "options": [
                {
                    "title": o.title,
                    "description": o.description,
                    "estimated_confidence": o.estimated_confidence,
                    "supporting_count": len(o.supporting_evidence),
                    "opposing_count": len(o.opposing_evidence),
                    "risk_count": len(o.risks),
                }
                for o in self.options
            ],
            "key_belief_count": len(self.key_beliefs),
            "risk_count": len(self.risks),
            "uncertainty_count": len(self.uncertainties),
            "precedent_count": len(self.past_precedents),
            "recommendation": self.recommendation,
            "confidence_in_recommendation": self.confidence_in_recommendation,
            "has_contradictions": self.has_contradictions,
            "total_evidence": self.total_evidence,
        }
