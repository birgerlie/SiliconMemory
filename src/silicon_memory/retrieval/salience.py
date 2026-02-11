"""Salience-weighted retrieval profiles.

Each profile defines weights for combining different retrieval signals
into a single salience score. Profiles are optimized for different
use cases like decision support, exploration, and context recall.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SalienceProfile:
    """Defines how to weight different retrieval signals.

    All weights should sum to approximately 1.0 for interpretability,
    but this is not enforced — they are normalized internally.

    Attributes:
        vector_weight: Weight for embedding similarity
        text_weight: Weight for BM25 keyword match
        temporal_weight: Weight for recency
        temporal_half_life_days: Half-life for temporal decay
        confidence_weight: Weight for belief confidence
        graph_proximity_weight: Weight for graph distance to context
        entropy_weight: Weight for belief entropy/uncertainty
        entropy_direction: "prefer_low" for certain facts, "prefer_high" for exploration
    """

    vector_weight: float = 0.3
    text_weight: float = 0.1
    temporal_weight: float = 0.15
    temporal_half_life_days: float = 30.0
    confidence_weight: float = 0.2
    graph_proximity_weight: float = 0.15
    entropy_weight: float = 0.1
    entropy_direction: str = "prefer_low"

    def __post_init__(self) -> None:
        if self.entropy_direction not in ("prefer_low", "prefer_high"):
            raise ValueError(
                f"entropy_direction must be 'prefer_low' or 'prefer_high', "
                f"got '{self.entropy_direction}'"
            )

    def to_search_weights(self) -> dict[str, float]:
        """Convert to a dictionary suitable for SiliconDB search configuration.

        Returns:
            Dictionary mapping weight names to values
        """
        return {
            "vector": self.vector_weight,
            "text": self.text_weight,
            "temporal": self.temporal_weight,
            "temporal_half_life_hours": self.temporal_half_life_days * 24,
            "confidence": self.confidence_weight,
            "graph_proximity": self.graph_proximity_weight,
        }

    @property
    def total_weight(self) -> float:
        """Sum of all weights (for normalization)."""
        return (
            self.vector_weight
            + self.text_weight
            + self.temporal_weight
            + self.confidence_weight
            + self.graph_proximity_weight
            + self.entropy_weight
        )


# Preset profiles optimized for different use cases
PROFILES: dict[str, SalienceProfile] = {
    "decision_support": SalienceProfile(
        vector_weight=0.3,
        text_weight=0.1,
        temporal_weight=0.15,
        temporal_half_life_days=30,
        confidence_weight=0.25,
        graph_proximity_weight=0.15,
        entropy_weight=0.05,
        entropy_direction="prefer_low",
    ),
    "exploration": SalienceProfile(
        vector_weight=0.4,
        text_weight=0.2,
        temporal_weight=0.05,
        temporal_half_life_days=365,
        confidence_weight=0.1,
        graph_proximity_weight=0.1,
        entropy_weight=0.15,
        entropy_direction="prefer_high",
    ),
    "context_recall": SalienceProfile(
        vector_weight=0.2,
        text_weight=0.1,
        temporal_weight=0.3,
        temporal_half_life_days=7,
        confidence_weight=0.1,
        graph_proximity_weight=0.25,
        entropy_weight=0.05,
        entropy_direction="prefer_low",
    ),
}
