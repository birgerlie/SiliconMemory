"""Transitive inference engine for the reflection pipeline.

Discovers indirect relationships by chaining known facts:
- A→B + B→C  →  infer A→C (2-hop)
- A→B + B→C + C→D  →  infer A→D (3-hop)

Also supports backward dreaming: when new evidence arrives,
re-evaluate and update existing beliefs that share entities.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.reflection.types import (
    Pattern,
    PatternType,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of a transitive inference pass."""

    inferred_beliefs: list[Belief] = field(default_factory=list)
    chains_found: int = 0
    backward_updates: int = 0
    paths_explored: int = 0


@dataclass
class InferenceChain:
    """A chain of beliefs forming a transitive inference."""

    beliefs: list[Belief]
    hops: int
    confidence: float
    subject: str
    object: str
    path_description: str


class TransitiveInferenceEngine:
    """Discovers indirect relationships by chaining known beliefs.

    Implements two modes:
    - Forward dreaming: new beliefs → find chains → infer new facts
    - Backward dreaming: new beliefs → find related old beliefs → update them

    Uses SiliconDB graph traversal for efficient path finding.

    Example:
        >>> engine = TransitiveInferenceEngine(memory)
        >>> result = await engine.infer_forward(new_beliefs)
        >>> print(f"Found {result.chains_found} transitive chains")
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()

    async def infer_forward(
        self,
        new_beliefs: list[Belief],
        max_hops: int = 3,
        min_confidence: float = 0.3,
    ) -> InferenceResult:
        """Forward dreaming: find transitive chains from new beliefs.

        For each new belief with triplet (S, P, O):
        1. Find existing beliefs where S or O appears as subject/object
        2. Chain them to form multi-hop inferences
        3. Create new inferred beliefs with decayed confidence

        Args:
            new_beliefs: Recently committed beliefs to chain from
            max_hops: Maximum chain length (2-3 recommended)
            min_confidence: Minimum confidence for included beliefs

        Returns:
            InferenceResult with discovered chains and inferred beliefs
        """
        result = InferenceResult()

        # Build adjacency from new beliefs
        for belief in new_beliefs:
            if not belief.triplet:
                continue

            # Find 2-hop chains: new_belief.object == existing.subject
            chains = await self._find_chains_from(
                belief, max_hops, min_confidence
            )
            result.paths_explored += 1

            for chain in chains:
                result.chains_found += 1
                inferred = self._chain_to_belief(chain)
                if inferred:
                    result.inferred_beliefs.append(inferred)

        # Deduplicate inferred beliefs
        result.inferred_beliefs = self._deduplicate(result.inferred_beliefs)

        logger.info(
            "Forward inference: %d paths explored, %d chains, %d inferred beliefs",
            result.paths_explored, result.chains_found,
            len(result.inferred_beliefs),
        )
        return result

    async def infer_backward(
        self,
        new_beliefs: list[Belief],
        min_confidence: float = 0.3,
    ) -> InferenceResult:
        """Backward dreaming: update existing beliefs based on new evidence.

        When new beliefs arrive:
        1. Find existing beliefs that share entities
        2. If new belief corroborates → boost existing confidence
        3. If new belief contradicts → flag for review
        4. If new belief adds context → enrich existing metadata

        Args:
            new_beliefs: Recently committed beliefs
            min_confidence: Minimum confidence for lookups

        Returns:
            InferenceResult with backward_updates count
        """
        result = InferenceResult()
        backend = self._memory._backend

        for belief in new_beliefs:
            if not belief.triplet:
                continue

            # Find existing beliefs about the same subject
            try:
                related = await backend.get_beliefs_by_entity(
                    belief.triplet.subject
                )
            except Exception:
                continue

            for existing in related:
                if not existing.triplet or existing.id == belief.id:
                    continue

                # Same subject + same predicate = corroboration
                if (existing.triplet.subject.lower() == belief.triplet.subject.lower()
                        and existing.triplet.predicate.lower() == belief.triplet.predicate.lower()):
                    if existing.triplet.object.lower() == belief.triplet.object.lower():
                        # Corroboration: boost confidence via Bayesian observation
                        try:
                            await backend.update_belief_confidence(
                                existing.id, delta=0.1
                            )
                            result.backward_updates += 1
                        except Exception:
                            pass
                    else:
                        # Potential contradiction: record negative observation
                        try:
                            await backend.update_belief_confidence(
                                existing.id, delta=-0.05
                            )
                            result.backward_updates += 1
                        except Exception:
                            pass

            # Also find beliefs about the object
            try:
                related_obj = await backend.get_beliefs_by_entity(
                    belief.triplet.object
                )
            except Exception:
                continue

            for existing in related_obj:
                if not existing.triplet or existing.id == belief.id:
                    continue
                # New evidence about an entity the existing belief mentions
                # → record observation to slightly boost
                if existing.triplet.subject.lower() == belief.triplet.object.lower():
                    try:
                        await backend.update_belief_confidence(
                            existing.id, delta=0.02
                        )
                        result.backward_updates += 1
                    except Exception:
                        pass

        logger.info(
            "Backward dreaming: %d beliefs updated", result.backward_updates
        )
        return result

    async def _find_chains_from(
        self,
        start_belief: Belief,
        max_hops: int,
        min_confidence: float,
    ) -> list[InferenceChain]:
        """Find transitive chains starting from a belief.

        Uses BFS over entity-belief connections to find multi-hop paths.
        """
        if not start_belief.triplet:
            return []

        chains: list[InferenceChain] = []
        backend = self._memory._backend

        # 2-hop: start.object == mid.subject, get mid.object
        try:
            mid_beliefs = await backend.get_beliefs_by_entity(
                start_belief.triplet.object
            )
        except Exception:
            return []

        for mid in mid_beliefs:
            if not mid.triplet or mid.id == start_belief.id:
                continue
            if mid.confidence < min_confidence:
                continue
            # Must chain: start.object == mid.subject
            if mid.triplet.subject.lower() != start_belief.triplet.object.lower():
                continue

            chain_confidence = min(start_belief.confidence, mid.confidence) * 0.7
            if chain_confidence < min_confidence:
                continue

            path_desc = (
                f"{start_belief.triplet.subject} "
                f"-[{start_belief.triplet.predicate}]-> "
                f"{start_belief.triplet.object} "
                f"-[{mid.triplet.predicate}]-> "
                f"{mid.triplet.object}"
            )

            chains.append(InferenceChain(
                beliefs=[start_belief, mid],
                hops=2,
                confidence=chain_confidence,
                subject=start_belief.triplet.subject,
                object=mid.triplet.object,
                path_description=path_desc,
            ))

            # 3-hop: mid.object == end.subject
            if max_hops >= 3:
                try:
                    end_beliefs = await backend.get_beliefs_by_entity(
                        mid.triplet.object
                    )
                except Exception:
                    continue

                for end in end_beliefs:
                    if not end.triplet or end.id in (start_belief.id, mid.id):
                        continue
                    if end.confidence < min_confidence:
                        continue
                    if end.triplet.subject.lower() != mid.triplet.object.lower():
                        continue

                    chain_3_conf = min(
                        start_belief.confidence, mid.confidence, end.confidence
                    ) * 0.5
                    if chain_3_conf < min_confidence:
                        continue

                    path_desc_3 = (
                        f"{start_belief.triplet.subject} "
                        f"-[{start_belief.triplet.predicate}]-> "
                        f"{start_belief.triplet.object} "
                        f"-[{mid.triplet.predicate}]-> "
                        f"{mid.triplet.object} "
                        f"-[{end.triplet.predicate}]-> "
                        f"{end.triplet.object}"
                    )

                    chains.append(InferenceChain(
                        beliefs=[start_belief, mid, end],
                        hops=3,
                        confidence=chain_3_conf,
                        subject=start_belief.triplet.subject,
                        object=end.triplet.object,
                        path_description=path_desc_3,
                    ))

        return chains

    def _chain_to_belief(self, chain: InferenceChain) -> Belief | None:
        """Convert an inference chain into a new belief."""
        if chain.subject.lower() == chain.object.lower():
            return None  # Self-referential chain

        # Build predicate from chain
        predicates = [
            b.triplet.predicate for b in chain.beliefs if b.triplet
        ]
        if len(predicates) == 2:
            predicate = f"connected to via {predicates[0]} + {predicates[1]}"
        elif len(predicates) == 3:
            predicate = f"connected to via {predicates[0]} + {predicates[1]} + {predicates[2]}"
        else:
            predicate = "transitively connected to"

        evidence_ids = [b.id for b in chain.beliefs]
        chain_descriptions = [
            f"{b.triplet.subject} {b.triplet.predicate} {b.triplet.object}"
            for b in chain.beliefs if b.triplet
        ]

        return Belief(
            id=uuid4(),
            content=f"{chain.subject} {predicate} {chain.object}",
            triplet=Triplet(
                subject=chain.subject,
                predicate=predicate,
                object=chain.object,
            ),
            confidence=chain.confidence,
            source=Source(
                id="transitive_inference",
                type=SourceType.REFLECTION,
                reliability=0.5,
                metadata={
                    "inference_type": "transitive",
                    "hops": chain.hops,
                    "chain": chain_descriptions,
                    "path": chain.path_description,
                    "evidence_beliefs": [str(eid) for eid in evidence_ids],
                },
            ),
            status=BeliefStatus.PROVISIONAL,
            evidence_for=evidence_ids,
            tags={"inferred", "transitive"},
        )

    def _deduplicate(self, beliefs: list[Belief]) -> list[Belief]:
        """Deduplicate inferred beliefs by subject+object."""
        seen: dict[str, Belief] = {}
        for b in beliefs:
            if not b.triplet:
                continue
            key = f"{b.triplet.subject.lower()}|{b.triplet.object.lower()}"
            if key in seen:
                if b.confidence > seen[key].confidence:
                    seen[key] = b
            else:
                seen[key] = b
        return list(seen.values())
