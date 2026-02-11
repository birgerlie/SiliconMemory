"""Entity explorer for comprehensive entity profiles."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from silicon_memory.core.types import Belief
from silicon_memory.graph.types import (
    EntityProfile,
    GraphEdge,
    GraphNode,
    GraphPath,
    NodeType,
)
from silicon_memory.graph.queries import GraphQuery

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class EntityExplorer:
    """Explore entities and their relationships in the knowledge graph.

    Provides comprehensive views of entities including:
    - All beliefs about the entity
    - Related entities (neighbors in graph)
    - Causal chains involving the entity
    - Contradicting beliefs

    Example:
        >>> explorer = EntityExplorer(memory)
        >>> profile = await explorer.explore("Python")
        >>> print(profile.summary())
    """

    def __init__(self, memory: "SiliconMemory") -> None:
        self._memory = memory

    async def explore(
        self,
        entity: str,
        depth: int = 2,
        min_confidence: float = 0.3,
    ) -> EntityProfile:
        """Build comprehensive profile of an entity.

        Args:
            entity: Entity name to explore
            depth: How many hops to explore
            min_confidence: Minimum confidence for beliefs

        Returns:
            EntityProfile with all information about the entity
        """
        profile = EntityProfile(entity=entity)

        # Get all beliefs about the entity
        beliefs = await self._memory._backend.get_beliefs_by_entity(entity)
        beliefs = [b for b in beliefs if b.confidence >= min_confidence]

        # Process beliefs
        all_confidences = []
        timestamps = []

        for belief in beliefs:
            # Add to profile
            belief_dict = {
                "id": str(belief.id),
                "content": belief.content or (
                    belief.triplet.as_text() if belief.triplet else ""
                ),
                "confidence": belief.confidence,
                "status": belief.status.value,
            }

            if belief.source:
                belief_dict["source"] = belief.source.id

            profile.direct_beliefs.append(belief_dict)
            all_confidences.append(belief.confidence)

            if belief.temporal and belief.temporal.observed_at:
                timestamps.append(belief.temporal.observed_at)

            # Extract related entities from triplets
            if belief.triplet:
                # Add related entity (the "other" entity in the triplet)
                if belief.triplet.subject.lower() != entity.lower():
                    related = belief.triplet.subject
                    relation = f"is {belief.triplet.predicate} by"
                else:
                    related = belief.triplet.object
                    relation = belief.triplet.predicate

                profile.related_entities.append((related, relation, belief.confidence))

                # Build edges
                if belief.triplet.subject.lower() == entity.lower():
                    profile.outgoing_edges.append(GraphEdge(
                        source=entity,
                        target=belief.triplet.object,
                        type=belief.triplet.predicate,
                        confidence=belief.confidence,
                    ))
                else:
                    profile.incoming_edges.append(GraphEdge(
                        source=belief.triplet.subject,
                        target=entity,
                        type=belief.triplet.predicate,
                        confidence=belief.confidence,
                    ))

        # Calculate statistics
        profile.total_mentions = len(beliefs)
        if all_confidences:
            profile.average_confidence = sum(all_confidences) / len(all_confidences)
        if timestamps:
            profile.first_mentioned = min(timestamps)
            profile.last_mentioned = max(timestamps)

        # Find contradictions
        profile.contradictions = await self._find_contradictions(entity, beliefs)

        # Explore paths using graph query
        if depth > 1:
            profile.paths = await self._explore_paths(entity, depth, min_confidence)

        # Create the main node
        profile.node = GraphNode(
            id=entity,
            label=entity,
            type=NodeType.ENTITY,
            confidence=profile.average_confidence,
            metadata={
                "mentions": profile.total_mentions,
                "beliefs": len(profile.direct_beliefs),
            },
        )

        # Deduplicate related entities (keep highest confidence)
        profile.related_entities = self._dedupe_related(profile.related_entities)

        return profile

    async def _find_contradictions(
        self,
        entity: str,
        beliefs: list[Belief],
    ) -> list[tuple[str, str]]:
        """Find contradicting beliefs about the entity."""
        contradictions = []

        # Group beliefs by predicate
        by_predicate: dict[str, list[Belief]] = defaultdict(list)
        for belief in beliefs:
            if belief.triplet:
                pred = belief.triplet.predicate.lower()
                by_predicate[pred].append(belief)

        # Check for conflicting objects with same predicate
        for pred, pred_beliefs in by_predicate.items():
            objects: dict[str, Belief] = {}
            for belief in pred_beliefs:
                if belief.triplet:
                    obj = belief.triplet.object.lower()
                    if obj in objects:
                        # Potential contradiction
                        existing = objects[obj]
                        if self._are_conflicting(belief, existing):
                            b1_content = belief.content or belief.triplet.as_text()
                            b2_content = existing.content or existing.triplet.as_text()
                            contradictions.append((b1_content, b2_content))
                    else:
                        objects[obj] = belief

        return contradictions

    def _are_conflicting(self, b1: Belief, b2: Belief) -> bool:
        """Check if two beliefs conflict."""
        if not b1.triplet or not b2.triplet:
            return False

        # Same subject and predicate
        if (b1.triplet.subject.lower() == b2.triplet.subject.lower() and
            b1.triplet.predicate.lower() == b2.triplet.predicate.lower()):
            # Different objects
            return b1.triplet.object.lower() != b2.triplet.object.lower()

        return False

    async def _explore_paths(
        self,
        entity: str,
        depth: int,
        min_confidence: float,
    ) -> list[GraphPath]:
        """Explore paths from the entity."""
        query = (GraphQuery(self._memory)
            .start(entity)
            .traverse(depth=depth)
            .filter(min_confidence=min_confidence)
            .with_paths()
            .limit(50))

        result = await query.execute()
        return result.paths

    def _dedupe_related(
        self,
        related: list[tuple[str, str, float]],
    ) -> list[tuple[str, str, float]]:
        """Deduplicate related entities, keeping highest confidence."""
        best: dict[str, tuple[str, str, float]] = {}

        for entity, relation, conf in related:
            key = entity.lower()
            if key not in best or conf > best[key][2]:
                best[key] = (entity, relation, conf)

        # Sort by confidence
        return sorted(best.values(), key=lambda x: x[2], reverse=True)

    async def compare_entities(
        self,
        entity1: str,
        entity2: str,
        min_confidence: float = 0.3,
    ) -> dict:
        """Compare two entities.

        Args:
            entity1: First entity
            entity2: Second entity
            min_confidence: Minimum confidence threshold

        Returns:
            Comparison results including shared properties, differences, relationships
        """
        profile1 = await self.explore(entity1, depth=1, min_confidence=min_confidence)
        profile2 = await self.explore(entity2, depth=1, min_confidence=min_confidence)

        # Find shared relations
        shared_relations = []
        for e1, r1, c1 in profile1.related_entities:
            for e2, r2, c2 in profile2.related_entities:
                if e1.lower() == e2.lower():
                    shared_relations.append({
                        "entity": e1,
                        "relation1": r1,
                        "relation2": r2,
                        "confidence": (c1 + c2) / 2,
                    })

        # Find direct relationship between entities
        direct_relation = None
        for e, r, c in profile1.related_entities:
            if e.lower() == entity2.lower():
                direct_relation = {"relation": r, "confidence": c}
                break
        if not direct_relation:
            for e, r, c in profile2.related_entities:
                if e.lower() == entity1.lower():
                    direct_relation = {"relation": r, "confidence": c}
                    break

        return {
            "entity1": {
                "name": entity1,
                "mentions": profile1.total_mentions,
                "average_confidence": profile1.average_confidence,
                "related_count": len(profile1.related_entities),
            },
            "entity2": {
                "name": entity2,
                "mentions": profile2.total_mentions,
                "average_confidence": profile2.average_confidence,
                "related_count": len(profile2.related_entities),
            },
            "shared_relations": shared_relations,
            "direct_relation": direct_relation,
            "unique_to_1": [
                (e, r) for e, r, c in profile1.related_entities
                if not any(e.lower() == e2.lower() for e2, _, _ in profile2.related_entities)
            ][:5],
            "unique_to_2": [
                (e, r) for e, r, c in profile2.related_entities
                if not any(e.lower() == e1.lower() for e1, _, _ in profile1.related_entities)
            ][:5],
        }

    async def find_common_ancestors(
        self,
        entities: list[str],
        max_depth: int = 3,
        min_confidence: float = 0.3,
    ) -> list[str]:
        """Find common ancestors/parents of multiple entities.

        Args:
            entities: List of entities to find common ancestors for
            max_depth: Maximum depth to search
            min_confidence: Minimum confidence threshold

        Returns:
            List of common ancestor entity names
        """
        if len(entities) < 2:
            return []

        # Get ancestors for each entity
        ancestor_sets = []
        for entity in entities:
            query = (GraphQuery(self._memory)
                .start(entity)
                .traverse(direction="incoming", depth=max_depth)
                .filter(min_confidence=min_confidence)
                .limit(100))

            result = await query.execute()
            ancestors = {node.id.lower() for node in result.nodes}
            ancestor_sets.append(ancestors)

        # Find intersection
        common = ancestor_sets[0]
        for ancestors in ancestor_sets[1:]:
            common = common & ancestors

        return list(common)
