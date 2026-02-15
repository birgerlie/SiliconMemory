"""Memory consolidation engine — bidirectional dreaming and generalization.

Implements cognitive-inspired memory consolidation:

1. **Dream Forward**: New experiences generate new inferences, hypotheses,
   and connect to existing knowledge via transitive chains.

2. **Dream Backward**: New information triggers re-evaluation of existing
   beliefs, updating confidence, detecting contradictions, and
   generalizing recurring patterns.

3. **Generalization**: Over time, frequently co-occurring patterns are
   abstracted into higher-level beliefs. Specific details decay while
   general patterns strengthen (hippocampal → neocortical transfer).

4. **Layered Clustering**: Different memory types (facts, relationships,
   arguments, events) form separate cluster hierarchies that are then
   cross-linked for multi-modal reasoning.

5. **Importance Decay**: Uses PageRank as proxy for importance.
   Low-importance beliefs decay faster, high-importance beliefs are
   preserved and generalized.
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
    BeliefCandidate,
    PatternType,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationStats:
    """Statistics from a consolidation pass."""

    generalizations_created: int = 0
    beliefs_decayed: int = 0
    beliefs_strengthened: int = 0
    clusters_found: int = 0
    cross_links_created: int = 0
    importance_scores_computed: int = 0


class MemoryConsolidator:
    """Consolidates memory through generalization, decay, and clustering.

    Models the cognitive process where:
    - Frequently accessed memories strengthen
    - Rarely accessed memories decay
    - Similar memories merge into generalizations
    - Cross-modal links form between different memory types

    Example:
        >>> consolidator = MemoryConsolidator(memory)
        >>> stats = await consolidator.consolidate()
        >>> print(f"Created {stats.generalizations_created} generalizations")
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()

    async def consolidate(self) -> ConsolidationStats:
        """Run full memory consolidation.

        1. Compute importance via PageRank
        2. Apply importance-weighted decay
        3. Find and create generalizations
        4. Build cross-modal links
        """
        stats = ConsolidationStats()
        db = self._memory._backend._db

        # Step 1: Compute importance scores
        importance = await self._compute_importance()
        stats.importance_scores_computed = len(importance)

        # Step 2: Apply decay to low-importance beliefs
        stats.beliefs_decayed = await self._apply_decay(importance)

        # Step 3: Strengthen high-importance beliefs
        stats.beliefs_strengthened = await self._strengthen_important(importance)

        # Step 4: Generalize recurring patterns
        stats.generalizations_created = await self._generalize()

        # Step 5: Build cross-modal clusters
        cluster_stats = await self._build_clusters()
        stats.clusters_found = cluster_stats["clusters"]
        stats.cross_links_created = cluster_stats["links"]

        logger.info(
            "Consolidation: %d generalizations, %d decayed, %d strengthened, "
            "%d clusters, %d cross-links",
            stats.generalizations_created, stats.beliefs_decayed,
            stats.beliefs_strengthened, stats.clusters_found,
            stats.cross_links_created,
        )
        return stats

    async def _compute_importance(self) -> dict[str, float]:
        """Compute entity importance via PageRank."""
        db = self._memory._backend._db
        try:
            scores = db.pagerank(damping_factor=0.85)
            return scores
        except Exception as e:
            logger.warning("PageRank failed: %s", e)
            return {}

    async def _apply_decay(
        self,
        importance: dict[str, float],
    ) -> int:
        """Decay low-importance beliefs.

        Beliefs about entities with low PageRank score get a small
        negative confidence adjustment. This models forgetting of
        unimportant details over time.
        """
        if not importance:
            return 0

        backend = self._memory._backend
        decay_count = 0

        # Find beliefs with low importance
        median_score = sorted(importance.values())[len(importance) // 2] if importance else 0

        for ext_id, score in importance.items():
            if score >= median_score:
                continue
            # Only decay beliefs (not experiences or procedures)
            if "/belief-" not in ext_id:
                continue

            try:
                # Small negative observation for low-importance beliefs
                backend._db.record_observation(
                    external_id=ext_id,
                    confirmed=False,
                    source="importance_decay",
                )
                decay_count += 1
            except Exception:
                pass

            if decay_count >= 100:  # Cap per cycle
                break

        return decay_count

    async def _strengthen_important(
        self,
        importance: dict[str, float],
    ) -> int:
        """Strengthen high-importance beliefs.

        Beliefs about central entities get a small positive confidence
        boost, modeling the cognitive tendency to remember important
        things better.
        """
        if not importance:
            return 0

        backend = self._memory._backend
        strengthen_count = 0

        # Top 10% by importance
        sorted_scores = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_threshold = len(sorted_scores) // 10 or 1

        for ext_id, score in sorted_scores[:top_threshold]:
            if "/belief-" not in ext_id:
                continue

            try:
                backend._db.record_observation(
                    external_id=ext_id,
                    confirmed=True,
                    source="importance_strengthen",
                )
                strengthen_count += 1
            except Exception:
                pass

            if strengthen_count >= 50:  # Cap per cycle
                break

        return strengthen_count

    async def _generalize(self) -> int:
        """Find recurring patterns and create generalizations.

        Looks for beliefs that share the same predicate across many
        subjects/objects. When the same pattern repeats enough times,
        creates a higher-level generalized belief.

        Example:
          - "Maxwell recruited victim A"
          - "Maxwell recruited victim B"
          - "Maxwell recruited victim C"
          → Generalization: "Maxwell systematically recruited victims"
        """
        backend = self._memory._backend
        generalizations_created = 0

        # Query triples directly to get beliefs WITH triplet data.
        # (query_beliefs returns search docs first which lack triplets)
        try:
            triples = backend._db.query_triples(min_probability=0.4, k=1000)
            beliefs = []
            for t in triples:
                b = backend._triple_to_belief(t)
                if b and b.triplet:
                    beliefs.append(b)
        except Exception:
            return 0

        # Group by (subject, predicate) to find repeated patterns
        pattern_groups: dict[str, list[Belief]] = defaultdict(list)
        for b in beliefs:
            if not b.triplet:
                continue
            key = f"{b.triplet.subject.lower()}|{b.triplet.predicate.lower()}"
            pattern_groups[key].append(b)

        for key, group in pattern_groups.items():
            if len(group) < 3:  # Need 3+ instances to generalize
                continue

            subject, predicate = key.split("|", 1)
            objects = [b.triplet.object for b in group if b.triplet]
            avg_confidence = sum(b.confidence for b in group) / len(group)

            # Create generalization
            gen_content = (
                f"{group[0].triplet.subject} systematically "
                f"{predicate} multiple parties "
                f"(including {', '.join(objects[:5])})"
            )
            gen_belief = Belief(
                id=uuid4(),
                content=gen_content,
                triplet=Triplet(
                    subject=group[0].triplet.subject,
                    predicate=f"systematically {predicate}",
                    object=f"multiple ({len(objects)})",
                ),
                confidence=min(0.9, avg_confidence + 0.1),
                source=Source(
                    id="consolidation_generalization",
                    type=SourceType.REFLECTION,
                    reliability=0.6,
                    metadata={
                        "generalization_type": "predicate_pattern",
                        "instance_count": len(group),
                        "instances": [str(b.id) for b in group[:10]],
                        "objects": objects[:10],
                    },
                ),
                status=BeliefStatus.PROVISIONAL,
                evidence_for=[b.id for b in group],
                tags={"generalization", "systematic_pattern"},
            )

            try:
                await self._memory.commit_belief(gen_belief)
                generalizations_created += 1
            except Exception as e:
                logger.warning("Failed to commit generalization: %s", e)

        # Also look for object-side patterns: multiple subjects + same predicate + same object
        object_groups: dict[str, list[Belief]] = defaultdict(list)
        for b in beliefs:
            if not b.triplet:
                continue
            key = f"{b.triplet.predicate.lower()}|{b.triplet.object.lower()}"
            object_groups[key].append(b)

        for key, group in object_groups.items():
            if len(group) < 3:
                continue

            predicate, obj = key.split("|", 1)
            subjects = [b.triplet.subject for b in group if b.triplet]
            avg_confidence = sum(b.confidence for b in group) / len(group)

            gen_content = (
                f"Multiple parties ({', '.join(subjects[:5])}) "
                f"{predicate} {group[0].triplet.object}"
            )
            gen_belief = Belief(
                id=uuid4(),
                content=gen_content,
                triplet=Triplet(
                    subject=f"multiple ({len(subjects)})",
                    predicate=predicate,
                    object=group[0].triplet.object,
                ),
                confidence=min(0.9, avg_confidence + 0.1),
                source=Source(
                    id="consolidation_generalization",
                    type=SourceType.REFLECTION,
                    reliability=0.6,
                    metadata={
                        "generalization_type": "object_pattern",
                        "instance_count": len(group),
                        "instances": [str(b.id) for b in group[:10]],
                        "subjects": subjects[:10],
                    },
                ),
                status=BeliefStatus.PROVISIONAL,
                evidence_for=[b.id for b in group],
                tags={"generalization", "convergent_pattern"},
            )

            try:
                await self._memory.commit_belief(gen_belief)
                generalizations_created += 1
            except Exception as e:
                logger.warning("Failed to commit generalization: %s", e)

        return generalizations_created

    async def _build_clusters(self) -> dict[str, int]:
        """Build layered cross-modal clusters.

        Creates graph edges between beliefs of different types
        (facts, relationships, arguments, events) that share entities.
        This enables cross-modal reasoning like:
        "fact about X" + "argument involving X" + "event dated Y involving X"
        """
        result = {"clusters": 0, "links": 0}
        db = self._memory._backend._db

        # Use Louvain to find clusters
        try:
            communities = db.louvain_communities(resolution=1.0)
        except Exception:
            return result

        if not communities:
            return result

        # Group by community
        comm_map: dict[int, list[str]] = defaultdict(list)
        for ext_id, comm_id in communities.items():
            comm_map[comm_id].append(ext_id)

        result["clusters"] = len(comm_map)

        # For each community, create cross-modal co-occurrence links
        for comm_id, members in comm_map.items():
            if len(members) < 2:
                continue

            belief_ids = [m for m in members if "/belief-" in m]
            if len(belief_ids) >= 2:
                try:
                    db.add_cooccurrences(
                        belief_ids[:20],  # Cap at 20 per community
                        session_id=f"cluster-{comm_id}",
                    )
                    result["links"] += len(belief_ids) * (len(belief_ids) - 1) // 2
                except Exception:
                    pass

        return result
