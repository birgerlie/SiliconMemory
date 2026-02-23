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


_META_PREDICATES = {
    "possibly_connected_via",
    "has_conflicting_claim_on",
    "timeline_progresses_from_to",
    "hypothetically",
}


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
        max_nodes: int = 500,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()
        self._max_nodes = max_nodes or self._config.max_consolidation_nodes

    @staticmethod
    def _is_meta_predicate(predicate: str) -> bool:
        norm = " ".join(predicate.strip().lower().split())
        if not norm:
            return True
        if norm.startswith("systematically "):
            return True
        if norm in _META_PREDICATES:
            return True
        return False

    async def consolidate(self) -> ConsolidationStats:
        """Run full memory consolidation.

        1. Compute importance via PageRank
        2. Apply importance-weighted decay
        3. Find and create generalizations
        4. Build cross-modal links
        """
        stats = ConsolidationStats()

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
        """Compute entity importance via PageRank, truncated to top-N."""
        storage = self._memory._storage
        try:
            pr_results = await storage.pagerank(k=self._max_nodes if self._max_nodes else 10000)
            if isinstance(pr_results, list):
                scores = {
                    n.get("external_id", n.get("id", "")): n.get("score", 0.0)
                    for n in pr_results
                }
            elif isinstance(pr_results, dict) and "nodes" in pr_results:
                scores = {
                    n.get("external_id", n.get("id", "")): n.get("score", 0.0)
                    for n in pr_results["nodes"]
                }
            else:
                scores = pr_results if isinstance(pr_results, dict) else {}
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

        Uses batch update_and_apply_probabilities() when available,
        falling back to sequential record_observation() calls.
        """
        if not importance:
            return 0

        # Find beliefs with low importance
        median_score = sorted(importance.values())[len(importance) // 2] if importance else 0

        decay_ids: list[str] = []
        for ext_id, score in importance.items():
            if score >= median_score:
                continue
            if "/belief-" not in ext_id:
                continue
            decay_ids.append(ext_id)
            if len(decay_ids) >= 100:  # Cap per cycle
                break

        if not decay_ids:
            return 0

        # Try batch Monte Carlo update first
        storage = self._memory._storage
        try:
            evidence = [
                {"external_id": ext_id, "confidence": 0.3}
                for ext_id in decay_ids
            ]
            await storage.update_and_apply_probabilities(evidence)
            logger.info("Batch decay: %d beliefs via update_and_apply_probabilities", len(decay_ids))
            return len(decay_ids)
        except Exception:
            pass

        # Fallback: sequential record_observation
        decay_count = 0
        for ext_id in decay_ids:
            try:
                await storage.record_observation(
                    external_id=ext_id,
                    confirmed=False,
                    source="importance_decay",
                )
                decay_count += 1
            except Exception:
                pass

        return decay_count

    async def _strengthen_important(
        self,
        importance: dict[str, float],
    ) -> int:
        """Strengthen high-importance beliefs.

        Beliefs about central entities get a small positive confidence
        boost, modeling the cognitive tendency to remember important
        things better.

        Uses batch update_and_apply_probabilities() when available,
        falling back to sequential record_observation() calls.
        """
        if not importance:
            return 0

        # Top 10% by importance
        sorted_scores = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_threshold = len(sorted_scores) // 10 or 1

        strengthen_ids: list[str] = []
        for ext_id, score in sorted_scores[:top_threshold]:
            if "/belief-" not in ext_id:
                continue
            strengthen_ids.append(ext_id)
            if len(strengthen_ids) >= 50:  # Cap per cycle
                break

        if not strengthen_ids:
            return 0

        # Try batch Monte Carlo update first
        storage = self._memory._storage
        try:
            evidence = [
                {"external_id": ext_id, "confidence": 0.9}
                for ext_id in strengthen_ids
            ]
            await storage.update_and_apply_probabilities(evidence)
            logger.info(
                "Batch strengthen: %d beliefs via update_and_apply_probabilities",
                len(strengthen_ids),
            )
            return len(strengthen_ids)
        except Exception:
            pass

        # Fallback: sequential record_observation
        strengthen_count = 0
        for ext_id in strengthen_ids:
            try:
                await storage.record_observation(
                    external_id=ext_id,
                    confirmed=True,
                    source="importance_strengthen",
                )
                strengthen_count += 1
            except Exception:
                pass

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
        storage = self._memory._storage
        generalizations_created = 0

        # Query triples directly to get beliefs WITH triplet data.
        # (query_beliefs returns search docs first which lack triplets)
        try:
            triples = await storage.query_triples(min_probability=0.4, k=self._max_nodes if self._max_nodes else 100_000)
            beliefs = []
            for t in triples:
                b = backend._triple_to_belief(t)
                if b and b.triplet:
                    beliefs.append(b)
        except Exception:
            return 0

        # Generalize only from base extracted beliefs to avoid recursive
        # compounding over synthetic hypotheses/generalizations.
        base_beliefs: list[Belief] = []
        for b in beliefs:
            if not b.triplet:
                continue
            tags = {str(t).lower() for t in (b.tags or set())}
            if "hypothesis" in tags or "generalization" in tags:
                continue
            if self._is_meta_predicate(b.triplet.predicate):
                continue
            base_beliefs.append(b)

        # Group by (subject, predicate) to find repeated patterns
        pattern_groups: dict[str, list[Belief]] = defaultdict(list)
        for b in base_beliefs:
            if not b.triplet:
                continue
            key = f"{b.triplet.subject.lower()}|{b.triplet.predicate.lower()}"
            pattern_groups[key].append(b)

        for key, group in pattern_groups.items():
            if len(group) < 3:  # Need 3+ instances to generalize
                continue

            subject, predicate = key.split("|", 1)
            if self._is_meta_predicate(predicate):
                continue
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
        for b in base_beliefs:
            if not b.triplet:
                continue
            key = f"{b.triplet.predicate.lower()}|{b.triplet.object.lower()}"
            object_groups[key].append(b)

        for key, group in object_groups.items():
            if len(group) < 3:
                continue

            predicate, obj = key.split("|", 1)
            if self._is_meta_predicate(predicate):
                continue
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
        storage = self._memory._storage

        # Use Louvain to find clusters
        try:
            communities = await storage.louvain_communities(resolution=1.0)
        except Exception:
            return result

        if not communities:
            return result

        # Group by community
        comm_map: dict[int, list[str]] = defaultdict(list)
        for ext_id, comm_id in communities.items():
            comm_map[comm_id].append(ext_id)

        # Cap to 50 communities to bound processing
        if len(comm_map) > 50:
            # Keep the 50 largest communities
            sorted_comms = sorted(comm_map.items(), key=lambda x: len(x[1]), reverse=True)
            comm_map = dict(sorted_comms[:50])

        result["clusters"] = len(comm_map)

        # For each community, create cross-modal co-occurrence links
        for comm_id, members in comm_map.items():
            if len(members) < 2:
                continue

            belief_ids = [m for m in members if "/belief-" in m]
            if len(belief_ids) >= 2:
                try:
                    await storage.add_cooccurrences(
                        belief_ids[:20],  # Cap at 20 per community
                        session_id=f"cluster-{comm_id}",
                    )
                    result["links"] += len(belief_ids) * (len(belief_ids) - 1) // 2
                except Exception:
                    pass

        return result
