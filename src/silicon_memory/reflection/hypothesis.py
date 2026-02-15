"""Hypothesis generation engine using graph clustering and LLM reasoning.

Discovers implicit patterns by:
1. Running Louvain community detection on the belief graph
2. Identifying entity clusters that share many connections
3. Using LLM to generate hypotheses about what connects them
4. Storing hypotheses as provisional beliefs for later validation

Also uses PageRank to identify the most important entities in the
knowledge graph, enabling importance-weighted analysis.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.reflection.types import (
    BeliefCandidate,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM output
# ---------------------------------------------------------------------------

class GeneratedHypothesis(BaseModel):
    hypothesis: str
    subject: str
    predicate: str = "hypothetically"
    object: str
    evidence_summary: str
    confidence: float = 0.4


class HypothesisSet(BaseModel):
    hypotheses: list[GeneratedHypothesis] = []


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_HYPOTHESIS_PROMPT = """\
You are an investigative analyst generating hypotheses from a cluster of \
related facts discovered in court documents.

These entities frequently appear together in the same documents and share \
many connections. Your job is to generate hypotheses about what connects \
them — things that are plausible but not yet explicitly stated.

Entity cluster: {entities}

Known facts about this cluster:
{facts}

Entity importance scores (higher = more central):
{importance}

Generate 2-5 hypotheses about:
- Hidden connections between these entities
- Patterns that explain why they cluster together
- Potential actions, motivations, or roles not yet documented
- Temporal patterns (if dates suggest a sequence of events)

Rules:
- Each hypothesis must be grounded in the facts above
- Assign confidence 0.2-0.6 (these are hypotheses, not proven facts)
- The subject and object must be entities from the cluster
- Be specific — "they are connected" is too vague

Respond with a JSON object:
{{"hypotheses": [
  {{"hypothesis": "...", "subject": "...", "predicate": "...", "object": "...",
    "evidence_summary": "...", "confidence": 0.4}}
]}}
No markdown fences, just valid JSON."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HypothesisResult:
    """Result of hypothesis generation."""

    hypotheses: list[BeliefCandidate] = field(default_factory=list)
    communities_found: int = 0
    entities_analyzed: int = 0
    pagerank_computed: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HypothesisGenerator:
    """Generates hypotheses from knowledge graph structure.

    Uses Louvain community detection to find clusters of related entities,
    PageRank to identify the most important entities, then LLM to reason
    about what connects them.

    Example:
        >>> generator = HypothesisGenerator(memory, llm)
        >>> result = await generator.generate()
        >>> for h in result.hypotheses:
        ...     print(f"[{h.confidence:.2f}] {h.content}")
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        llm: Any,
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._config = config or ReflectionConfig()

    async def generate(
        self,
        max_communities: int = 10,
        min_community_size: int = 3,
    ) -> HypothesisResult:
        """Generate hypotheses from belief graph clusters.

        1. Run Louvain communities on the belief graph
        2. Run PageRank to get entity importance
        3. For each community, collect entities and known facts
        4. Use LLM to generate hypotheses about each cluster

        Args:
            max_communities: Max communities to analyze
            min_community_size: Min entities in a community to analyze

        Returns:
            HypothesisResult with generated hypothesis candidates
        """
        result = HypothesisResult()
        db = self._memory._backend._db

        # Step 1: Run Louvain community detection
        try:
            communities = db.louvain_communities(resolution=1.0)
        except Exception as e:
            logger.warning("Louvain community detection failed: %s", e)
            # Fallback: generate from belief clusters without graph
            return await self._generate_from_beliefs(result)

        if not communities:
            logger.info("No communities found in graph")
            return await self._generate_from_beliefs(result)

        # Group external_ids by community
        community_map: dict[int, list[str]] = defaultdict(list)
        for ext_id, comm_id in communities.items():
            community_map[comm_id].append(ext_id)

        # Sort by size, take largest
        sorted_communities = sorted(
            community_map.items(), key=lambda x: len(x[1]), reverse=True
        )
        result.communities_found = len(sorted_communities)
        logger.info("Louvain found %d communities", result.communities_found)

        # Step 2: Run PageRank
        pagerank_scores: dict[str, float] = {}
        try:
            pagerank_scores = db.pagerank(damping_factor=0.85)
            result.pagerank_computed = True
            logger.info("PageRank computed for %d nodes", len(pagerank_scores))
        except Exception as e:
            logger.warning("PageRank failed: %s", e)

        # Step 3: Analyze each community
        analyzed = 0
        for comm_id, ext_ids in sorted_communities:
            if analyzed >= max_communities:
                break
            if len(ext_ids) < min_community_size:
                continue

            hypotheses = await self._analyze_community(
                ext_ids, pagerank_scores
            )
            result.hypotheses.extend(hypotheses)
            result.entities_analyzed += len(ext_ids)
            analyzed += 1

        logger.info(
            "Hypothesis generation: %d communities analyzed, %d hypotheses generated",
            analyzed, len(result.hypotheses),
        )
        return result

    async def _analyze_community(
        self,
        external_ids: list[str],
        pagerank_scores: dict[str, float],
    ) -> list[BeliefCandidate]:
        """Analyze a single community to generate hypotheses."""
        backend = self._memory._backend

        # Extract entities from external IDs
        entities: set[str] = set()
        belief_descriptions: list[str] = []

        for ext_id in external_ids[:50]:  # Cap at 50 nodes
            # Get the node's data
            try:
                doc = backend._db.get(ext_id)
                if not doc:
                    continue
            except Exception:
                continue

            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            text = doc.text if hasattr(doc, 'text') else ""

            # If it's a belief triple
            subject = metadata.get("subject", "")
            object_val = metadata.get("object", "")
            predicate = metadata.get("predicate", "")

            if subject:
                entities.add(subject)
            if object_val:
                entities.add(object_val)
            if subject and predicate and object_val:
                belief_descriptions.append(
                    f"- {subject} {predicate} {object_val}"
                )
            elif text:
                belief_descriptions.append(f"- {text[:150]}")

        if len(entities) < 2 or not belief_descriptions:
            return []

        # Build importance scores for entities in this community
        importance_lines = []
        for entity in sorted(entities):
            # Find best matching pagerank score
            best_score = 0.0
            for ext_id, score in pagerank_scores.items():
                if entity.lower() in ext_id.lower():
                    best_score = max(best_score, score)
            if best_score > 0:
                importance_lines.append(f"- {entity}: {best_score:.4f}")

        # Build LLM prompt
        entities_str = ", ".join(sorted(entities)[:20])
        facts_str = "\n".join(belief_descriptions[:30])
        importance_str = "\n".join(importance_lines[:20]) or "- (PageRank not available)"

        prompt = _HYPOTHESIS_PROMPT.format(
            entities=entities_str,
            facts=facts_str,
            importance=importance_str,
        )

        # Call LLM
        try:
            llm_result = await self._llm.generate_structured(
                prompt,
                HypothesisSet,
                max_tokens=2000,
            )
        except Exception:
            try:
                raw = await self._llm.generate(
                    prompt, max_tokens=2000, temperature=0.5
                )
                llm_result = self._parse_raw(raw)
            except Exception as e:
                logger.warning("Hypothesis LLM call failed: %s", e)
                return []

        # Convert to BeliefCandidates
        candidates = []
        for h in llm_result.hypotheses:
            if not h.hypothesis or not h.subject or not h.object:
                continue
            candidates.append(BeliefCandidate(
                id=uuid4(),
                content=f"[HYPOTHESIS] {h.hypothesis}",
                subject=h.subject,
                predicate=h.predicate or "hypothetically",
                object=h.object,
                confidence=min(0.6, max(0.1, h.confidence)),
                source_context={
                    "type": "hypothesis",
                    "evidence_summary": h.evidence_summary,
                    "community_size": len(entities),
                    "entities_in_cluster": sorted(entities)[:10],
                },
                reasoning=f"Generated from Louvain community with {len(entities)} entities",
            ))

        return candidates

    async def _generate_from_beliefs(
        self,
        result: HypothesisResult,
    ) -> HypothesisResult:
        """Fallback: generate hypotheses from belief entity co-occurrence.

        When graph-based community detection isn't available,
        cluster beliefs by shared entities instead.
        """
        backend = self._memory._backend

        # Search for high-confidence beliefs
        try:
            beliefs = await backend.query_beliefs(
                query="*", limit=200, min_confidence=0.5
            )
        except Exception as e:
            logger.warning("Fallback belief query failed: %s", e)
            return result

        if not beliefs:
            return result

        # Build entity → beliefs map
        entity_beliefs: dict[str, list[Belief]] = defaultdict(list)
        for b in beliefs:
            if b.triplet:
                entity_beliefs[b.triplet.subject.lower()].append(b)
                entity_beliefs[b.triplet.object.lower()].append(b)

        # Find entities that appear in many beliefs (natural clusters)
        hub_entities = sorted(
            entity_beliefs.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:10]

        for entity, entity_beliefs_list in hub_entities:
            if len(entity_beliefs_list) < 3:
                continue
            # Collect all entities connected to this hub
            connected = set()
            facts = []
            for b in entity_beliefs_list:
                if b.triplet:
                    connected.add(b.triplet.subject)
                    connected.add(b.triplet.object)
                    facts.append(
                        f"- {b.triplet.subject} {b.triplet.predicate} {b.triplet.object}"
                    )

            if len(connected) < 3:
                continue

            prompt = _HYPOTHESIS_PROMPT.format(
                entities=", ".join(sorted(connected)[:15]),
                facts="\n".join(facts[:20]),
                importance="- (not available)",
            )

            try:
                llm_result = await self._llm.generate_structured(
                    prompt, HypothesisSet, max_tokens=2000
                )
            except Exception:
                try:
                    raw = await self._llm.generate(
                        prompt, max_tokens=2000, temperature=0.5
                    )
                    llm_result = self._parse_raw(raw)
                except Exception:
                    continue

            for h in llm_result.hypotheses:
                if not h.hypothesis or not h.subject or not h.object:
                    continue
                result.hypotheses.append(BeliefCandidate(
                    id=uuid4(),
                    content=f"[HYPOTHESIS] {h.hypothesis}",
                    subject=h.subject,
                    predicate=h.predicate or "hypothetically",
                    object=h.object,
                    confidence=min(0.6, max(0.1, h.confidence)),
                    source_context={
                        "type": "hypothesis",
                        "hub_entity": entity,
                        "evidence_summary": h.evidence_summary,
                    },
                    reasoning=f"Generated from entity hub '{entity}' with {len(entity_beliefs_list)} beliefs",
                ))

        return result

    def _parse_raw(self, raw: str) -> HypothesisSet:
        """Parse raw LLM text into HypothesisSet."""
        import json
        import re

        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group())
                return HypothesisSet.model_validate(parsed)
            except Exception:
                pass

        return HypothesisSet()
