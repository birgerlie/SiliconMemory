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

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import re
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
from silicon_memory.entities.date_normalizer import normalize_date
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
You are an analytical reasoning engine generating hypotheses from a cluster of \
related facts in a knowledge base.

These entities frequently appear together and share many connections. \
Generate hypotheses about what connects them — plausible inferences not yet \
explicitly stated.

Entity cluster: {entities}

Known facts about this cluster:
{facts}

Entity importance scores (higher = more central):
{importance}

Generate 2-5 hypotheses about:
- Relationships between these specific entities that explain why they cluster
- Temporal patterns (if dates suggest a sequence of events)
- Roles or motivations not yet documented

Rules:
- Each hypothesis must be grounded in the facts above
- Assign confidence 0.2-0.6 (these are hypotheses, not proven facts)
- The subject and object must be entities from the cluster
- Be specific — "they are connected" is too vague
- CRITICAL: Do NOT transfer actions or predicates from one entity to another. \
If entity A did something, do NOT hypothesize that entity B did the same thing \
unless there is specific evidence for B. For example, if person X "committed a crime", \
do NOT infer that person Y (a judge, lawyer, or organization) also "committed a crime" \
just because they appear in the same cluster.
- Only generate hypotheses about the RELATIONSHIP between entities, not about \
individual entities inheriting each other's attributes

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


@dataclass(frozen=True)
class CommunityEdge:
    """Structured belief edge used for deterministic hypothesis operators."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    source_doc_id: str = ""
    temporal_point: str = ""


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

    def _db_timeout_s(self) -> float:
        """Timeout for blocking graph-db operations during dream."""
        timeout_s = float(getattr(self._config, "dream_graph_call_timeout_s", 10.0))
        return max(0.1, timeout_s)

    async def _run_db_call(
        self,
        name: str,
        fn: Any,
        default: Any,
        *,
        warn: bool = True,
    ) -> Any:
        """Run a blocking DB function in a thread with timeout.

        Dream-phase graph APIs are synchronous in the current SiliconDB client.
        Running them in a thread keeps the event loop responsive and lets us
        fail fast instead of hanging an entire cycle.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn),
                timeout=self._db_timeout_s(),
            )
        except Exception as exc:
            if warn:
                logger.warning("Dream graph call failed (%s): %s", name, exc)
            return default

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @classmethod
    def _triplet_key(cls, subject: str, predicate: str, object_value: str) -> str:
        return (
            f"{cls._norm(subject)}|{cls._norm(predicate)}|{cls._norm(object_value)}"
        )

    @classmethod
    def _candidate_key(cls, candidate: BeliefCandidate) -> str:
        return cls._triplet_key(
            candidate.subject or "",
            candidate.predicate or "",
            candidate.object or "",
        )

    @staticmethod
    def _parse_date_key(value: str) -> tuple[int, int, int] | None:
        """Parse canonical-ish date strings for temporal ordering."""
        text = value.strip()
        if not text:
            return None
        if re.fullmatch(r"(19|20)\d{2}-\d{2}-\d{2}", text):
            dt = datetime.strptime(text, "%Y-%m-%d")
            return (dt.year, dt.month, dt.day)
        if re.fullmatch(r"(19|20)\d{2}-\d{2}", text):
            dt = datetime.strptime(text + "-01", "%Y-%m-%d")
            return (dt.year, dt.month, dt.day)
        if re.fullmatch(r"(19|20)\d{2}", text):
            return (int(text), 1, 1)
        return None

    @staticmethod
    def _is_temporal_predicate(predicate: str) -> bool:
        """Heuristic: predicate likely denotes a dated/temporal assertion."""
        norm = " ".join(predicate.strip().lower().split())
        if not norm:
            return False
        temporal_keywords = (
            "occurred_on",
            "date",
            "dated",
            "filed",
            "decided",
            "argued",
            "signed",
            "started",
            "began",
            "ended",
            "continued",
            "until",
            "during",
            "before",
            "after",
        )
        return any(keyword in norm for keyword in temporal_keywords)

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
        communities = await self._run_db_call(
            "louvain_communities",
            lambda: db.louvain_communities(resolution=1.0),
            {},
        )
        if not communities:
            # Fallback: generate from belief clusters without graph
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
        pr_results = await self._run_db_call(
            "pagerank",
            lambda: db.pagerank(k=100),
            {},
            warn=False,
        )
        if isinstance(pr_results, list):
            pagerank_scores = {
                n.get("external_id", n.get("id", "")): n.get("score", 0.0)
                for n in pr_results
            }
        elif isinstance(pr_results, dict) and "nodes" in pr_results:
            pagerank_scores = {
                n.get("external_id", n.get("id", "")): n.get("score", 0.0)
                for n in pr_results["nodes"]
            }
        else:
            pagerank_scores = pr_results if isinstance(pr_results, dict) else {}
        result.pagerank_computed = bool(pagerank_scores)
        if pagerank_scores:
            logger.info("PageRank computed for %d nodes", len(pagerank_scores))

        # Step 3: Analyze each community
        candidates: list[list[str]] = []
        for _, ext_ids in sorted_communities:
            if len(candidates) >= max_communities:
                break
            if len(ext_ids) < min_community_size:
                continue
            candidates.append(ext_ids)

        entity_conf_cache: dict[str, float] = {}
        sem = asyncio.Semaphore(3)

        async def _analyze(ext_ids: list[str]) -> tuple[list[BeliefCandidate], int]:
            async with sem:
                expanded_ids = await self._discover_outward_ids(
                    ext_ids,
                    max_hops=1,
                    max_nodes=80,
                )
                hypotheses = await self._analyze_community(
                    expanded_ids,
                    pagerank_scores,
                    entity_conf_cache,
                )
                return hypotheses, len(expanded_ids)

        analyzed = len(candidates)
        if candidates:
            analyzed_results = await asyncio.gather(*(_analyze(ids) for ids in candidates))
            for hypotheses, entity_count in analyzed_results:
                result.hypotheses.extend(hypotheses)
                result.entities_analyzed += entity_count

        logger.info(
            "Hypothesis generation: %d communities analyzed, %d hypotheses generated",
            analyzed, len(result.hypotheses),
        )
        if analyzed > 0 and not result.hypotheses:
            logger.info(
                "No hypotheses from community analysis; falling back to belief-based generation",
            )
            result = await self._generate_from_beliefs(result)

        # Always add a global temporal pass independent of community quality.
        temporal_candidates = await self._generate_global_temporal_hypotheses(limit=40)
        if temporal_candidates:
            merged = list(result.hypotheses)
            seen = {self._candidate_key(c) for c in merged}
            for candidate in temporal_candidates:
                key = self._candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
            result.hypotheses = self._rank_candidates_by_information_gain(
                merged,
                known_triplet_keys=set(),
            )[:150]

        return result

    async def _analyze_community(
        self,
        external_ids: list[str],
        pagerank_scores: dict[str, float],
        entity_conf_cache: dict[str, float] | None = None,
    ) -> list[BeliefCandidate]:
        """Analyze a single community to generate hypotheses."""
        backend = self._memory._backend
        local_conf_cache = entity_conf_cache if entity_conf_cache is not None else {}

        # Extract entities and structured belief edges from external IDs.
        entities: set[str] = set()
        edges: list[CommunityEdge] = []
        seen_edge_keys: set[str] = set()
        belief_descriptions: list[str] = []
        seen_descriptions: set[str] = set()
        hydrated_entities: set[str] = set()

        def _append_edge(
            subject: str,
            predicate: str,
            object_value: str,
            confidence: float,
            source_doc_id: str = "",
            temporal_point: str = "",
        ) -> None:
            edge = CommunityEdge(
                subject=subject,
                predicate=predicate,
                object=object_value,
                confidence=confidence,
                source_doc_id=source_doc_id,
                temporal_point=temporal_point,
            )
            edge_key = self._triplet_key(
                edge.subject,
                edge.predicate,
                edge.object,
            )
            if edge_key in seen_edge_keys:
                return
            seen_edge_keys.add(edge_key)
            edges.append(edge)
            entities.add(subject)
            entities.add(object_value)
            desc = f"- {subject} {predicate} {object_value}"
            if desc not in seen_descriptions:
                seen_descriptions.add(desc)
                belief_descriptions.append(desc)

        async def _hydrate_entity_context(entity_name: str) -> None:
            norm = self._norm(entity_name)
            if not norm or norm in hydrated_entities:
                return
            hydrated_entities.add(norm)
            try:
                nearby = await backend.get_beliefs_by_entity(entity_name)
            except Exception:
                nearby = []
            for belief in nearby[:16]:
                if not belief.triplet:
                    continue
                source_doc_id = ""
                temporal_point = ""
                if belief.source and belief.source.metadata:
                    source_doc_id = str(belief.source.metadata.get("grounding_doc_id", ""))
                if belief.temporal and belief.temporal.observed_at:
                    temporal_point = belief.temporal.observed_at.date().isoformat()
                _append_edge(
                    subject=belief.triplet.subject,
                    predicate=belief.triplet.predicate,
                    object_value=belief.triplet.object,
                    confidence=float(belief.confidence),
                    source_doc_id=source_doc_id,
                    temporal_point=temporal_point,
                )

        for ext_id in external_ids:
            if ext_id.startswith("entity:"):
                entity_name = ext_id.split("entity:", 1)[-1].strip()
                if entity_name:
                    entities.add(entity_name)
                    await _hydrate_entity_context(entity_name)

            # Primary path for belief nodes: read belief/triplet directly and
            # enrich with extraction-journal text/context where available.
            if "/belief-" in ext_id:
                try:
                    belief_uuid = UUID(ext_id.rsplit("belief-", 1)[-1])
                except Exception:
                    belief_uuid = None
                if belief_uuid is not None:
                    belief = await self._memory.get_belief(belief_uuid)
                    if belief and belief.triplet:
                        _append_edge(
                            subject=belief.triplet.subject,
                            predicate=belief.triplet.predicate,
                            object_value=belief.triplet.object,
                            confidence=float(belief.confidence),
                            source_doc_id=str(
                                ((belief.source.metadata or {}) if belief.source else {}).get(
                                    "grounding_doc_id", ""
                                )
                            ),
                            temporal_point=(
                                belief.temporal.observed_at.date().isoformat()
                                if belief.temporal and belief.temporal.observed_at else ""
                            ),
                        )

                extraction_ext_id = ext_id.replace("/belief-", "/extraction-")
                extraction_doc = await self._run_db_call(
                    "db.get:extraction",
                    lambda ext_id=extraction_ext_id: backend._db.get(ext_id),
                    None,
                    warn=False,
                )
                if isinstance(extraction_doc, dict):
                    emeta = extraction_doc.get("metadata", {}) or {}
                    etext = str(
                        emeta.get("content")
                        or extraction_doc.get("text")
                        or "",
                    ).strip()
                    if etext:
                        desc = f"- {etext[:180]}"
                        if desc not in seen_descriptions:
                            seen_descriptions.add(desc)
                            belief_descriptions.append(desc)

            # Get the node's data
            doc = await self._run_db_call(
                "db.get:community_node",
                lambda external_id=ext_id: backend._db.get(external_id),
                None,
                warn=False,
            )
            if not doc:
                continue

            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else (doc.metadata if hasattr(doc, 'metadata') else {})
            text = doc.get("text", "") if isinstance(doc, dict) else (doc.text if hasattr(doc, 'text') else "")

            # If it's a belief triple (fields stored with _ prefix)
            subject = metadata.get("_subject", "") or metadata.get("subject", "")
            object_val = metadata.get("_object", "") or metadata.get("object", "")
            predicate = metadata.get("_predicate", "") or metadata.get("predicate", "")

            if subject:
                entities.add(subject)
            if object_val:
                entities.add(object_val)
            if subject and predicate and object_val:
                conf_raw = metadata.get("confidence", 0.5)
                try:
                    conf = max(0.0, min(1.0, float(conf_raw)))
                except Exception:
                    conf = 0.5
                _append_edge(
                    subject=str(subject),
                    predicate=str(predicate),
                    object_value=str(object_val),
                    confidence=conf,
                    source_doc_id=str(
                        metadata.get("source_doc_id")
                        or metadata.get("grounding_doc_id")
                        or ""
                    ),
                    temporal_point=str(metadata.get("observed_at", "")).split("T", 1)[0],
                )
            elif text:
                desc = f"- {text[:150]}"
                if desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    belief_descriptions.append(desc)

        if len(entities) < 2 or not belief_descriptions:
            logger.debug(
                "Skipping community: %d entities, %d descriptions (from %d ext_ids)",
                len(entities), len(belief_descriptions), len(external_ids),
            )
            return []

        known_triplet_keys = {
            self._triplet_key(edge.subject, edge.predicate, edge.object) for edge in edges
        }
        deterministic_candidates = self._generate_deterministic_hypotheses(edges)

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
        facts_str = "\n".join(belief_descriptions[:16])
        importance_str = "\n".join(importance_lines[:12]) or "- (PageRank not available)"

        prompt = _HYPOTHESIS_PROMPT.format(
            entities=entities_str,
            facts=facts_str,
            importance=importance_str,
        )

        llm_candidates: list[BeliefCandidate] = []

        # Call LLM
        try:
            llm_result = await self._llm.generate_structured(
                prompt,
                HypothesisSet,
                max_tokens=1200,
            )
        except Exception:
            try:
                raw = await self._llm.generate(
                    prompt, max_tokens=1200, temperature=0.5
                )
                llm_result = self._parse_raw(raw)
            except Exception as e:
                logger.warning("Hypothesis LLM call failed: %s", e)
                llm_result = HypothesisSet()

        # Convert to BeliefCandidates
        community_support = min(1.0, len(belief_descriptions) / 20.0)
        for h in llm_result.hypotheses:
            if not h.hypothesis or not h.subject or not h.object:
                continue
            subj_support = await self._entity_confidence(h.subject, local_conf_cache)
            obj_support = await self._entity_confidence(h.object, local_conf_cache)
            node_support = (subj_support + obj_support) / 2.0
            blended_conf = (
                0.6 * float(h.confidence)
                + 0.3 * node_support
                + 0.1 * community_support
            )
            llm_candidates.append(BeliefCandidate(
                id=uuid4(),
                content=f"[HYPOTHESIS] {h.hypothesis}",
                subject=h.subject,
                predicate=h.predicate or "hypothetically",
                object=h.object,
                confidence=min(0.8, max(0.1, blended_conf)),
                source_context={
                    "type": "hypothesis",
                    "evidence_summary": h.evidence_summary,
                    "community_size": len(entities),
                    "entities_in_cluster": sorted(entities)[:10],
                    "node_support": round(node_support, 4),
                    "community_support": round(community_support, 4),
                },
                reasoning=f"Generated from Louvain community with {len(entities)} entities",
            ))

        ranked = self._rank_candidates_by_information_gain(
            deterministic_candidates + llm_candidates,
            known_triplet_keys=known_triplet_keys,
        )
        return ranked[:20]

    def _generate_deterministic_hypotheses(
        self,
        edges: list[CommunityEdge],
    ) -> list[BeliefCandidate]:
        """Generate hypothesis candidates without an LLM.

        Operators:
        - Two-hop bridge discovery (A->B and B->C, but no direct A->C)
        - Contradiction discovery (same subject+predicate, conflicting objects)
        """
        if not edges:
            return []

        out_by_subject: dict[str, list[CommunityEdge]] = defaultdict(list)
        direct_pairs: set[tuple[str, str]] = set()
        for edge in edges:
            s_key = self._norm(edge.subject)
            o_key = self._norm(edge.object)
            out_by_subject[s_key].append(edge)
            direct_pairs.add((s_key, o_key))

        candidates: list[BeliefCandidate] = []
        seen_candidate_keys: set[str] = set()

        # Operator 1: bridge links through 2-hop paths.
        for edge in edges:
            mid_key = self._norm(edge.object)
            for next_edge in out_by_subject.get(mid_key, []):
                if self._norm(edge.subject) == self._norm(next_edge.object):
                    continue
                if (self._norm(edge.subject), self._norm(next_edge.object)) in direct_pairs:
                    continue
                subject = edge.subject
                mid = edge.object
                object_value = next_edge.object
                candidate = BeliefCandidate(
                    id=uuid4(),
                    content=(
                        f"[HYPOTHESIS] Hidden bridge: {subject} may connect to "
                        f"{object_value} via {mid}"
                    ),
                    subject=subject,
                    predicate="possibly_connected_via",
                    object=object_value,
                    confidence=max(
                        0.2,
                        min(0.75, min(edge.confidence, next_edge.confidence) * 0.7),
                    ),
                    source_context={
                        "type": "hypothesis",
                        "operator": "two_hop_bridge",
                        "bridge_entity": mid,
                        "path": [
                            f"{edge.subject} {edge.predicate} {edge.object}",
                            f"{next_edge.subject} {next_edge.predicate} {next_edge.object}",
                        ],
                        "path_support": round(min(edge.confidence, next_edge.confidence), 4),
                        "source_doc_id": edge.source_doc_id or next_edge.source_doc_id,
                    },
                    reasoning="Deterministic two-hop discovery found a missing direct link",
                )
                key = self._candidate_key(candidate)
                if key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(key)
                candidates.append(candidate)

        # Operator 2: conflicting object values on same (subject, predicate).
        by_sp: dict[tuple[str, str], list[CommunityEdge]] = defaultdict(list)
        for edge in edges:
            by_sp[(self._norm(edge.subject), self._norm(edge.predicate))].append(edge)
        for group in by_sp.values():
            object_map: dict[str, CommunityEdge] = {}
            for edge in group:
                o_key = self._norm(edge.object)
                current = object_map.get(o_key)
                if current is None or edge.confidence > current.confidence:
                    object_map[o_key] = edge
            if len(object_map) < 2:
                continue

            top = sorted(
                object_map.values(),
                key=lambda item: item.confidence,
                reverse=True,
            )[:2]
            left, right = top[0], top[1]
            subject = left.subject
            predicate = left.predicate
            object_value = f"{left.object} vs {right.object}"
            candidate = BeliefCandidate(
                id=uuid4(),
                content=(
                    f"[HYPOTHESIS] Conflict detected: {subject} {predicate} is reported as "
                    f"{left.object} and {right.object}"
                ),
                subject=subject,
                predicate="has_conflicting_claim_on",
                object=object_value,
                confidence=max(
                    0.25,
                    min(0.8, 0.35 + 0.25 * min(left.confidence, right.confidence)),
                ),
                source_context={
                    "type": "hypothesis",
                    "operator": "contradiction_scan",
                    "conflicting_predicate": predicate,
                    "conflicting_objects": [left.object, right.object],
                    "path_support": round(min(left.confidence, right.confidence), 4),
                    "source_doc_ids": [left.source_doc_id, right.source_doc_id],
                },
                reasoning="Deterministic contradiction scan found incompatible object values",
            )
            key = self._candidate_key(candidate)
            if key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(key)
            candidates.append(candidate)

        # Operator 3: temporal sequence discovery for same subject.
        by_subject: dict[str, list[tuple[tuple[int, int, int], CommunityEdge]]] = defaultdict(list)
        for edge in edges:
            parsed = None
            if edge.temporal_point:
                parsed = self._parse_date_key(edge.temporal_point)
            if parsed is None and self._is_temporal_predicate(edge.predicate):
                parsed = self._parse_date_key(edge.object)
            if parsed is None:
                continue
            by_subject[self._norm(edge.subject)].append((parsed, edge))

        for subject_key, dated_edges in by_subject.items():
            if len(dated_edges) < 2:
                continue
            dated_edges.sort(key=lambda item: item[0])
            earliest = dated_edges[0][1]
            latest = dated_edges[-1][1]
            if earliest.object == latest.object:
                continue
            candidate = BeliefCandidate(
                id=uuid4(),
                content=(
                    f"[HYPOTHESIS] Temporal progression: {earliest.subject} "
                    f"moved from {earliest.temporal_point or earliest.object} "
                    f"to {latest.temporal_point or latest.object}"
                ),
                subject=earliest.subject,
                predicate="timeline_progresses_from_to",
                object=(
                    f"{earliest.temporal_point or earliest.object} -> "
                    f"{latest.temporal_point or latest.object}"
                ),
                confidence=max(
                    0.25,
                    min(0.8, min(earliest.confidence, latest.confidence) * 0.8),
                ),
                source_context={
                    "type": "hypothesis",
                    "operator": "temporal_sequence",
                    "temporal_from": earliest.temporal_point or earliest.object,
                    "temporal_to": latest.temporal_point or latest.object,
                    "support_predicates": [earliest.predicate, latest.predicate],
                    "path_support": round(min(earliest.confidence, latest.confidence), 4),
                    "source_doc_ids": [earliest.source_doc_id, latest.source_doc_id],
                },
                reasoning="Deterministic temporal sequencing found ordered dated events for the same subject",
            )
            key = self._candidate_key(candidate)
            if key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(key)
            candidates.append(candidate)

        return candidates

    async def _generate_global_temporal_hypotheses(
        self,
        limit: int = 40,
    ) -> list[BeliefCandidate]:
        """Generate timeline progression hypotheses from all visible triples."""
        backend = self._memory._backend
        try:
            triples = backend._query_triples(k=max(limit * 40, 3000))
        except Exception:
            return []

        by_subject: dict[str, list[tuple[tuple[int, int, int], dict[str, Any]]]] = defaultdict(list)

        for t in triples:
            metadata = getattr(t, "metadata", {}) or {}
            external_id = getattr(t, "external_id", "")
            if not backend._can_access(metadata, _external_id=external_id):
                continue

            subject = str(getattr(t, "subject", "") or "").strip()
            predicate = str(getattr(t, "predicate", "") or "").strip()
            obj = str(getattr(t, "object_value", "") or "").strip()
            if not subject:
                continue

            temporal_point = ""
            source_doc_id = ""
            temporal_explicit = False
            relative_temporal = False

            source_meta = metadata.get("source_metadata")
            if isinstance(source_meta, dict):
                source_block = source_meta.get("source_document")
                if not isinstance(source_block, dict):
                    source_block = source_meta.get("source")
                if isinstance(source_block, dict):
                    source_doc_id = str(
                        source_block.get("document_id")
                        or source_block.get("title")
                        or "",
                    )
                object_date = source_meta.get("object_date")
                if isinstance(object_date, dict):
                    if bool(object_date.get("relative", False)):
                        relative_temporal = True
                    else:
                        temporal_point = str(object_date.get("canonical") or "").strip()
                        temporal_explicit = bool(temporal_point)
                if not temporal_point:
                    date_norm = source_meta.get("date_normalized")
                    if isinstance(date_norm, dict):
                        if bool(date_norm.get("relative", False)):
                            relative_temporal = True
                        else:
                            temporal_point = str(date_norm.get("canonical") or "").strip()
                            temporal_explicit = bool(temporal_point)

            if relative_temporal and not temporal_point:
                continue

            if not temporal_point:
                # Parse date-like object text only for temporal predicates.
                if self._is_temporal_predicate(predicate):
                    normalized_obj = normalize_date(obj)
                    if (
                        normalized_obj
                        and not normalized_obj.ambiguous
                        and not normalized_obj.relative
                    ):
                        temporal_point = normalized_obj.canonical
                        temporal_explicit = True
                    else:
                        temporal_point = obj
                        temporal_explicit = self._parse_date_key(temporal_point) is not None

            parsed = self._parse_date_key(temporal_point)
            if parsed is None or not temporal_explicit:
                continue

            try:
                confidence = max(0.0, min(1.0, float(getattr(t, "probability", 0.5))))
            except Exception:
                confidence = 0.5

            by_subject[self._norm(subject)].append(
                (
                    parsed,
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "temporal_point": temporal_point,
                        "confidence": confidence,
                        "source_doc_id": source_doc_id,
                    },
                ),
            )

        candidates: list[BeliefCandidate] = []
        for subject_key, items in by_subject.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            earliest = items[0][1]
            latest = items[-1][1]
            if earliest["temporal_point"] == latest["temporal_point"]:
                continue

            candidates.append(BeliefCandidate(
                id=uuid4(),
                content=(
                    f"[HYPOTHESIS] Timeline progression: {earliest['subject']} spans "
                    f"{earliest['temporal_point']} to {latest['temporal_point']}"
                ),
                subject=earliest["subject"],
                predicate="timeline_progresses_from_to",
                object=f"{earliest['temporal_point']} -> {latest['temporal_point']}",
                confidence=max(
                    0.25,
                    min(0.8, min(float(earliest["confidence"]), float(latest["confidence"])) * 0.85),
                ),
                source_context={
                    "type": "hypothesis",
                    "operator": "global_temporal_sequence",
                    "temporal_from": earliest["temporal_point"],
                    "temporal_to": latest["temporal_point"],
                    "support_predicates": [earliest["predicate"], latest["predicate"]],
                    "temporal_evidence_count": len(items),
                    "path_support": round(
                        min(float(earliest["confidence"]), float(latest["confidence"])),
                        4,
                    ),
                    "source_doc_ids": [
                        earliest.get("source_doc_id", ""),
                        latest.get("source_doc_id", ""),
                    ],
                },
                reasoning="Global temporal sequencing found multiple dated assertions for the same subject",
            ))
            if len(candidates) >= limit:
                break

        return candidates

    def _information_gain_score(
        self,
        candidate: BeliefCandidate,
        known_triplet_keys: set[str],
    ) -> tuple[float, dict[str, float]]:
        """Score a hypothesis by expected information gain."""
        candidate_key = self._candidate_key(candidate)
        novelty = 1.0 if candidate_key not in known_triplet_keys else 0.0

        ctx = candidate.source_context or {}
        support_values: list[float] = []
        for field_name in ("path_support", "node_support", "community_support"):
            raw = ctx.get(field_name)
            if raw is None:
                continue
            try:
                support_values.append(max(0.0, min(1.0, float(raw))))
            except Exception:
                continue
        evidential_support = (
            sum(support_values) / len(support_values)
            if support_values else max(0.0, min(1.0, float(candidate.confidence)))
        )

        # Highest when confidence is near 0.5: likely high uncertainty reduction.
        uncertainty_potential = 1.0 - min(1.0, abs(float(candidate.confidence) - 0.5) * 2.0)

        actionability = 0.6
        if self._norm(candidate.predicate or "") in {
            "possibly_connected_via",
            "has_conflicting_claim_on",
            "timeline_progresses_from_to",
        }:
            actionability = 0.9

        score = (
            0.35 * novelty
            + 0.30 * evidential_support
            + 0.20 * uncertainty_potential
            + 0.15 * actionability
        )
        components = {
            "novelty": round(novelty, 4),
            "evidential_support": round(evidential_support, 4),
            "uncertainty_potential": round(uncertainty_potential, 4),
            "actionability": round(actionability, 4),
        }
        return max(0.0, min(1.0, score)), components

    def _rank_candidates_by_information_gain(
        self,
        candidates: list[BeliefCandidate],
        known_triplet_keys: set[str],
    ) -> list[BeliefCandidate]:
        """Deduplicate and rank hypothesis candidates by information gain."""
        best_by_key: dict[str, BeliefCandidate] = {}
        best_score: dict[str, float] = {}

        for candidate in candidates:
            if not candidate.subject or not candidate.object:
                continue
            score, components = self._information_gain_score(
                candidate,
                known_triplet_keys=known_triplet_keys,
            )
            if candidate.source_context is None:
                candidate.source_context = {}
            candidate.source_context["information_gain"] = round(score, 4)
            candidate.source_context["info_gain_components"] = components

            key = self._candidate_key(candidate)
            previous = best_score.get(key, -1.0)
            if score > previous:
                best_score[key] = score
                best_by_key[key] = candidate

        ranked = list(best_by_key.values())
        ranked.sort(
            key=lambda c: (
                float((c.source_context or {}).get("information_gain", 0.0)),
                float(c.confidence),
            ),
            reverse=True,
        )
        return ranked

    async def _entity_confidence(
        self,
        entity: str,
        cache: dict[str, float],
    ) -> float:
        """Estimate support for an entity from nearby belief probabilities."""
        key = entity.strip().lower()
        if not key:
            return 0.5
        if key in cache:
            return cache[key]
        try:
            beliefs = await self._memory._backend.get_beliefs_by_entity(entity)
        except Exception:
            beliefs = []
        if not beliefs:
            cache[key] = 0.5
            return cache[key]
        probs = [max(0.0, min(1.0, float(b.confidence))) for b in beliefs[:40]]
        if not probs:
            cache[key] = 0.5
            return cache[key]
        # Weighted toward stronger evidence while still averaging neighbors.
        score = (sum(probs) / len(probs) + max(probs)) / 2.0
        cache[key] = max(0.0, min(1.0, score))
        return cache[key]

    async def _discover_outward_ids(
        self,
        seed_external_ids: list[str],
        max_hops: int = 2,
        max_nodes: int = 120,
    ) -> list[str]:
        """Expand a community outward via graph neighborhood discovery.

        Prefers entity_neighbors() (typed edges with target_entity field)
        when available. Falls back to db.neighbors() with manual key-guessing.
        """
        storage = self._memory._storage
        seen: set[str] = {x for x in seed_external_ids if x}
        frontier: set[str] = set(seen)
        use_native = True  # Will flip to False if entity_neighbors fails

        for _ in range(max_hops):
            if not frontier or len(seen) >= max_nodes:
                break
            nxt: set[str] = set()
            for current in list(frontier):
                for direction in ("outgoing", "incoming"):
                    neighbors: list[Any] = []

                    if use_native:
                        try:
                            neighbors = await self._run_db_call(
                                f"entity_neighbors:{direction}",
                                lambda node=current, dir_name=direction: (
                                    storage._db.entity_neighbors(
                                        name=node, direction=dir_name,
                                    )
                                ),
                                None,
                                warn=False,
                            )
                            if neighbors is None:
                                use_native = False
                                neighbors = []
                        except Exception:
                            use_native = False
                            neighbors = []

                    if not use_native:
                        # Fallback: db.neighbors() with manual key extraction
                        neighbors = await self._run_db_call(
                            f"neighbors:{direction}",
                            lambda node=current, dir_name=direction: (
                                storage._db.neighbors(node, direction=dir_name) or []
                            ),
                            [],
                            warn=False,
                        )

                    for node in neighbors:
                        # Extract neighbor ID: native API uses typed fields,
                        # fallback uses dict key-guessing
                        nid = ""
                        if use_native:
                            nid = (
                                getattr(node, "target_entity", "")
                                or (node.get("target_entity", "") if isinstance(node, dict) else "")
                                or getattr(node, "external_id", "")
                                or (node.get("external_id", "") if isinstance(node, dict) else "")
                            )
                        else:
                            if isinstance(node, dict):
                                for key in (
                                    "external_id", "id", "to_id", "from_id",
                                    "target", "source", "neighbor_id",
                                ):
                                    value = node.get(key)
                                    if isinstance(value, str) and value and value != current:
                                        nid = value
                                        break

                        if not nid or nid in seen:
                            continue
                        seen.add(nid)
                        nxt.add(nid)
                        if len(seen) >= max_nodes:
                            break
                    if len(seen) >= max_nodes:
                        break
                if len(seen) >= max_nodes:
                    break
            frontier = nxt

        return list(seen)

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
                query="*", limit=300, min_confidence=0.3
            )
        except Exception as e:
            logger.warning("Fallback belief query failed: %s", e)
            return result

        if not beliefs:
            try:
                beliefs = await backend.query_beliefs(
                    query="the", limit=300, min_confidence=0.3,
                )
            except Exception:
                beliefs = []

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
