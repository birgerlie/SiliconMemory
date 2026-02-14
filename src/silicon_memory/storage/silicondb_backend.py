"""SiliconDB backend - all memory operations backed by SiliconDB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now
from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Experience,
    KnowledgeProof,
    Procedure,
    RecallResult,
    Source,
    SourceType,
    TemporalContext,
    Triplet,
)
from silicon_memory.core.decision import Decision, DecisionStatus
from silicon_memory.core.exceptions import AuthorizationError, TenantIsolationError
from silicon_memory.security.types import (
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)
from silicon_memory.security.authorization import Permission, PolicyEngine
from silicon_memory.temporal.decay import DecayConfig, apply_decay


@dataclass
class SiliconDBConfig:
    """Configuration for SiliconDB backend."""

    path: str | Path
    language: str = "english"
    enable_graph: bool = True
    enable_async: bool = True
    auto_embedder: bool = True
    embedder_model: str = "base"


class SiliconDBBackend:
    """Unified memory backend using SiliconDB.

    All memory operations (semantic, episodic, procedural, working) are
    stored in SiliconDB using different node_types and metadata.

    Memory types are distinguished by:
    - semantic: Triples with node_type="belief"
    - episodic: Documents with node_type="experience"
    - procedural: Documents with node_type="procedure"
    - working: Documents with node_type="working" + ttl metadata

    SiliconDB provides:
    - Memory-mapped files for fast access
    - WAL for durability
    - Auto-embedding with E5
    - Belief system with probabilities
    - Contradiction detection
    - Graph relationships

    Multi-user security:
    - All operations require UserContext
    - External IDs: {tenant_id}/{user_id}/{type}-{uuid}
    - Access filtering based on privacy levels and ABAC policies
    """

    # Node types for different memory layers
    NODE_TYPE_BELIEF = "belief"
    NODE_TYPE_EXPERIENCE = "experience"
    NODE_TYPE_PROCEDURE = "procedure"
    NODE_TYPE_WORKING = "working"

    def __init__(
        self,
        config: SiliconDBConfig,
        user_context: UserContext,
        decay_config: DecayConfig | None = None,
    ) -> None:
        """Initialize the SiliconDB backend.

        Args:
            config: Database configuration
            user_context: Required user context for all operations
            decay_config: Optional decay configuration
        """
        if not user_context:
            raise ValueError("user_context is required")

        try:
            from silicondb import SiliconDB
        except ImportError as e:
            raise ImportError(
                "SiliconDB is required. Install with: pip install silicondb"
            ) from e

        self._db = SiliconDB(
            path=str(config.path),
            language=config.language,
            enable_graph=config.enable_graph,
            enable_async=config.enable_async,
            auto_embedder=config.auto_embedder,
            embedder_model=config.embedder_model,
        )
        self._decay_config = decay_config or DecayConfig()
        self._user_context = user_context
        self._policy_engine = PolicyEngine()
        # Track working memory keys in-process (workaround for SiliconDB #125:
        # empty-query search with metadata filter returns no results)
        self._working_keys: set[str] = set()

    def _build_external_id(self, entity_type: str, entity_id: UUID | str) -> str:
        """Build the new external ID format: {tenant_id}/{user_id}/{type}-{uuid}."""
        return f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_type}-{entity_id}"

    def _get_user_prefix(self) -> str:
        """Get the prefix for the current user's documents."""
        return f"{self._user_context.tenant_id}/{self._user_context.user_id}/"

    def _can_access(self, metadata: dict[str, Any], permission: Permission = Permission.READ) -> bool:
        """Check if current user can access a document based on its metadata."""
        # Extract privacy metadata
        owner_id = metadata.get("owner_id")
        tenant_id = metadata.get("tenant_id")
        privacy_level = metadata.get("privacy_level", "private")
        shared_with = metadata.get("shared_with", [])

        # Owner always has access
        if owner_id == self._user_context.user_id:
            return True

        # Check tenant isolation
        if tenant_id and tenant_id != self._user_context.tenant_id:
            # Different tenant - only public is accessible
            return privacy_level == "public"

        # Admin has full access within tenant
        if self._user_context.is_admin():
            return True

        # Check privacy level
        if privacy_level == "public":
            return True

        if privacy_level == "workspace":
            return self._user_context.can_access_workspace()

        # Private - check if explicitly shared
        return self._user_context.user_id in shared_with

    def _create_privacy_metadata(
        self,
        privacy_level: PrivacyLevel | None = None,
        classification: DataClassification = DataClassification.INTERNAL,
    ) -> dict[str, Any]:
        """Create privacy metadata for a new document."""
        privacy = PrivacyMetadata.create_for_user(
            self._user_context,
            privacy_level=privacy_level,
            classification=classification,
        )
        return privacy.to_dict()

    def close(self) -> None:
        """Close the database."""
        self._db.close()

    def __enter__(self) -> "SiliconDBBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ========== Search Result Helpers ==========

    @staticmethod
    def _rget(r, attr: str, default=None):
        """Get attribute from a search result (dict or object)."""
        if isinstance(r, dict):
            return r.get(attr, default)
        return getattr(r, attr, default)

    # ========== Semantic Memory (Beliefs/Triplets) ==========

    async def commit_belief(self, belief: Belief) -> None:
        """Store a belief as a SiliconDB triple."""
        external_id = self._build_external_id("belief", belief.id)

        # Build sources dict for SiliconDB
        sources = None
        if belief.source:
            sources = {belief.source.id: belief.source.reliability}

        # Build metadata with privacy info
        privacy_meta = self._create_privacy_metadata(
            privacy_level=belief.privacy.privacy_level if belief.privacy else None,
        )

        metadata = {
            "belief_id": str(belief.id),
            "status": belief.status.value,
            "tags": list(belief.tags) if isinstance(belief.tags, set) else belief.tags,
            "evidence_for": [str(e) for e in belief.evidence_for],
            "evidence_against": [str(e) for e in belief.evidence_against],
            **privacy_meta,
        }

        if belief.temporal:
            metadata["observed_at"] = belief.temporal.observed_at.isoformat()
            if belief.temporal.valid_from:
                metadata["valid_from"] = belief.temporal.valid_from.isoformat()
            if belief.temporal.valid_until:
                metadata["valid_until"] = belief.temporal.valid_until.isoformat()
            if belief.temporal.last_verified:
                metadata["last_verified"] = belief.temporal.last_verified.isoformat()

        if belief.triplet:
            # Store as triple
            self._db.insert_triple(
                external_id=external_id,
                subject=belief.triplet.subject,
                predicate=belief.triplet.predicate,
                object_value=belief.triplet.object,
                probability=belief.confidence,
                sources=sources,
                metadata=metadata,
            )
        else:
            # Store as document with belief content
            self._db.ingest(
                external_id=external_id,
                text=belief.content or "",
                metadata=metadata,
                node_type=self.NODE_TYPE_BELIEF,
                probability=belief.confidence,
                sources=list(sources.keys()) if sources else None,
            )

    async def get_belief(self, belief_id: UUID) -> Belief | None:
        """Get a belief by ID."""
        external_id = self._build_external_id("belief", belief_id)

        # Try triple first
        triples = self._db.query_triples(k=1)
        for t in triples:
            if t.external_id == external_id:
                # Check access
                if not self._can_access(t.metadata or {}):
                    return None
                return self._triple_to_belief(t)

        # Try document
        try:
            doc = self._db.get(external_id)
            if doc:
                metadata = doc.get("metadata", {})
                if not self._can_access(metadata):
                    return None
                return self._doc_to_belief(doc)
        except Exception:
            pass

        return None

    async def query_beliefs(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        include_contested: bool = False,
        search_weights: dict[str, Any] | None = None,
    ) -> list[Belief]:
        """Query beliefs by semantic similarity."""
        # Query triples
        results = []
        user_prefix = self._get_user_prefix()

        # Use SiliconDB search for semantic similarity
        search_results = self._weighted_search(query, k=limit * 4, weights_dict=search_weights)

        for r in search_results:
            if self._rget(r, "node_type") != self.NODE_TYPE_BELIEF:
                continue
            if self._rget(r, "probability", 1.0) < min_confidence:
                continue

            metadata = self._rget(r, "metadata") or {}

            # Access control check
            if not self._can_access(metadata):
                continue

            status = BeliefStatus(metadata.get("status", "provisional"))

            if status == BeliefStatus.REJECTED:
                continue
            if status == BeliefStatus.CONTESTED and not include_contested:
                continue

            belief = self._search_result_to_belief(r)
            if belief:
                results.append(belief)

            if len(results) >= limit:
                break

        # Also query triples
        triples = self._db.query_triples(min_probability=min_confidence, k=limit * 2)
        for t in triples:
            # Access control check
            if not self._can_access(self._rget(t, "metadata") or {}):
                continue
            belief = self._triple_to_belief(t)
            if belief and belief.id not in {b.id for b in results}:
                results.append(belief)

        return results[:limit]

    async def get_beliefs_by_entity(self, entity: str) -> list[Belief]:
        """Get all beliefs about an entity."""
        # Query triples where entity is subject or object
        as_subject = self._db.query_triples(subject=entity, k=100)
        as_object = self._db.query_triples(object_value=entity, k=100)

        beliefs = []
        seen_ids = set()

        for t in as_subject + as_object:
            # Access control check
            if not self._can_access(self._rget(t, "metadata") or {}):
                continue
            belief = self._triple_to_belief(t)
            if belief and belief.id not in seen_ids:
                beliefs.append(belief)
                seen_ids.add(belief.id)

        return beliefs

    async def find_contradictions(self, belief: Belief) -> list[Belief]:
        """Find beliefs that contradict the given belief."""
        if not belief.triplet:
            return []

        # Use SiliconDB's contradiction detection
        contradictions = self._db.detect_triple_contradictions(min_probability=0.0)

        results = []
        for c in contradictions:
            if (c.subject.lower() == belief.triplet.subject.lower() and
                c.predicate.lower() == belief.triplet.predicate.lower()):
                for obj in c.conflicting_objects:
                    if obj.object_value.lower() != belief.triplet.object.lower():
                        # This is a contradicting belief
                        contra_belief = await self.get_belief(
                            UUID(obj.external_id.replace("belief-", ""))
                        )
                        if contra_belief:
                            results.append(contra_belief)

        return results

    async def update_belief_confidence(
        self,
        belief_id: UUID,
        delta: float,
    ) -> bool:
        """Update a belief's confidence using Bayesian observation."""
        external_id = self._build_external_id("belief", belief_id)

        # Use SiliconDB's record_observation for Bayesian updates
        try:
            self._db.record_observation(
                external_id=external_id,
                confirmed=delta > 0,
                source="confidence_update",
            )
            return True
        except Exception:
            return False

    async def _query_beliefs_with_entropy(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        include_contested: bool = False,
        search_weights: dict[str, Any] | None = None,
    ) -> list[tuple[Belief, float]]:
        """Like query_beliefs but also returns per-result entropy.

        Used internally by ``recall()`` so entropy can be carried into
        ``RecallResult`` for post-retrieval reranking.
        """
        results: list[tuple[Belief, float]] = []

        search_results = self._weighted_search(query, k=limit * 4, weights_dict=search_weights)

        for r in search_results:
            if self._rget(r, "node_type") != self.NODE_TYPE_BELIEF:
                continue
            if self._rget(r, "probability", 1.0) < min_confidence:
                continue
            metadata = self._rget(r, "metadata") or {}
            if not self._can_access(metadata):
                continue
            status = BeliefStatus(metadata.get("status", "provisional"))
            if status == BeliefStatus.REJECTED:
                continue
            if status == BeliefStatus.CONTESTED and not include_contested:
                continue
            belief = self._search_result_to_belief(r)
            if belief:
                entropy = self._rget(r, "entropy", 0.0) or 0.0
                results.append((belief, entropy))
            if len(results) >= limit:
                break

        return results

    # ========== Episodic Memory (Experiences) ==========

    async def record_experience(self, experience: Experience) -> None:
        """Record an experience."""
        external_id = self._build_external_id("experience", experience.id)

        # Build metadata with privacy info
        privacy_meta = self._create_privacy_metadata(
            privacy_level=experience.privacy.privacy_level if experience.privacy else None,
        )

        metadata = {
            "experience_id": str(experience.id),
            "occurred_at": experience.occurred_at.isoformat(),
            "outcome": experience.outcome,
            "processed": experience.processed,
            "session_id": experience.session_id,
            "sequence_id": experience.sequence_id,
            "tags": list(experience.tags) if isinstance(experience.tags, set) else experience.tags,
            **privacy_meta,
        }

        if experience.causal_parent:
            metadata["causal_parent"] = str(experience.causal_parent)

        self._db.ingest(
            external_id=external_id,
            text=experience.content,
            metadata=metadata,
            node_type=self.NODE_TYPE_EXPERIENCE,
        )

        # Add causal edge if parent exists
        if experience.causal_parent:
            parent_id = self._build_external_id("experience", experience.causal_parent)
            try:
                self._db.add_edge(
                    from_id=parent_id,
                    to_id=external_id,
                    edge_type="causes",
                )
            except Exception:
                pass  # Parent may not exist yet

    async def get_experience(self, experience_id: UUID) -> Experience | None:
        """Get an experience by ID."""
        external_id = self._build_external_id("experience", experience_id)
        try:
            doc = self._db.get(external_id)
            if doc:
                metadata = doc.get("metadata", {})
                if not self._can_access(metadata):
                    return None
                return self._doc_to_experience(doc)
        except Exception:
            pass
        return None

    async def query_experiences(
        self,
        query: str,
        limit: int = 10,
        search_weights: dict[str, Any] | None = None,
    ) -> list[Experience]:
        """Query experiences by semantic similarity."""
        results = self._weighted_search(query, k=limit * 4, weights_dict=search_weights)

        experiences = []
        for r in results:
            if self._rget(r, "node_type") != self.NODE_TYPE_EXPERIENCE:
                continue
            # Access control check
            if not self._can_access(self._rget(r, "metadata") or {}):
                continue
            exp = self._search_result_to_experience(r)
            if exp:
                experiences.append(exp)
            if len(experiences) >= limit:
                break

        return experiences

    async def get_recent_experiences(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[Experience]:
        """Get recent experiences."""
        cutoff = utc_now() - timedelta(hours=hours)

        # Search with time filter in metadata
        # SiliconDB supports metadata filtering
        results = self._db.search(
            query="",  # Empty query to get all
            k=limit * 4,
            filter={"node_type": self.NODE_TYPE_EXPERIENCE},
        )

        experiences = []
        for r in results:
            if self._rget(r, "node_type") != self.NODE_TYPE_EXPERIENCE:
                continue
            # Access control check
            if not self._can_access(self._rget(r, "metadata") or {}):
                continue
            exp = self._search_result_to_experience(r)
            if exp and exp.occurred_at >= cutoff:
                experiences.append(exp)

        # Sort by recency
        experiences.sort(key=lambda e: e.occurred_at, reverse=True)
        return experiences[:limit]

    async def get_unprocessed_experiences(self, limit: int = 100) -> list[Experience]:
        """Get unprocessed experiences for reflection."""
        results = self._db.search(
            query="",
            k=limit * 4,
            filter={"processed": False},
        )

        experiences = []
        for r in results:
            if self._rget(r, "node_type") != self.NODE_TYPE_EXPERIENCE:
                continue
            # Access control check
            if not self._can_access(self._rget(r, "metadata") or {}):
                continue
            exp = self._search_result_to_experience(r)
            if exp and not exp.processed:
                experiences.append(exp)

        experiences.sort(key=lambda e: e.occurred_at)
        return experiences[:limit]

    async def mark_experiences_processed(self, experience_ids: list[UUID]) -> None:
        """Mark experiences as processed."""
        for eid in experience_ids:
            external_id = self._build_external_id("experience", eid)
            try:
                self._db.update(external_id, metadata={"processed": True})
            except Exception:
                pass

    # ========== Procedural Memory (Procedures) ==========

    async def commit_procedure(self, procedure: Procedure) -> None:
        """Store a procedure."""
        external_id = self._build_external_id("procedure", procedure.id)

        # Combine name, description, and steps for searchable text
        text_parts = [procedure.name]
        if procedure.description:
            text_parts.append(procedure.description)
        if procedure.trigger:
            text_parts.append(f"Trigger: {procedure.trigger}")
        text_parts.extend(f"Step {i+1}: {s}" for i, s in enumerate(procedure.steps))

        # Build metadata with privacy info
        privacy_meta = self._create_privacy_metadata(
            privacy_level=procedure.privacy.privacy_level if procedure.privacy else None,
        )

        metadata = {
            "procedure_id": str(procedure.id),
            "name": procedure.name,
            "description": procedure.description,
            "steps": procedure.steps,
            "trigger": procedure.trigger,
            "success_count": procedure.success_count,
            "failure_count": procedure.failure_count,
            "tags": list(procedure.tags) if isinstance(procedure.tags, set) else procedure.tags,
            **privacy_meta,
        }

        if procedure.source:
            metadata["source_id"] = procedure.source.id

        self._db.ingest(
            external_id=external_id,
            text="\n".join(text_parts),
            metadata=metadata,
            node_type=self.NODE_TYPE_PROCEDURE,
            probability=procedure.confidence,
        )

    async def get_procedure(self, procedure_id: UUID) -> Procedure | None:
        """Get a procedure by ID."""
        external_id = self._build_external_id("procedure", procedure_id)
        try:
            doc = self._db.get(external_id)
            if doc:
                metadata = doc.get("metadata", {})
                if not self._can_access(metadata):
                    return None
                return self._doc_to_procedure(doc)
        except Exception:
            pass
        return None

    async def find_applicable_procedures(
        self,
        context: str,
        limit: int = 5,
        min_confidence: float = 0.0,
        search_weights: dict[str, Any] | None = None,
    ) -> list[Procedure]:
        """Find procedures applicable to the context."""
        results = self._weighted_search(context, k=limit * 4, weights_dict=search_weights)

        procedures = []
        for r in results:
            if self._rget(r, "node_type") != self.NODE_TYPE_PROCEDURE:
                continue
            if self._rget(r, "probability", 1.0) < min_confidence:
                continue
            # Access control check
            if not self._can_access(self._rget(r, "metadata") or {}):
                continue
            proc = self._search_result_to_procedure(r)
            if proc:
                procedures.append(proc)
            if len(procedures) >= limit:
                break

        return procedures

    async def record_procedure_outcome(
        self,
        procedure_id: UUID,
        success: bool,
    ) -> bool:
        """Record an outcome for a procedure."""
        external_id = self._build_external_id("procedure", procedure_id)

        try:
            doc = self._db.get(external_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            if success:
                metadata["success_count"] = metadata.get("success_count", 0) + 1
            else:
                metadata["failure_count"] = metadata.get("failure_count", 0) + 1

            # Update confidence based on success rate
            total = metadata["success_count"] + metadata["failure_count"]
            if total > 0:
                # Bayesian-style confidence update
                self._db.record_observation(external_id, confirmed=success)

            self._db.update(external_id, metadata=metadata)
            return True
        except Exception:
            return False

    # ========== Working Memory (TTL-based) ==========

    async def set_working(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300,
    ) -> None:
        """Set a working memory value with TTL."""
        external_id = self._build_external_id("working", key)
        expires_at = utc_now() + timedelta(seconds=ttl_seconds)

        # Build metadata with privacy info
        privacy_meta = self._create_privacy_metadata()

        metadata = {
            "key": key,
            "value": value,
            "expires_at": expires_at.isoformat(),
            "created_at": utc_now().isoformat(),
            **privacy_meta,
        }

        # Use ingest to store (will update if exists)
        try:
            self._db.update(
                external_id=external_id,
                text=str(value),
                metadata=metadata,
            )
        except Exception:
            # Document doesn't exist, create it
            self._db.ingest(
                external_id=external_id,
                text=str(value),
                metadata=metadata,
                node_type=self.NODE_TYPE_WORKING,
            )

        self._working_keys.add(key)

    async def get_working(self, key: str) -> Any | None:
        """Get a working memory value if not expired."""
        external_id = self._build_external_id("working", key)

        try:
            doc = self._db.get(external_id)
            if not doc:
                return None

            metadata = doc.get("metadata", {})

            # Access control check
            if not self._can_access(metadata):
                return None

            expires_at_str = metadata.get("expires_at")

            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if utc_now() > expires_at:
                    # Expired - delete and return None
                    self._db.delete(external_id)
                    return None

            return metadata.get("value")
        except Exception:
            return None

    async def delete_working(self, key: str) -> bool:
        """Delete a working memory value."""
        external_id = self._build_external_id("working", key)
        try:
            self._db.delete(external_id)
            self._working_keys.discard(key)
            return True
        except Exception:
            return False

    async def get_all_working(self) -> dict[str, Any]:
        """Get all non-expired working memory values.

        Uses in-process key tracking to iterate over known keys via
        direct get(), working around SiliconDB #125 (empty-query search
        with metadata filter returns no results).
        """
        result = {}
        expired_keys = []

        for key in self._working_keys:
            value = await self.get_working(key)
            if value is not None:
                result[key] = value
            else:
                expired_keys.append(key)

        # Clean up expired keys from tracking set
        for key in expired_keys:
            self._working_keys.discard(key)

        return result

    async def cleanup_expired_working(self) -> int:
        """Clean up expired working memory entries. Returns count deleted.

        Uses in-process key tracking (workaround for SiliconDB #125).
        """
        now = utc_now()
        deleted = 0
        expired_keys = []

        for key in self._working_keys:
            external_id = self._build_external_id("working", key)
            try:
                doc = self._db.get(external_id)
                if not doc:
                    expired_keys.append(key)
                    continue
                metadata = doc.get("metadata", {})
                expires_at_str = metadata.get("expires_at")
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if now > expires_at:
                        self._db.delete(external_id)
                        expired_keys.append(key)
                        deleted += 1
            except Exception:
                pass

        for key in expired_keys:
            self._working_keys.discard(key)

        return deleted

    # ========== Knowledge Proofs ==========

    async def build_knowledge_proof(
        self,
        query: str,
        min_confidence: float = 0.3,
    ) -> KnowledgeProof:
        """Build a knowledge proof for 'what do you know' queries."""
        now = utc_now()

        # Query beliefs
        beliefs = await self.query_beliefs(
            query,
            limit=50,
            min_confidence=min_confidence,
            include_contested=True,
        )

        # Collect sources
        sources: list[Source] = []
        seen_sources: set[str] = set()
        for b in beliefs:
            if b.source and b.source.id not in seen_sources:
                sources.append(b.source)
                seen_sources.add(b.source.id)

        # Find contradictions using SiliconDB
        contradictions: list[tuple[Belief, Belief]] = []
        db_contradictions = self._db.detect_triple_contradictions(
            min_probability=min_confidence
        )

        for c in db_contradictions:
            # Match contradictions to our beliefs
            belief_ids = {str(b.id) for b in beliefs}
            conflict_beliefs = []
            conflicting = self._rget(c, "conflicting_objects") or []
            for obj in conflicting:
                ext_id = self._rget(obj, "external_id", "")
                bid = ext_id.replace("belief-", "")
                if bid in belief_ids:
                    for b in beliefs:
                        if str(b.id) == bid:
                            conflict_beliefs.append(b)
                            break

            # Create pairs
            for i, b1 in enumerate(conflict_beliefs):
                for b2 in conflict_beliefs[i + 1:]:
                    contradictions.append((b1, b2))

        # Check temporal validity
        temporal_validity = {}
        for b in beliefs:
            if b.temporal:
                temporal_validity[b.id] = b.temporal.is_valid_at(now)
            else:
                temporal_validity[b.id] = True

        # Build evidence summary
        evidence_summary = {
            b.id: {
                "for": len(b.evidence_for),
                "against": len(b.evidence_against),
            }
            for b in beliefs
        }

        # Calculate total confidence with decay
        if beliefs:
            total = 0.0
            for b in beliefs:
                conf = b.confidence
                if b.temporal:
                    age = b.temporal.age_seconds(now)
                    conf = apply_decay(conf, age, self._decay_config)
                total += conf
            total_confidence = total / len(beliefs)
        else:
            total_confidence = 0.0

        return KnowledgeProof(
            query=query,
            beliefs=beliefs,
            total_confidence=total_confidence,
            sources=sources,
            contradictions=contradictions,
            temporal_validity=temporal_validity,
            evidence_summary=evidence_summary,
        )

    # ========== Unified Recall ==========

    async def recall(
        self,
        query: str,
        max_facts: int = 20,
        max_experiences: int = 10,
        max_procedures: int = 5,
        min_confidence: float = 0.3,
        include_working: bool = True,
        search_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Unified recall across all memory types.

        Args:
            search_weights: Optional weight dict for salience-weighted retrieval.
                Keys may include: vector, text, temporal, confidence, graph_proximity.
                Passed through to SiliconDB search when available.
        """
        now = utc_now()

        # Query all memory types, forwarding search_weights
        facts_with_entropy = await self._query_beliefs_with_entropy(
            query, limit=max_facts, min_confidence=min_confidence,
            search_weights=search_weights,
        )
        experiences = await self.query_experiences(
            query, limit=max_experiences, search_weights=search_weights,
        )
        procedures = await self.find_applicable_procedures(
            query, limit=max_procedures, search_weights=search_weights,
        )

        # Get working context
        working_context = {}
        if include_working:
            working_context = await self.get_all_working()

        # Build recall results with decay applied
        fact_results = []
        for b, ent in facts_with_entropy:
            conf = b.confidence
            if b.temporal:
                age = b.temporal.age_seconds(now)
                conf = apply_decay(conf, age, self._decay_config)

            fact_results.append(RecallResult(
                content=b.content or (b.triplet.as_text() if b.triplet else ""),
                confidence=conf,
                source=b.source,
                memory_type="semantic",
                relevance_score=conf,
                temporal=b.temporal,
                belief_id=b.id,
                triplet=b.triplet,
                evidence_count=b.evidence_count,
                entropy=ent,
            ))

        exp_results = []
        for e in experiences:
            recency = self._recency_score(e.occurred_at, now)
            exp_results.append(RecallResult(
                content=f"{e.content} → {e.outcome or 'no outcome'}",
                confidence=0.9,
                source=None,
                memory_type="episodic",
                relevance_score=recency,
            ))

        proc_results = []
        for p in procedures:
            proc_results.append(RecallResult(
                content=f"{p.name}: {' → '.join(p.steps)}",
                confidence=p.confidence,
                source=p.source,
                memory_type="procedural",
                relevance_score=p.confidence * p.success_rate,
            ))

        # Post-retrieval entropy reranking
        entropy_weight = (search_weights or {}).get("entropy_weight", 0)
        if entropy_weight > 0:
            entropy_direction = (search_weights or {}).get("entropy_direction", "prefer_low")
            fact_results = self._apply_entropy_reranking(
                fact_results, entropy_weight, entropy_direction,
            )

        return {
            "facts": fact_results,
            "experiences": exp_results,
            "procedures": proc_results,
            "working_context": working_context,
            "total_items": len(fact_results) + len(exp_results) + len(proc_results),
            "query": query,
            "as_of": now,
        }

    # ========== Helper Methods ==========

    # Keys from SearchWeights dataclass that the SiliconDB constructor accepts.
    _SEARCH_WEIGHT_KEYS = frozenset({
        "vector", "text", "temporal", "confidence", "graph_proximity",
        "ppr_damping_factor", "ppr_iterations", "temporal_half_life_hours",
        "fusion", "rrf_k",
    })

    def _build_search_weights(self, weights_dict: dict[str, Any] | None):
        """Build a SiliconDB ``SearchWeights`` from a plain dict.

        Filters out keys not accepted by the dataclass (e.g.
        ``entropy_weight``, ``entropy_direction``, ``graph_context_nodes``)
        so the remaining kwargs are safe to unpack.

        Returns ``None`` when *weights_dict* is falsy.
        """
        if not weights_dict:
            return None
        from silicondb.types import SearchWeights

        filtered = {k: v for k, v in weights_dict.items() if k in self._SEARCH_WEIGHT_KEYS}
        return SearchWeights(**filtered) if filtered else None

    def _weighted_search(
        self,
        query: str,
        k: int,
        weights_dict: dict[str, Any] | None = None,
        **extra_kwargs,
    ) -> list:
        """Search SiliconDB, routing to the scored endpoint when weights
        require extended features.

        Works with both the high-level ``SiliconDB`` (which accepts a
        ``weights`` kwarg) and the low-level ``SiliconDBNative`` (which
        needs an explicit ``search_scored`` call with a JSON payload).
        """
        import json as _json

        sw = self._build_search_weights(weights_dict)

        # High-level API accepts `weights` directly
        if sw is not None and hasattr(self._db.search, '__func__'):
            # Check if the search method accepts 'weights' (high-level API)
            import inspect
            sig = inspect.signature(self._db.search)
            if "weights" in sig.parameters:
                return self._db.search(query=query, k=k, weights=sw, **extra_kwargs)

        # Fall back: if we have extended weights, route via search_scored
        if sw is not None and hasattr(self._db, "search_scored"):
            needs_scored = (
                sw.confidence > 0
                or sw.graph_proximity > 0
                or sw.temporal > 0
                or getattr(sw, "fusion", "weighted_sum") != "weighted_sum"
            )
            if needs_scored:
                scoring = {
                    "vector": sw.vector,
                    "text": sw.text,
                    "temporal": sw.temporal,
                    "confidence": sw.confidence,
                    "graph_proximity": sw.graph_proximity,
                    "ppr_damping_factor": sw.ppr_damping_factor,
                    "ppr_iterations": sw.ppr_iterations,
                    "temporal_half_life_hours": sw.temporal_half_life_hours,
                    "fusion": getattr(sw, "fusion", "weighted_sum"),
                }
                return self._db.search_scored(
                    query=query, k=k,
                    scoring_json=_json.dumps(scoring),
                    **extra_kwargs,
                )
            # Only basic vector/text weights — use plain search
            return self._db.search(
                query=query, k=k,
                vector_weight=sw.vector, text_weight=sw.text,
                **extra_kwargs,
            )

        # No weights at all — plain search
        return self._db.search(query=query, k=k, **extra_kwargs)

    @staticmethod
    def _apply_entropy_reranking(
        results: list[RecallResult],
        entropy_weight: float,
        entropy_direction: str,
    ) -> list[RecallResult]:
        """Adjust relevance scores using belief entropy from search results.

        SiliconDB ``SearchResult`` populates ``.entropy`` (Shannon entropy of
        the belief probability).  During ``recall()`` construction we stash
        that value on ``RecallResult.entropy`` when available.  This method
        blends it into the existing ``relevance_score``.

        * ``prefer_low``  → high-confidence (low-entropy) results float up.
        * ``prefer_high`` → uncertain (high-entropy) results float up.
        """
        if not results:
            return results

        for r in results:
            ent = getattr(r, "entropy", None) or 0.0
            # Normalise entropy roughly into [0, 1] — Shannon entropy of
            # a Bernoulli with p=0.5 is ~0.693, so dividing by ln(2) ≈ 0.693
            # maps max uncertainty to ≈1.0.
            ent_norm = min(ent / 0.693, 1.0)
            if entropy_direction == "prefer_high":
                adjustment = ent_norm * entropy_weight
            else:
                adjustment = (1.0 - ent_norm) * entropy_weight
            r.relevance_score = r.relevance_score * (1 - entropy_weight) + adjustment

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def _recency_score(self, occurred_at: datetime, as_of: datetime) -> float:
        """Compute recency score (1.0 = now, 0.0 = 1 week ago)."""
        age_hours = (as_of - occurred_at).total_seconds() / 3600
        return max(0.0, 1.0 - (age_hours / 168))

    def _triple_to_belief(self, t) -> Belief | None:
        """Convert SiliconDB TripleResult to Belief."""
        try:
            metadata = t.metadata or {}
            belief_id = UUID(metadata.get("belief_id", str(uuid4())))

            triplet = Triplet(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object_value,
            )

            source = None
            if t.sources:
                source_id = list(t.sources.keys())[0]
                source = Source(
                    id=source_id,
                    type=SourceType.OBSERVATION,
                    reliability=t.sources[source_id],
                )

            temporal = None
            if "observed_at" in metadata:
                temporal = TemporalContext(
                    observed_at=datetime.fromisoformat(metadata["observed_at"]),
                    valid_from=datetime.fromisoformat(metadata["valid_from"]) if metadata.get("valid_from") else None,
                    valid_until=datetime.fromisoformat(metadata["valid_until"]) if metadata.get("valid_until") else None,
                    last_verified=datetime.fromisoformat(metadata["last_verified"]) if metadata.get("last_verified") else None,
                )

            return Belief(
                id=belief_id,
                triplet=triplet,
                confidence=t.probability,
                source=source,
                status=BeliefStatus(metadata.get("status", "active")),
                tags=metadata.get("tags", []),
                temporal=temporal,
                evidence_for=[UUID(e) for e in metadata.get("evidence_for", [])],
                evidence_against=[UUID(e) for e in metadata.get("evidence_against", [])],
            )
        except Exception:
            return None

    def _doc_to_belief(self, doc: dict) -> Belief | None:
        """Convert SiliconDB document to Belief."""
        try:
            metadata = doc.get("metadata", {})
            return Belief(
                id=UUID(metadata.get("belief_id", str(uuid4()))),
                content=doc.get("text", ""),
                confidence=doc.get("probability", 1.0),
                status=BeliefStatus(metadata.get("status", "active")),
                tags=metadata.get("tags", []),
            )
        except Exception:
            return None

    def _search_result_to_belief(self, r) -> Belief | None:
        """Convert SiliconDB SearchResult to Belief."""
        try:
            metadata = self._rget(r, "metadata") or {}
            return Belief(
                id=UUID(metadata.get("belief_id", str(uuid4()))),
                content=self._rget(r, "text", ""),
                confidence=self._rget(r, "probability", 1.0),
                status=BeliefStatus(metadata.get("status", "active")),
                tags=metadata.get("tags", []),
            )
        except Exception:
            return None

    def _doc_to_experience(self, doc: dict) -> Experience | None:
        """Convert SiliconDB document to Experience."""
        try:
            metadata = doc.get("metadata", {})
            return Experience(
                id=UUID(metadata.get("experience_id", str(uuid4()))),
                content=doc.get("text", ""),
                occurred_at=datetime.fromisoformat(metadata["occurred_at"]) if metadata.get("occurred_at") else utc_now(),
                outcome=metadata.get("outcome"),
                processed=metadata.get("processed", False),
                session_id=metadata.get("session_id"),
                sequence_id=metadata.get("sequence_id"),
                causal_parent=UUID(metadata["causal_parent"]) if metadata.get("causal_parent") else None,
            )
        except Exception:
            return None

    def _search_result_to_experience(self, r) -> Experience | None:
        """Convert SiliconDB SearchResult to Experience."""
        try:
            metadata = self._rget(r, "metadata") or {}
            return Experience(
                id=UUID(metadata.get("experience_id", str(uuid4()))),
                content=self._rget(r, "text") or "",
                occurred_at=datetime.fromisoformat(metadata["occurred_at"]) if metadata.get("occurred_at") else utc_now(),
                outcome=metadata.get("outcome"),
                processed=metadata.get("processed", False),
                session_id=metadata.get("session_id"),
                sequence_id=metadata.get("sequence_id"),
            )
        except Exception:
            return None

    def _doc_to_procedure(self, doc: dict) -> Procedure | None:
        """Convert SiliconDB document to Procedure."""
        try:
            metadata = doc.get("metadata", {})

            source = None
            if metadata.get("source_id"):
                source = Source(
                    id=metadata["source_id"],
                    type=SourceType.OBSERVATION,
                    reliability=0.5,
                    metadata={"name": metadata.get("source_name", metadata["source_id"])},
                )

            return Procedure(
                id=UUID(metadata.get("procedure_id", str(uuid4()))),
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                steps=metadata.get("steps", []),
                trigger=metadata.get("trigger", ""),
                confidence=doc.get("probability", 0.5),
                success_count=metadata.get("success_count", 0),
                failure_count=metadata.get("failure_count", 0),
                source=source,
            )
        except Exception:
            return None

    def _search_result_to_procedure(self, r) -> Procedure | None:
        """Convert SiliconDB SearchResult to Procedure."""
        try:
            metadata = self._rget(r, "metadata") or {}

            source = None
            if metadata.get("source_id"):
                source = Source(
                    id=metadata["source_id"],
                    type=SourceType.OBSERVATION,
                    reliability=0.5,
                    metadata={"name": metadata.get("source_name", metadata["source_id"])},
                )

            return Procedure(
                id=UUID(metadata.get("procedure_id", str(uuid4()))),
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                steps=metadata.get("steps", []),
                trigger=metadata.get("trigger", ""),
                confidence=self._rget(r, "probability", 0.5),
                success_count=metadata.get("success_count", 0),
                failure_count=metadata.get("failure_count", 0),
                source=source,
            )
        except Exception:
            return None

    # ==================== Decision Record Methods ====================

    async def commit_decision(self, decision: Decision) -> None:
        """Store a decision record as a SiliconDB document."""
        import json

        ext_id = self._build_external_id("decision", decision.id)
        text = f"{decision.title}: {decision.description}"

        metadata = {
            **self._create_privacy_metadata(),
            "node_type": "decision",
            "decision_data": json.dumps(decision.to_dict()),
            "decision_id": str(decision.id),
            "status": decision.status.value,
            "decided_at": decision.decided_at.isoformat(),
            "decided_by": decision.decided_by,
            "session_id": decision.session_id,
            "belief_snapshot_id": decision.belief_snapshot_id,
        }

        self._db.ingest(
            external_id=ext_id,
            text=text,
            node_type="decision",
            metadata=metadata,
        )

        # Create graph edges for assumptions
        for assumption in decision.assumptions:
            belief_ext_id = str(assumption.belief_id)
            try:
                self._db.add_edge(ext_id, belief_ext_id, "assumes")
            except Exception:
                pass  # Belief may not exist as a vertex yet

    async def recall_decisions(
        self,
        query: str,
        k: int = 10,
        min_confidence: float = 0.0,
    ) -> list[Decision]:
        """Search decisions by semantic similarity."""
        import json

        results = self._db.search(
            query=query,
            k=k,
            filter={"node_type": "decision"},
        )

        decisions = []
        for r in results:
            metadata = r.metadata or {} if hasattr(r, "metadata") else r.get("metadata", {})
            decision_data = metadata.get("decision_data")
            if decision_data:
                try:
                    data = json.loads(decision_data)
                    decisions.append(Decision.from_dict(data))
                except Exception:
                    pass
        return decisions

    async def get_decision(self, decision_id: UUID) -> Decision | None:
        """Get a decision by UUID."""
        import json

        # Search by decision_id in metadata
        results = self._db.search(
            query=str(decision_id),
            k=5,
            filter={"node_type": "decision"},
        )

        for r in results:
            metadata = r.metadata or {} if hasattr(r, "metadata") else r.get("metadata", {})
            if metadata.get("decision_id") == str(decision_id):
                decision_data = metadata.get("decision_data")
                if decision_data:
                    try:
                        return Decision.from_dict(json.loads(decision_data))
                    except Exception:
                        pass
        return None

    async def record_decision_outcome(
        self,
        decision_id: UUID,
        outcome: str,
    ) -> bool:
        """Record the outcome of a decision."""
        import json

        decision = await self.get_decision(decision_id)
        if not decision:
            return False

        decision.outcome = outcome
        decision.outcome_recorded_at = utc_now()

        # Update the stored decision data
        results = self._db.search(
            query=str(decision_id),
            k=5,
            filter={"node_type": "decision"},
        )

        for r in results:
            metadata = r.metadata or {} if hasattr(r, "metadata") else r.get("metadata", {})
            if metadata.get("decision_id") == str(decision_id):
                ext_id = r.external_id if hasattr(r, "external_id") else r.get("external_id", "")
                metadata["decision_data"] = json.dumps(decision.to_dict())
                self._db.update(ext_id, metadata=metadata)
                return True
        return False

    async def revise_decision(
        self,
        decision_id: UUID,
        new_decision: Decision,
    ) -> Decision | None:
        """Create a revision of a decision, superseding the original."""
        import json

        original = await self.get_decision(decision_id)
        if not original:
            return None

        # Supersede the original
        original.status = DecisionStatus.SUPERSEDED
        results = self._db.search(
            query=str(decision_id),
            k=5,
            filter={"node_type": "decision"},
        )
        for r in results:
            metadata = r.metadata or {} if hasattr(r, "metadata") else r.get("metadata", {})
            if metadata.get("decision_id") == str(decision_id):
                ext_id = r.external_id if hasattr(r, "external_id") else r.get("external_id", "")
                metadata["decision_data"] = json.dumps(original.to_dict())
                metadata["status"] = DecisionStatus.SUPERSEDED.value
                self._db.update(ext_id, metadata=metadata)
                break

        # Create the new decision linked to original
        new_decision.revision_of = decision_id
        await self.commit_decision(new_decision)
        return new_decision

    async def snapshot_beliefs(self, belief_ids: list[str]) -> dict[str, Any]:
        """Create a snapshot of selected beliefs.

        Uses SiliconDB's snapshot_beliefs when available, otherwise builds
        a lightweight snapshot from current belief data.
        """
        if hasattr(self._db, "snapshot_beliefs"):
            return self._db.snapshot_beliefs(belief_ids)

        # Fallback: build snapshot manually from current beliefs
        snapshot_id = str(uuid4())
        snapshot_data = {}
        for bid in belief_ids:
            try:
                results = self._db.search(query=bid, k=3)
                for r in results:
                    ext_id = self._rget(r, "external_id", "")
                    if bid in ext_id:
                        snapshot_data[bid] = {
                            "confidence": self._rget(r, "probability", 0.5),
                            "text": self._rget(r, "text", ""),
                        }
                        break
            except Exception:
                pass
        return {"snapshot_id": snapshot_id, "beliefs": snapshot_data}

    # ========== Context Switch Snapshots ==========

    NODE_TYPE_SNAPSHOT = "snapshot"

    async def store_snapshot(self, snapshot: "ContextSnapshot") -> None:
        """Store a context snapshot as a SiliconDB document.

        The snapshot is stored with ``node_type="snapshot"`` so it can be
        retrieved via metadata-filtered search.
        """
        import json as _json
        from silicon_memory.snapshot.types import ContextSnapshot

        external_id = self._build_external_id("snapshot", snapshot.id)

        metadata = {
            **self._create_privacy_metadata(),
            "node_type": self.NODE_TYPE_SNAPSHOT,
            "task_context": snapshot.task_context,
            "created_at": snapshot.created_at.isoformat(),
            "session_id": snapshot.session_id,
            "snapshot_data": _json.dumps(snapshot.to_dict()),
        }

        text = (
            f"Context snapshot for {snapshot.task_context}. "
            f"{snapshot.summary}"
        )

        self._db.ingest(
            external_id=external_id,
            text=text,
            metadata=metadata,
            node_type=self.NODE_TYPE_SNAPSHOT,
        )

    async def query_snapshots_by_context(
        self,
        task_context: str | None = None,
        limit: int = 10,
    ) -> list["ContextSnapshot"]:
        """Retrieve context snapshots, optionally filtered by task context.

        Returns snapshots sorted by ``created_at`` descending (most recent first).
        """
        import json as _json
        from silicon_memory.snapshot.types import ContextSnapshot
        from datetime import datetime, timezone
        from uuid import UUID as _UUID

        user_prefix = self._get_user_prefix()

        filt: dict[str, Any] | None = {"node_type": self.NODE_TYPE_SNAPSHOT}
        if task_context:
            filt["task_context"] = task_context

        results = self._db.search(
            query=task_context or "",
            k=limit * 3,
            filter=filt,
        )

        snapshots: list[ContextSnapshot] = []
        for r in results:
            ext_id = self._rget(r, "external_id", "")
            if not ext_id.startswith(user_prefix):
                continue

            meta = self._rget(r, "metadata") or {}
            raw = meta.get("snapshot_data")
            if not raw:
                continue

            try:
                data = _json.loads(raw) if isinstance(raw, str) else raw
                # Explicit task_context filter (SiliconDB metadata filters
                # may not work perfectly with compound conditions)
                if task_context and data.get("task_context") != task_context:
                    continue
                exp_ids = [
                    _UUID(eid) for eid in data.get("recent_experiences", [])
                ]
                snap = ContextSnapshot(
                    id=_UUID(data["id"]),
                    task_context=data.get("task_context", ""),
                    summary=data.get("summary", ""),
                    working_memory=data.get("working_memory", {}),
                    recent_experiences=exp_ids,
                    next_steps=data.get("next_steps", []),
                    open_questions=data.get("open_questions", []),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    session_id=data.get("session_id"),
                    user_id=meta.get("owner_id"),
                    tenant_id=meta.get("tenant_id"),
                )
                snapshots.append(snap)
            except (KeyError, ValueError):
                continue

        # Sort by created_at descending
        snapshots.sort(key=lambda s: s.created_at, reverse=True)
        return snapshots[:limit]
