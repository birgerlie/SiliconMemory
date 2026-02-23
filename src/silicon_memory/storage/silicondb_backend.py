"""SiliconDB backend — backwards-compatible facade over decomposed stores.

This module preserves the original SiliconDBBackend API surface so that
existing callers (router, reflection engine, security services, tests)
continue to work without changes during the incremental migration.

New code should import from the individual stores directly:
    from silicon_memory.storage.engine import StorageLayer
    from silicon_memory.storage.beliefs import BeliefStore
    etc.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from silicon_memory.core.decision import Decision
from silicon_memory.core.types import (
    Belief,
    Experience,
    Procedure,
)
from silicon_memory.security.authorization import Permission
from silicon_memory.security.types import UserContext
from silicon_memory.storage._converters import (
    doc_to_belief as _doc_to_belief_fn,
)
from silicon_memory.storage._converters import (
    doc_to_experience as _doc_to_experience_fn,
)
from silicon_memory.storage._converters import (
    doc_to_procedure as _doc_to_procedure_fn,
)
from silicon_memory.storage._converters import (
    recency_score as _recency_score_fn,
)
from silicon_memory.storage._converters import (
    search_result_to_belief as _search_result_to_belief_fn,
)
from silicon_memory.storage._converters import (
    search_result_to_experience as _search_result_to_experience_fn,
)
from silicon_memory.storage._converters import (
    search_result_to_procedure as _search_result_to_procedure_fn,
)
from silicon_memory.storage._converters import (
    triple_to_belief as _triple_to_belief_fn,
)
from silicon_memory.storage._helpers import (
    apply_entropy_reranking as _apply_entropy_reranking_fn,
)
from silicon_memory.storage._helpers import (
    as_bool as _as_bool_fn,
)
from silicon_memory.storage._helpers import (
    canonical_claim_key_from_metadata,
)
from silicon_memory.storage._helpers import (
    looks_reflection_generated as _looks_reflection_generated_fn,
)
from silicon_memory.storage._helpers import (
    norm_text as _norm_text_fn,
)
from silicon_memory.storage._helpers import (
    normalize_dict as _normalize_dict_fn,
)
from silicon_memory.storage._helpers import (
    normalize_tags as _normalize_tags_fn,
)
from silicon_memory.storage._helpers import (
    normalize_triplet as _normalize_triplet_fn,
)
from silicon_memory.storage._helpers import (
    normalize_uuid_list as _normalize_uuid_list_fn,
)
from silicon_memory.storage._helpers import (
    parse_source_type as _parse_source_type_fn,
)

# Re-export helpers that callers may use directly
from silicon_memory.storage._helpers import (
    rget as _rget_fn,
)
from silicon_memory.storage._helpers import (
    token_jaccard as _token_jaccard_fn,
)
from silicon_memory.storage._helpers import (
    triplet_key as _triplet_key_fn,
)
from silicon_memory.storage._helpers import (
    triplet_similarity as _triplet_similarity_fn,
)
from silicon_memory.storage.beliefs import BeliefStore
from silicon_memory.storage.config import (
    NODE_TYPE_BELIEF,
    NODE_TYPE_BELIEF_SURFACE,
    NODE_TYPE_DREAM_RUN,
    NODE_TYPE_EXPERIENCE,
    NODE_TYPE_EXTRACTION_ITEM,
    NODE_TYPE_PROCEDURE,
    NODE_TYPE_REFLECTION_RUN,
    NODE_TYPE_SNAPSHOT,
    NODE_TYPE_WORKING,
    NODE_TYPE_WORKING_INDEX,
    SiliconDBConfig,
)
from silicon_memory.storage.decisions import DecisionStore
from silicon_memory.storage.engine import StorageLayer
from silicon_memory.storage.experiences import ExperienceStore
from silicon_memory.storage.knowledge import KnowledgeQuery
from silicon_memory.storage.observability import ObservabilityStore
from silicon_memory.storage.procedures import ProcedureStore
from silicon_memory.storage.raptor import RaptorStore
from silicon_memory.storage.reflection_tracking import ReflectionTracker
from silicon_memory.storage.snapshots import SnapshotStore
from silicon_memory.storage.working import WorkingStore
from silicon_memory.temporal.decay import DecayConfig

logger = logging.getLogger(__name__)

# Re-export SiliconDBConfig for backwards compat
__all__ = ["SiliconDBConfig", "SiliconDBBackend"]


class SiliconDBBackend:
    """Backwards-compatible facade delegating to decomposed stores.

    All public methods delegate to the appropriate domain store.
    The _db attribute is still accessible for callers that need raw
    SiliconDBClient access (security services, reflection engine).
    """

    # Node type constants — preserved for callers that reference them
    NODE_TYPE_BELIEF = NODE_TYPE_BELIEF
    NODE_TYPE_BELIEF_SURFACE = NODE_TYPE_BELIEF_SURFACE
    NODE_TYPE_EXPERIENCE = NODE_TYPE_EXPERIENCE
    NODE_TYPE_PROCEDURE = NODE_TYPE_PROCEDURE
    NODE_TYPE_WORKING = NODE_TYPE_WORKING
    NODE_TYPE_WORKING_INDEX = NODE_TYPE_WORKING_INDEX
    NODE_TYPE_EXTRACTION_ITEM = NODE_TYPE_EXTRACTION_ITEM
    NODE_TYPE_REFLECTION_RUN = NODE_TYPE_REFLECTION_RUN
    NODE_TYPE_DREAM_RUN = NODE_TYPE_DREAM_RUN
    NODE_TYPE_SNAPSHOT = NODE_TYPE_SNAPSHOT

    def __init__(
        self,
        config: SiliconDBConfig,
        user_context: UserContext,
        decay_config: DecayConfig | None = None,
    ) -> None:
        if not user_context:
            raise ValueError("user_context is required")

        # Core storage layer
        self._storage = StorageLayer(config, user_context, decay_config)
        self._config = config
        self._user_context = user_context

        # Expose _db for backwards-compat callers
        self._db = self._storage._db

        # Domain stores
        self._belief_store = BeliefStore(self._storage)
        self._experience_store = ExperienceStore(self._storage)
        self._procedure_store = ProcedureStore(self._storage)
        self._working_store = WorkingStore(self._storage)
        self._decision_store = DecisionStore(self._storage)
        self._raptor_store = RaptorStore(self._storage)
        self._observability_store = ObservabilityStore(self._storage)
        self._snapshot_store = SnapshotStore(self._storage)

        # Cross-domain stores
        self._reflection_tracker = ReflectionTracker(
            self._storage, self._belief_store
        )
        self._knowledge_query = KnowledgeQuery(
            self._storage,
            self._belief_store,
            self._experience_store,
            self._procedure_store,
            self._working_store,
        )

    # ---- Backwards-compat property shims for callers accessing internals ----

    @property
    def _working_keys(self) -> set[str]:
        return self._storage._working_keys

    @_working_keys.setter
    def _working_keys(self, value: set[str]) -> None:
        self._storage._working_keys = value

    @property
    def _experience_external_ids(self) -> set[str]:
        return self._storage._experience_external_ids

    @property
    def _extraction_external_ids(self) -> set[str]:
        return self._storage._extraction_external_ids

    @property
    def _mutation_semaphore(self):
        return self._storage._mutation_semaphore

    @property
    def _decay_config(self):
        return self._storage._decay_config

    @property
    def _policy_engine(self):
        return self._storage._policy_engine

    # ---- Lifecycle ----

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> SiliconDBBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---- ID + Access Control (delegated to StorageLayer) ----

    def _build_external_id(self, entity_type: str, entity_id: UUID | str) -> str:
        return self._storage.build_external_id(entity_type, entity_id)

    def _get_user_prefix(self) -> str:
        return self._storage.get_user_prefix()

    def _can_access(
        self,
        metadata: dict[str, Any],
        permission: Permission = Permission.READ,
        _external_id: str = "",
    ) -> bool:
        return self._storage.can_access(metadata, permission, _external_id)

    def _create_privacy_metadata(self, **kwargs) -> dict[str, Any]:
        return self._storage.create_privacy_metadata(**kwargs)

    # ---- Retry / Idempotency (delegated to StorageLayer) ----

    async def _run_db(self, op_name, fn, *, timeout_s=None, retry=True):
        return await self._storage.run_db(
            op_name, fn, timeout_s=timeout_s, retry=retry
        )

    def _make_idempotency_key(self, op_name, payload):
        return self._storage.make_idempotency_key(op_name, payload)

    def _is_duplicate_write(self, idem_key):
        return self._storage.is_duplicate_write(idem_key)

    def _mark_write_seen(self, idem_key):
        return self._storage.mark_write_seen(idem_key)

    def _ensure_reliability_state(self) -> None:
        """No-op — StorageLayer initializes everything in __init__."""
        pass

    # ---- Static/class helpers (delegated to module functions) ----

    @staticmethod
    def _rget(r, attr, default=None):
        return _rget_fn(r, attr, default)

    @staticmethod
    def _normalize_tags(raw):
        return _normalize_tags_fn(raw)

    @staticmethod
    def _normalize_dict(raw):
        return _normalize_dict_fn(raw)

    @staticmethod
    def _normalize_triplet(raw):
        return _normalize_triplet_fn(raw)

    @staticmethod
    def _triplet_key(subject, predicate, object_value):
        return _triplet_key_fn(subject, predicate, object_value)

    @staticmethod
    def _norm_text(value):
        return _norm_text_fn(value)

    @classmethod
    def _token_jaccard(cls, text_a, text_b):
        return _token_jaccard_fn(text_a, text_b)

    @classmethod
    def _triplet_similarity(cls, left, right):
        return _triplet_similarity_fn(left, right)

    @staticmethod
    def _as_bool(value, default=False):
        return _as_bool_fn(value, default)

    @staticmethod
    def _parse_source_type(value):
        return _parse_source_type_fn(value)

    @staticmethod
    def _normalize_uuid_list(raw):
        return _normalize_uuid_list_fn(raw)

    @staticmethod
    def _looks_reflection_generated(belief):
        return _looks_reflection_generated_fn(
            belief.tags,
            str(belief.source.id if belief.source else ""),
            str(
                belief.source.type.value
                if belief.source and hasattr(belief.source.type, "value")
                else ""
            ),
        )

    @staticmethod
    def _canonical_claim_key_from_belief(belief):
        metadata = belief.metadata if isinstance(belief.metadata, dict) else {}
        return canonical_claim_key_from_metadata(metadata, str(belief.id))

    @staticmethod
    def _apply_entropy_reranking(results, entropy_weight, entropy_direction="prefer_low"):
        return _apply_entropy_reranking_fn(results, entropy_weight, entropy_direction)

    # ---- Converters (delegated to module functions) ----

    def _triple_to_belief(self, t):
        return _triple_to_belief_fn(t)

    def _doc_to_belief(self, doc):
        return _doc_to_belief_fn(doc)

    def _search_result_to_belief(self, r):
        return _search_result_to_belief_fn(r)

    def _doc_to_experience(self, doc):
        return _doc_to_experience_fn(doc)

    def _search_result_to_experience(self, r):
        return _search_result_to_experience_fn(r)

    def _doc_to_procedure(self, doc):
        return _doc_to_procedure_fn(doc)

    def _search_result_to_procedure(self, r):
        return _search_result_to_procedure_fn(r)

    def _recency_score(self, occurred_at, as_of):
        return _recency_score_fn(occurred_at, as_of)

    # ---- Search helpers (delegated to StorageLayer) ----

    def _query_triples(self, **kwargs):
        return self._storage.query_triples_sync(**kwargs)

    def _weighted_search(self, query, k, weights_dict=None, **extra):
        return self._storage.weighted_search(query, k, weights_dict, **extra)

    def _search_by_type(self, query, node_types, target, **kwargs):
        return self._storage.search_by_type(query, node_types, target, **kwargs)

    # ---- Belief helpers (delegated to BeliefStore) ----

    def _build_evidence_refs(self, belief: Belief) -> list[dict[str, Any]]:
        return self._belief_store._build_evidence_refs(belief)

    async def _upsert_belief_surface(self, *args: Any, **kwargs: Any) -> None:
        return await self._belief_store._upsert_belief_surface(*args, **kwargs)

    async def _write_extraction_item_for_belief(self, *args: Any, **kwargs: Any) -> None:
        return await self._belief_store._write_extraction_item_for_belief(*args, **kwargs)

    # ========== Belief Operations (delegated to BeliefStore) ==========

    async def commit_belief(self, belief: Belief) -> None:
        await self._belief_store.commit_belief(belief)

    async def get_belief(self, belief_id: UUID) -> Belief | None:
        return await self._belief_store.get_belief(belief_id)

    async def query_beliefs(self, query, limit=10, min_confidence=0.0,
                            include_contested=False, search_weights=None):
        return await self._belief_store.query_beliefs(
            query, limit, min_confidence, include_contested, search_weights
        )

    async def get_beliefs_by_entity(self, entity: str):
        return await self._belief_store.get_beliefs_by_entity(entity)

    async def get_beliefs_by_tag(self, tag, limit=100, min_confidence=0.0):
        return await self._belief_store.get_beliefs_by_tag(tag, limit, min_confidence)

    async def find_contradictions(self, belief: Belief):
        return await self._belief_store.find_contradictions(belief)

    async def update_belief_confidence(self, belief_id: UUID, delta: float):
        return await self._belief_store.update_belief_confidence(belief_id, delta)

    async def update_belief_status(self, belief_id, new_status, reason=""):
        return await self._belief_store.update_belief_status(
            belief_id, new_status, reason
        )

    async def mark_beliefs_reflection_processed(self, external_ids):
        return await self._belief_store.mark_beliefs_reflection_processed(external_ids)

    async def find_claim_merge_candidates(self, belief, limit=8):
        return await self._belief_store.find_claim_merge_candidates(belief, limit)

    async def get_evidence_for_belief(self, belief_id: UUID):
        return await self._belief_store.get_evidence_for_belief(belief_id)

    async def get_beliefs_from_experience(self, experience_id: UUID):
        return await self._reflection_tracker.get_beliefs_from_experience(experience_id)

    async def _query_beliefs_with_entropy(self, query, limit=10, min_confidence=0.0,
                                          include_contested=False, search_weights=None):
        return await self._belief_store.query_beliefs_with_entropy(
            query, limit, min_confidence, include_contested, search_weights
        )

    # ========== Experience Operations (delegated to ExperienceStore) ==========

    async def record_experience(self, experience: Experience) -> None:
        await self._experience_store.record_experience(experience)

    async def get_experience(self, experience_id: UUID):
        return await self._experience_store.get_experience(experience_id)

    async def wait_for_experience_visibility(self, experience_ids, **kwargs):
        return await self._experience_store.wait_for_experience_visibility(
            experience_ids, **kwargs
        )

    async def query_experiences(self, query, limit=10, search_weights=None):
        return await self._experience_store.query_experiences(
            query, limit, search_weights
        )

    async def get_recent_experiences(self, hours=24, limit=100):
        return await self._experience_store.get_recent_experiences(hours, limit)

    async def get_unprocessed_experiences(self, limit=100):
        return await self._experience_store.get_unprocessed_experiences(limit)

    async def mark_experiences_processed(self, experience_ids):
        return await self._experience_store.mark_experiences_processed(experience_ids)

    async def get_unextracted_experiences(self, limit=10000):
        return await self._experience_store.get_unextracted_experiences(limit)

    async def mark_experiences_extracted(self, experience_ids):
        return await self._experience_store.mark_experiences_extracted(experience_ids)

    async def count_extraction_progress(self):
        return await self._experience_store.count_extraction_progress()

    # Expose internal search helpers for engine.py callers
    def _search_experiences_broad(self, target, search_weights=None):
        return self._experience_store._search_experiences_broad(target, search_weights)

    async def _search_experiences_broad_async(self, target, search_weights=None):
        return await self._experience_store._search_experiences_broad_async(
            target, search_weights
        )

    async def _get_experience_doc(self, external_id):
        return await self._experience_store._get_experience_doc(external_id)

    async def _hydrate_experience_from_external_id(self, external_id):
        return await self._experience_store._hydrate_experience_from_external_id(
            external_id
        )

    # ========== Procedure Operations (delegated to ProcedureStore) ==========

    async def commit_procedure(self, procedure: Procedure) -> None:
        await self._procedure_store.commit_procedure(procedure)

    async def get_procedure(self, procedure_id: UUID):
        return await self._procedure_store.get_procedure(procedure_id)

    async def find_applicable_procedures(self, context, limit=5, **kwargs):
        return await self._procedure_store.find_applicable_procedures(
            context, limit, **kwargs
        )

    async def record_procedure_outcome(self, procedure_id: UUID, success: bool):
        return await self._procedure_store.record_procedure_outcome(
            procedure_id, success
        )

    # ========== Working Memory (delegated to WorkingStore) ==========

    async def set_working(self, key, value, ttl_seconds=300):
        return await self._working_store.set_working(key, value, ttl_seconds)

    async def get_working(self, key):
        return await self._working_store.get_working(key)

    async def delete_working(self, key):
        return await self._working_store.delete_working(key)

    async def get_all_working(self):
        return await self._working_store.get_all_working()

    async def cleanup_expired_working(self):
        return await self._working_store.cleanup_expired_working()

    # ========== Knowledge Proofs + Recall (delegated to KnowledgeQuery) ==========

    async def build_knowledge_proof(self, query, min_confidence=0.3):
        return await self._knowledge_query.build_knowledge_proof(
            query, min_confidence
        )

    async def recall(self, query, max_facts=20, max_experiences=10,
                     max_procedures=5, min_confidence=0.3, include_working=True,
                     search_weights=None):
        return await self._knowledge_query.recall(
            query, max_facts, max_experiences, max_procedures,
            min_confidence, include_working, search_weights,
        )

    # ========== RAPTOR (delegated to RaptorStore) ==========

    async def build_raptor_tree(self, cluster_size=10, max_levels=5):
        return await self._raptor_store.build_raptor_tree(cluster_size, max_levels)

    async def search_raptor_hybrid(self, query, k=10, tree_boost=0.3):
        return await self._raptor_store.search_raptor_hybrid(query, k, tree_boost)

    def get_raptor_stats(self):
        return self._raptor_store._s.get_raptor_stats_sync()

    # ========== Observability (delegated to ObservabilityStore) ==========

    async def get_event_stats(self):
        return await self._observability_store.get_event_stats()

    async def replay_mutations(self, from_time=None, limit=100):
        return await self._observability_store.replay_mutations(from_time, limit)

    async def detect_triple_contradictions(self, min_probability=0.3):
        return await self._observability_store.detect_triple_contradictions(
            min_probability
        )

    async def get_uncertain_beliefs(self, min_entropy=0.5, k=50):
        return await self._observability_store.get_uncertain_beliefs(min_entropy, k)

    # ========== Reflection Tracking (delegated to ReflectionTracker) ==========

    async def get_unprocessed_extraction_items(self, limit=1000):
        return await self._reflection_tracker.get_unprocessed_extraction_items(limit)

    async def mark_extraction_items_processed(self, extraction_external_ids, run_id=""):
        return await self._reflection_tracker.mark_extraction_items_processed(
            extraction_external_ids, run_id
        )

    async def record_reflection_run(self, run_id, status, metrics):
        return await self._reflection_tracker.record_reflection_run(
            run_id, status, metrics
        )

    async def record_dream_run(self, run_id, status, metrics):
        return await self._reflection_tracker.record_dream_run(
            run_id, status, metrics
        )

    # ========== Decisions (delegated to DecisionStore) ==========

    async def commit_decision(self, decision: Decision) -> None:
        await self._decision_store.commit_decision(decision)

    async def recall_decisions(self, query, k=10, min_confidence=0.0):
        return await self._decision_store.recall_decisions(query, k, min_confidence)

    async def get_decision(self, decision_id: UUID):
        return await self._decision_store.get_decision(decision_id)

    async def record_decision_outcome(self, decision_id, outcome, notes=""):
        return await self._decision_store.record_decision_outcome(
            decision_id, outcome
        )

    async def update_decision_status(self, decision_id, status, reason=None):
        return await self._decision_store.update_decision_status(
            decision_id, status, reason
        )

    async def revise_decision(self, decision_id, new_decision):
        return await self._decision_store.revise_decision(decision_id, new_decision)

    # ========== Snapshots (delegated to SnapshotStore) ==========

    async def snapshot_beliefs(self, belief_ids):
        return await self._snapshot_store.snapshot_beliefs(belief_ids)

    async def store_snapshot(self, snapshot):
        return await self._snapshot_store.store_snapshot(snapshot)

    async def query_snapshots_by_context(self, task_context=None, limit=10):
        return await self._snapshot_store.query_snapshots_by_context(
            task_context, limit
        )
