"""Memory router using SiliconDB backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend

from silicon_memory.core.decision import Decision
from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Experience,
    KnowledgeProof,
    Procedure,
    RecallResult,
    SourceType,
)
from silicon_memory.core.utils import utc_now
from silicon_memory.ingestion.types import IngestionAdapter, IngestionResult
from silicon_memory.retrieval.salience import PROFILES, SalienceProfile
from silicon_memory.security.audit import AuditAction, AuditLogger
from silicon_memory.security.config import SecurityConfig
from silicon_memory.security.forgetting import (
    ForgetResult,
    ForgettingService,
)
from silicon_memory.security.inspector import (
    CorrectionResult,
    MemoryInspection,
    MemoryInspector,
    MemoryRecord,
)
from silicon_memory.security.preferences import MemoryPreferences
from silicon_memory.security.transparency import (
    AccessLogEntry,
    ProvenanceChain,
    TransparencyService,
)
from silicon_memory.security.types import (
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)
from silicon_memory.snapshot.service import SnapshotService
from silicon_memory.snapshot.types import ContextSnapshot, SnapshotConfig
from silicon_memory.storage import (
    BeliefStore,
    DecisionStore,
    ExperienceStore,
    KnowledgeQuery,
    ObservabilityStore,
    ProcedureStore,
    RaptorStore,
    ReflectionTracker,
    SnapshotStore,
    StorageLayer,
    WorkingStore,
)
from silicon_memory.storage.config import SiliconDBConfig
from silicon_memory.temporal.decay import DecayConfig


@dataclass
class CrossReferenceResult:
    """Result of cross-referencing internal and external beliefs."""

    query: str
    internal_beliefs: list[Any] = field(default_factory=list)
    external_beliefs: list[Any] = field(default_factory=list)
    agreements: list[tuple[Any, Any]] = field(default_factory=list)
    contradictions: list[tuple[Any, Any]] = field(default_factory=list)


@dataclass
class RecallContext:
    """Context for memory recall."""

    query: str
    as_of: datetime | None = None
    budget_tokens: int = 4000
    min_confidence: float = 0.3
    include_episodic: bool = True
    include_procedural: bool = True
    include_working: bool = True
    max_facts: int = 20
    max_experiences: int = 10
    max_procedures: int = 5
    source_type: str | None = None
    salience_profile: SalienceProfile | str | None = None
    graph_context_nodes: list[str] | None = None


@dataclass
class RecallResponse:
    """Response from memory recall."""

    facts: list[RecallResult]
    experiences: list[RecallResult]
    procedures: list[RecallResult]
    working_context: dict[str, Any]
    total_items: int
    query: str
    as_of: datetime


@dataclass
class IngestBatchReceipt:
    """Batch ingest receipt for experience writes."""

    batch_id: str
    submitted_at: datetime
    accepted_count: int
    failed_count: int = 0
    accepted_experience_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestBatchResult:
    """Final result for a batch ingest operation."""

    receipt: IngestBatchReceipt
    state: str
    visibility_ok: bool | None = None


class SiliconMemory:
    """Unified memory interface backed by SiliconDB.

    This is the main entry point for the silicon-memory system.
    All storage is handled by SiliconDB with its mmap + WAL architecture.

    Multi-user security:
    - All operations require UserContext
    - Automatic tenant/user isolation
    - Privacy controls (private, workspace, public)
    - Forgetting, transparency, and inspection APIs

    Example:
        >>> from silicon_memory import SiliconMemory
        >>> from silicon_memory.security import UserContext, PrivacyLevel
        >>>
        >>> user_ctx = UserContext(
        ...     user_id="user-123",
        ...     tenant_id="acme-corp",
        ...     default_privacy=PrivacyLevel.PRIVATE,
        ... )
        >>>
        >>> with SiliconMemory("/path/to/db", user_context=user_ctx) as memory:
        ...     # Store a belief
        ...     await memory.commit_belief(belief)
        ...
        ...     # Recall relevant memories
        ...     response = await memory.recall(RecallContext(query="Python"))
        ...
        ...     # Ask "what do you know"
        ...     proof = await memory.what_do_you_know("Python")
        ...
        ...     # Forget a session
        ...     await memory.forget_session("session-abc")
        ...
        ...     # Why do you know?
        ...     chains = await memory.why_do_you_know("Python")
    """

    def __init__(
        self,
        path: str | Path,
        user_context: UserContext,
        db_grpc_host: str = "127.0.0.1",
        db_grpc_port: int = 8643,
        db_retry_attempts: int = 3,
        db_retry_base_ms: int = 100,
        db_retry_max_ms: int = 2000,
        db_request_timeout_s: float = 10.0,
        db_max_inflight_mutations: int = 32,
        db_idempotency_ttl_s: int = 600,
        language: str = "english",
        enable_graph: bool = True,
        auto_embedder: bool = True,
        embedder_model: str = "base",
        decay_config: DecayConfig | None = None,
        security_config: SecurityConfig | None = None,
        snapshot_config: SnapshotConfig | None = None,
        llm_provider: Any = None,
    ) -> None:
        """Initialize SiliconMemory.

        Args:
            path: Path to the database
            user_context: Required user context for all operations
            db_grpc_host: SiliconDB gRPC host when transport is "grpc"
            db_grpc_port: SiliconDB gRPC port when transport is "grpc"
            db_retry_attempts: Max retries for transient DB failures
            db_retry_base_ms: Base backoff delay in milliseconds
            db_retry_max_ms: Max backoff delay in milliseconds
            db_request_timeout_s: Per-call timeout budget in seconds
            db_max_inflight_mutations: Backpressure limit for concurrent writes
            db_idempotency_ttl_s: TTL for write idempotency keys
            language: Language for text processing
            enable_graph: Enable graph relationships
            auto_embedder: Enable automatic embedding
            embedder_model: Embedding model to use
            decay_config: Optional decay configuration
            security_config: Optional security configuration
            snapshot_config: Optional snapshot configuration
            llm_provider: Optional LLM provider for snapshot summaries
        """
        if not user_context:
            raise ValueError("user_context is required")

        self._user_context = user_context
        self._security_config = security_config or SecurityConfig()

        config = SiliconDBConfig(
            path=path,
            grpc_host=db_grpc_host,
            grpc_port=db_grpc_port,
            retry_attempts=db_retry_attempts,
            retry_base_ms=db_retry_base_ms,
            retry_max_ms=db_retry_max_ms,
            request_timeout_s=db_request_timeout_s,
            max_inflight_mutations=db_max_inflight_mutations,
            idempotency_ttl_s=db_idempotency_ttl_s,
            language=language,
            enable_graph=enable_graph,
            auto_embedder=auto_embedder,
            embedder_model=embedder_model,
        )

        # Core storage layer + domain stores (bypass SiliconDBBackend facade)
        self._storage = StorageLayer(config, user_context, decay_config)
        self._beliefs = BeliefStore(self._storage)
        self._experiences = ExperienceStore(self._storage)
        self._procedures = ProcedureStore(self._storage)
        self._working = WorkingStore(self._storage)
        self._decisions = DecisionStore(self._storage)
        self._raptor = RaptorStore(self._storage)
        self._observability = ObservabilityStore(self._storage)
        self._snapshots = SnapshotStore(self._storage)
        self._reflection_tracker = ReflectionTracker(
            self._storage, self._beliefs
        )
        self._knowledge = KnowledgeQuery(
            self._storage,
            self._beliefs,
            self._experiences,
            self._procedures,
            self._working,
        )

        # Initialize security services
        self._forgetting_service = ForgettingService(self._storage)
        self._inspector = MemoryInspector(self._storage)
        self._audit_logger = AuditLogger(
            self._storage,
            retention_days=self._security_config.audit_retention_days,
            log_reads=self._security_config.audit_read_operations,
            max_entries=self._security_config.max_audit_entries,
        )
        self._transparency_service = TransparencyService(
            self._storage,
            max_access_log_entries=self._security_config.max_access_log_entries,
        )

        # Initialize snapshot service
        self._snapshot_service = SnapshotService(
            memory=self,
            snapshots=self._snapshots,
            config=snapshot_config,
            llm_provider=llm_provider,
        )

        # User preferences (loaded on demand)
        self._preferences: MemoryPreferences | None = None

    @property
    def user_context(self) -> UserContext:
        """Get the current user context."""
        return self._user_context

    @property
    def _backend(self) -> SiliconDBBackend:
        """Backwards-compat: creates facade on first access."""
        if not hasattr(self, "_SiliconMemory__backend_facade"):
            from silicon_memory.storage.silicondb_backend import SiliconDBBackend as _Facade

            # Create facade sharing our StorageLayer
            facade = object.__new__(_Facade)
            facade._storage = self._storage
            facade._config = self._storage.config
            facade._user_context = self._storage.user_context
            facade._db = self._storage._db
            # Wire up stores
            facade._belief_store = self._beliefs
            facade._experience_store = self._experiences
            facade._procedure_store = self._procedures
            facade._working_store = self._working
            facade._decision_store = self._decisions
            facade._raptor_store = self._raptor
            facade._observability_store = self._observability
            facade._snapshot_store = self._snapshots
            facade._reflection_tracker = self._reflection_tracker
            facade._knowledge_query = self._knowledge
            self.__backend_facade = facade
        return self.__backend_facade

    def close(self) -> None:
        """Close the memory system."""
        self._storage.close()

    def __enter__(self) -> SiliconMemory:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> SiliconMemory:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ========== Unified Recall ==========

    async def recall(self, ctx: RecallContext) -> RecallResponse:
        """Recall relevant memories across all layers.

        If a salience_profile is specified (by name or object), the
        profile's weights are passed to the backend for weighted retrieval.

        Args:
            ctx: Recall context with query and parameters

        Returns:
            RecallResponse with memories from all layers
        """
        # Resolve salience profile
        search_weights = None
        if ctx.salience_profile is not None:
            profile = ctx.salience_profile
            if isinstance(profile, str):
                profile = PROFILES.get(profile)
            if isinstance(profile, SalienceProfile):
                search_weights = profile.to_search_weights()

        # Resolve graph context seeds for PPR when graph_proximity is active
        if search_weights and search_weights.get("graph_proximity", 0) > 0:
            seeds = ctx.graph_context_nodes
            if not seeds:
                seeds = await self._resolve_context_seeds(ctx.query)
            if seeds:
                search_weights["graph_context_nodes"] = seeds

        recall_kwargs: dict[str, Any] = {
            "query": ctx.query,
            "max_facts": ctx.max_facts,
            "max_experiences": ctx.max_experiences if ctx.include_episodic else 0,
            "max_procedures": ctx.max_procedures if ctx.include_procedural else 0,
            "min_confidence": ctx.min_confidence,
            "include_working": ctx.include_working,
        }
        if search_weights is not None:
            recall_kwargs["search_weights"] = search_weights

        result = await self._knowledge.recall(**recall_kwargs)

        if ctx.source_type:
            mode = ctx.source_type.lower()

            def _include(item: RecallResult) -> bool:
                src_type = item.source.type if item.source else SourceType.OBSERVATION
                is_external = src_type == SourceType.EXTERNAL
                if mode == "external":
                    return is_external
                if mode == "internal":
                    return not is_external
                return True

            result["facts"] = [x for x in result["facts"] if _include(x)]
            result["experiences"] = [x for x in result["experiences"] if _include(x)]
            result["procedures"] = [x for x in result["procedures"] if _include(x)]
            result["total_items"] = (
                len(result["facts"]) + len(result["experiences"]) + len(result["procedures"])
            )

        return RecallResponse(
            facts=result["facts"],
            experiences=result["experiences"],
            procedures=result["procedures"],
            working_context=result["working_context"],
            total_items=result["total_items"],
            query=result["query"],
            as_of=result["as_of"],
        )

    async def _resolve_context_seeds(self, query: str) -> list[str]:
        """Auto-resolve PPR seed nodes from working memory and query.

        Looks up the ``current_topic`` key in working memory and combines it
        with a lightweight search against beliefs to produce a small set of
        external IDs that can serve as PPR seeds.
        """
        seeds: list[str] = []

        # Use current_topic from working memory if available
        current_topic = await self._working.get_working("current_topic")
        search_query = f"{query} {current_topic}" if current_topic else query

        # Find a handful of matching belief nodes to use as seeds
        beliefs = await self._beliefs.query_beliefs(search_query, limit=5)
        for b in beliefs:
            ext_id = self._storage.build_external_id("belief", b.id)
            seeds.append(ext_id)

        return seeds

    async def build_knowledge_tree(
        self,
        cluster_size: int = 10,
        max_levels: int = 5,
    ) -> dict:
        """Build a RAPTOR hierarchical tree over the document store."""
        return await self._raptor.build_raptor_tree(cluster_size, max_levels)

    async def search_raptor(
        self,
        query: str,
        k: int = 10,
        tree_boost: float = 0.3,
    ) -> list[RecallResult]:
        """Search using RAPTOR hierarchical retrieval."""
        return await self._raptor.search_raptor_hybrid(query, k, tree_boost)

    async def what_do_you_know(
        self,
        query: str,
        min_confidence: float = 0.3,
    ) -> KnowledgeProof:
        """Answer 'what do you know about X' with proof.

        Args:
            query: The topic to query
            min_confidence: Minimum confidence threshold

        Returns:
            KnowledgeProof with beliefs, sources, and contradictions
        """
        return await self._knowledge.build_knowledge_proof(query, min_confidence)

    # ========== Semantic Memory (Beliefs) ==========

    async def commit_belief(self, belief: Belief) -> None:
        """Commit a belief to semantic memory."""
        await self._beliefs.commit_belief(belief)

    async def get_belief(self, belief_id: UUID) -> Belief | None:
        """Get a belief by ID."""
        return await self._beliefs.get_belief(belief_id)

    async def query_beliefs(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[Belief]:
        """Query beliefs by semantic similarity."""
        return await self._beliefs.query_beliefs(query, limit, min_confidence)

    async def find_contradictions(self, belief: Belief) -> list[Belief]:
        """Find beliefs that contradict the given belief."""
        return await self._beliefs.find_contradictions(belief)

    async def get_beliefs_from_experience(self, experience_id: UUID) -> list[Belief]:
        """Get beliefs extracted from a specific experience."""
        return await self._reflection_tracker.get_beliefs_from_experience(experience_id)

    async def update_belief_status(
        self,
        belief_id: UUID,
        new_status: BeliefStatus,
        reason: str = "",
    ) -> bool:
        """Update a belief lifecycle status."""
        return await self._beliefs.update_belief_status(belief_id, new_status, reason)

    # ========== Episodic Memory (Experiences) ==========

    async def record_experience(self, experience: Experience) -> None:
        """Record an experience to episodic memory."""
        await self._experiences.record_experience(experience)

    async def get_experience(self, experience_id: UUID) -> Experience | None:
        """Get an experience by ID."""
        return await self._experiences.get_experience(experience_id)

    async def get_recent_experiences(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[Experience]:
        """Get recent experiences."""
        return await self._experiences.get_recent_experiences(hours, limit)

    async def mark_experiences_processed(self, experience_ids: list[UUID]) -> None:
        """Mark experiences as processed by reflection."""
        await self._experiences.mark_experiences_processed(experience_ids)

    async def get_unextracted_experiences(self, limit: int = 10000) -> list[Experience]:
        """Get experiences not yet processed by extraction."""
        return await self._experiences.get_unextracted_experiences(limit)

    async def mark_experiences_extracted(self, experience_ids: list[UUID]) -> None:
        """Mark experiences as extracted."""
        await self._experiences.mark_experiences_extracted(experience_ids)

    async def count_extraction_progress(self) -> dict[str, int]:
        """Count extracted vs unextracted experiences."""
        return await self._experiences.count_extraction_progress()

    async def wait_for_ingest_visibility(
        self,
        experience_ids: list[str],
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.1,
    ) -> bool:
        """Wait until ingested experiences become visible in search."""
        parsed: list[UUID] = []
        for raw in experience_ids:
            try:
                parsed.append(UUID(str(raw)))
            except Exception:
                continue
        if not parsed:
            return True
        return await self._experiences.wait_for_experience_visibility(
            parsed,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )

    async def ingest_experiences_batch(
        self,
        experiences: list[Experience],
        *,
        wait_for_visibility: bool = True,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.1,
    ) -> IngestBatchResult:
        """Record a batch of experiences and optionally wait for visibility.

        State values:
        - SUCCEEDED: all accepted and (if requested) visibility reached
        - PARTIAL: some writes failed but at least one accepted
        - FAILED: no writes accepted
        - TIMEOUT: accepted writes did not reach visibility within timeout
        """
        batch_id = str(uuid4())
        accepted_ids: list[str] = []
        errors: list[str] = []

        for experience in experiences:
            try:
                await self.record_experience(experience)
                accepted_ids.append(str(experience.id))
            except Exception as exc:  # noqa: PERF203
                errors.append(f"{experience.id}: {exc}")

        receipt = IngestBatchReceipt(
            batch_id=batch_id,
            submitted_at=utc_now(),
            accepted_count=len(accepted_ids),
            failed_count=len(errors),
            accepted_experience_ids=accepted_ids,
            errors=errors,
        )

        if not accepted_ids:
            return IngestBatchResult(
                receipt=receipt,
                state="FAILED",
                visibility_ok=False if wait_for_visibility else None,
            )

        visibility_ok: bool | None = None
        if wait_for_visibility:
            visibility_ok = await self.wait_for_ingest_visibility(
                accepted_ids,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )

        if wait_for_visibility and not visibility_ok:
            state = "TIMEOUT"
        elif errors:
            state = "PARTIAL"
        else:
            state = "SUCCEEDED"
        return IngestBatchResult(
            receipt=receipt,
            state=state,
            visibility_ok=visibility_ok,
        )

    # ========== Procedural Memory (Procedures) ==========

    async def commit_procedure(self, procedure: Procedure) -> None:
        """Commit a procedure to procedural memory."""
        await self._procedures.commit_procedure(procedure)

    async def get_procedure(self, procedure_id: UUID) -> Procedure | None:
        """Get a procedure by ID."""
        return await self._procedures.get_procedure(procedure_id)

    async def find_applicable_procedures(
        self,
        context: str,
        limit: int = 5,
    ) -> list[Procedure]:
        """Find procedures applicable to the context."""
        return await self._procedures.find_applicable_procedures(context, limit)

    async def record_procedure_outcome(
        self,
        procedure_id: UUID,
        success: bool,
    ) -> bool:
        """Record an outcome for a procedure."""
        return await self._procedures.record_procedure_outcome(procedure_id, success)

    # ========== Working Memory ==========

    async def set_context(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set a value in working memory."""
        await self._working.set_working(key, value, ttl_seconds)

    async def get_context(self, key: str) -> Any | None:
        """Get a value from working memory."""
        return await self._working.get_working(key)

    async def delete_context(self, key: str) -> bool:
        """Delete a value from working memory."""
        return await self._working.delete_working(key)

    async def get_all_context(self) -> dict[str, Any]:
        """Get all working memory context."""
        return await self._working.get_all_working()

    async def cleanup_expired(self) -> int:
        """Clean up expired working memory entries."""
        return await self._working.cleanup_expired_working()

    # ========== Decision Records ==========

    async def commit_decision(self, decision: Decision) -> str | None:
        """Store a decision record with a belief snapshot.

        Creates a snapshot of the assumptions' current belief states,
        then stores the decision as a document.

        Args:
            decision: The Decision to commit

        Returns:
            The belief snapshot ID, or None if no assumptions
        """
        # Create belief snapshot from assumptions
        snapshot_id = None
        if decision.assumptions:
            belief_ids = [str(a.belief_id) for a in decision.assumptions]
            snapshot = await self._snapshots.snapshot_beliefs(belief_ids)
            snapshot_id = snapshot.get("snapshot_id")
            decision.belief_snapshot_id = snapshot_id

        await self._decisions.commit_decision(decision)
        return snapshot_id

    async def recall_decisions(
        self,
        query: str,
        k: int = 10,
        min_confidence: float = 0.0,
    ) -> list[Decision]:
        """Search decisions by semantic similarity.

        Args:
            query: Search query
            k: Number of results
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching decisions
        """
        return await self._decisions.recall_decisions(query, k, min_confidence)

    async def get_decision(self, decision_id: UUID) -> Decision | None:
        """Get a decision by ID with current vs original assumption confidences.

        Args:
            decision_id: The decision UUID

        Returns:
            Decision or None
        """
        decision = await self._decisions.get_decision(decision_id)
        if not decision:
            return None

        drift_items: list[dict[str, Any]] = []
        for assumption in decision.assumptions:
            current_confidence: float | None = None
            try:
                belief = await self._beliefs.get_belief(assumption.belief_id)
                if belief is not None:
                    current_confidence = belief.confidence
            except Exception:
                current_confidence = None

            delta = (
                (current_confidence - assumption.confidence_at_decision)
                if current_confidence is not None
                else None
            )
            abs_delta = abs(delta) if delta is not None else None
            exceeds = bool(
                assumption.is_critical
                and abs_delta is not None
                and abs_delta > 0.2
            )
            drift_items.append(
                {
                    "belief_id": str(assumption.belief_id),
                    "description": assumption.description,
                    "is_critical": assumption.is_critical,
                    "confidence_at_decision": assumption.confidence_at_decision,
                    "current_confidence": current_confidence,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "drift_threshold_exceeded": exceeds,
                }
            )

        decision.metadata["assumption_drift"] = drift_items
        decision.metadata["needs_revisit"] = any(
            item["drift_threshold_exceeded"] for item in drift_items
        )
        return decision

    async def record_outcome(
        self,
        decision_id: UUID,
        outcome: str,
    ) -> bool:
        """Record the outcome of a decision.

        Args:
            decision_id: The decision UUID
            outcome: Description of the outcome

        Returns:
            True if successful
        """
        return await self._decisions.record_decision_outcome(decision_id, outcome)

    async def revise_decision(
        self,
        decision_id: UUID,
        new_decision: Decision,
    ) -> Decision | None:
        """Create a linked revision of a decision, superseding the original.

        Args:
            decision_id: The original decision UUID
            new_decision: The new revised decision

        Returns:
            The new Decision, or None if original not found
        """
        return await self._decisions.revise_decision(decision_id, new_decision)

    async def generate_decision_brief(
        self,
        question: str,
        llm_provider: Any | None = None,
    ) -> dict[str, Any]:
        """Generate a structured decision brief for a question."""
        from silicon_memory.decision.synthesis import DecisionBriefGenerator

        generator = DecisionBriefGenerator(self, llm_provider=llm_provider)
        brief = await generator.generate(question)
        return brief.to_dict()

    # ========== Context Snapshots ==========

    async def create_snapshot(
        self,
        task_context: str,
        llm_provider: Any = None,
    ) -> ContextSnapshot:
        """Create a context snapshot for task switching.

        Captures working memory, recent experiences, and generates a summary
        (via LLM or rule-based fallback).

        Args:
            task_context: Identifier for the task being snapshotted
            llm_provider: Optional LLM provider override

        Returns:
            The created ContextSnapshot
        """
        return await self._snapshot_service.create_snapshot(
            task_context, llm_provider
        )

    async def get_latest_snapshot(
        self,
        task_context: str,
    ) -> ContextSnapshot | None:
        """Get the most recent snapshot for a task context.

        Args:
            task_context: The task context identifier

        Returns:
            The latest ContextSnapshot or None
        """
        return await self._snapshot_service.get_latest_snapshot(task_context)

    async def list_snapshots(
        self,
        task_context: str | None = None,
        limit: int = 10,
    ) -> list[ContextSnapshot]:
        """List snapshots, optionally filtered by task context.

        Args:
            task_context: Optional filter
            limit: Maximum results

        Returns:
            List of ContextSnapshot objects sorted by created_at desc
        """
        return await self._snapshot_service.list_snapshots(task_context, limit)

    # ========== Cross-Reference API ==========

    async def cross_reference(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.3,
    ) -> CrossReferenceResult:
        """Cross-reference internal and external beliefs on a topic.

        Retrieves beliefs from internal and external sources separately,
        then identifies agreements and contradictions.

        Args:
            query: The topic to cross-reference
            limit: Maximum beliefs per source type
            min_confidence: Minimum confidence threshold

        Returns:
            CrossReferenceResult with categorized beliefs
        """
        from silicon_memory.core.types import SourceType

        all_beliefs = await self._beliefs.query_beliefs(
            query, limit=limit * 2, min_confidence=min_confidence
        )

        internal = []
        external = []
        for b in all_beliefs:
            if b.source and b.source.type == SourceType.EXTERNAL:
                external.append(b)
            else:
                internal.append(b)

        internal = internal[:limit]
        external = external[:limit]

        # Find agreements and contradictions
        agreements: list[tuple[Any, Any]] = []
        contradictions_list: list[tuple[Any, Any]] = []

        for ib in internal:
            for eb in external:
                ib_text = ib.content or (ib.triplet.as_text() if ib.triplet else "")
                eb_text = eb.content or (eb.triplet.as_text() if eb.triplet else "")
                if not ib_text or not eb_text:
                    continue
                # Simple heuristic: if both have the same topic/content direction
                # Check if they share significant word overlap
                ib_words = set(ib_text.lower().split())
                eb_words = set(eb_text.lower().split())
                overlap = len(ib_words & eb_words)
                total = min(len(ib_words), len(eb_words))
                if total > 0 and overlap / total > 0.3:
                    # Check for contradiction markers
                    contra_words = {"not", "no", "never", "incorrect", "wrong", "false"}
                    ib_has_neg = bool(ib_words & contra_words)
                    eb_has_neg = bool(eb_words & contra_words)
                    if ib_has_neg != eb_has_neg:
                        contradictions_list.append((ib, eb))
                    else:
                        agreements.append((ib, eb))

        return CrossReferenceResult(
            query=query,
            internal_beliefs=internal,
            external_beliefs=external,
            agreements=agreements,
            contradictions=contradictions_list,
        )

    # ========== Ingestion API ==========

    async def ingest_from(
        self,
        adapter: IngestionAdapter,
        content: str | bytes,
        metadata: dict[str, Any],
        llm_provider: Any = None,
    ) -> IngestionResult:
        """Ingest content through an adapter with user context injection.

        Args:
            adapter: The ingestion adapter to use
            content: Raw content to ingest
            metadata: Source metadata
            llm_provider: Optional LLM provider for enhanced extraction

        Returns:
            IngestionResult with statistics
        """
        # Inject user context into metadata
        metadata = {
            **metadata,
            "user_id": self._user_context.user_id,
            "tenant_id": self._user_context.tenant_id,
        }

        try:
            return await adapter.ingest(
                content=content,
                metadata=metadata,
                memory=self,
                llm_provider=llm_provider,
            )
        except Exception as e:
            return IngestionResult(
                source_type=adapter.source_type,
                errors=[f"Ingestion failed: {e}"],
            )

    # ========== Forgetting API ==========

    async def forget_entity(
        self,
        entity_id: str,
        entity_type: str | None = None,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget a single entity by ID.

        Args:
            entity_id: The entity ID to delete
            entity_type: Optional entity type (belief, experience, procedure)
            reason: Optional reason for deletion

        Returns:
            ForgetResult with operation status
        """
        result = await self._forgetting_service.forget_entity(
            self._user_context, entity_id, entity_type, reason
        )
        if self._security_config.audit_delete_operations:
            await self._audit_logger.log_forget(
                self._user_context,
                scope="entity",
                deleted_count=result.deleted_count,
                details={"entity_id": entity_id, "entity_type": entity_type},
            )
        return result

    async def forget_session(
        self,
        session_id: str,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all data from a session.

        Args:
            session_id: The session ID
            reason: Optional reason for deletion

        Returns:
            ForgetResult with operation status
        """
        result = await self._forgetting_service.forget_session(
            self._user_context, session_id, reason
        )
        if self._security_config.audit_delete_operations:
            await self._audit_logger.log_forget(
                self._user_context,
                scope="session",
                deleted_count=result.deleted_count,
                details={"session_id": session_id},
            )
        return result

    async def forget_before(
        self,
        timestamp: datetime,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all data before a timestamp (GDPR).

        Args:
            timestamp: Delete everything before this time
            reason: Optional reason for deletion

        Returns:
            ForgetResult with operation status
        """
        result = await self._forgetting_service.forget_before(
            self._user_context, timestamp, reason
        )
        if self._security_config.audit_delete_operations:
            await self._audit_logger.log_forget(
                self._user_context,
                scope="time_range",
                deleted_count=result.deleted_count,
                details={"before_timestamp": timestamp.isoformat()},
            )
        return result

    async def selective_forget(
        self,
        query: str,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget data matching a semantic query.

        Args:
            query: Semantic search query
            reason: Optional reason for deletion

        Returns:
            ForgetResult with operation status
        """
        result = await self._forgetting_service.selective_forget(
            self._user_context, query, reason
        )
        if self._security_config.audit_delete_operations:
            await self._audit_logger.log_forget(
                self._user_context,
                scope="query",
                deleted_count=result.deleted_count,
                details={"query": query},
            )
        return result

    async def forget_topics(
        self,
        topics: list[str],
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all data with specified topics.

        Args:
            topics: List of topics to delete
            reason: Optional reason for deletion

        Returns:
            ForgetResult with operation status
        """
        result = await self._forgetting_service.forget_topics(
            self._user_context, topics, reason
        )
        if self._security_config.audit_delete_operations:
            await self._audit_logger.log_forget(
                self._user_context,
                scope="topic",
                deleted_count=result.deleted_count,
                details={"topics": topics},
            )
        return result

    async def forget_all(
        self,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all user data (GDPR erasure request).

        Args:
            reason: Optional reason for deletion

        Returns:
            ForgetResult with operation status
        """
        result = await self._forgetting_service.forget_all(
            self._user_context, reason
        )
        if self._security_config.audit_delete_operations:
            await self._audit_logger.log_forget(
                self._user_context,
                scope="all",
                deleted_count=result.deleted_count,
                details={"gdpr_erasure": True},
            )
        return result

    # ========== Transparency API ==========

    async def why_do_you_know(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ProvenanceChain]:
        """Answer "why do you know about X?".

        Returns provenance chains explaining where knowledge came from.

        Args:
            query: The topic/query to explain
            limit: Maximum number of chains to return

        Returns:
            List of ProvenanceChain objects
        """
        return await self._transparency_service.why_do_you_know(
            self._user_context, query, limit
        )

    async def get_provenance(
        self,
        entity_id: str,
    ) -> ProvenanceChain | None:
        """Get the provenance chain for a specific entity.

        Args:
            entity_id: The entity ID

        Returns:
            ProvenanceChain or None if not found
        """
        return await self._transparency_service.get_provenance(
            self._user_context, entity_id
        )

    async def get_access_log(
        self,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[AccessLogEntry]:
        """Get the access log for a user or specific entity.

        Args:
            entity_id: Optional entity ID to filter by
            limit: Maximum entries to return

        Returns:
            List of AccessLogEntry objects
        """
        return await self._transparency_service.get_access_log(
            self._user_context, entity_id, limit
        )

    # ========== Inspector API ==========

    async def inspect_memories(self) -> MemoryInspection:
        """Get an overview of the user's memory contents.

        Returns:
            MemoryInspection with statistics
        """
        return await self._inspector.inspect_memories(self._user_context)

    async def export_memories(
        self,
        entity_types: list[str] | None = None,
    ) -> AsyncIterator[MemoryRecord]:
        """Export user's memories as an async iterator.

        Args:
            entity_types: Optional filter by entity types

        Yields:
            MemoryRecord objects
        """
        if self._security_config.audit_enabled:
            inspection = await self._inspector.inspect_memories(self._user_context)
            await self._audit_logger.log_export(
                self._user_context,
                record_count=inspection.total_items,
            )

        async for record in self._inspector.export_memories(
            self._user_context, entity_types
        ):
            yield record

    async def import_memories(
        self,
        records: list[MemoryRecord],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Import memories from records.

        Args:
            records: List of MemoryRecord objects to import
            overwrite: Whether to overwrite existing records

        Returns:
            Summary of import operation
        """
        result = await self._inspector.import_memories(
            self._user_context, records, overwrite
        )
        if self._security_config.audit_enabled:
            await self._audit_logger.log_import(
                self._user_context,
                imported_count=result["imported"],
                failed_count=result["failed"],
            )
        return result

    async def correct_memory(
        self,
        entity_id: str,
        corrections: dict[str, Any],
    ) -> CorrectionResult:
        """Correct a specific memory.

        Args:
            entity_id: The entity to correct
            corrections: Dictionary of fields to correct

        Returns:
            CorrectionResult
        """
        result = await self._inspector.correct_memory(
            self._user_context, entity_id, corrections
        )
        if self._security_config.audit_write_operations and result.success:
            await self._audit_logger.log(
                self._user_context,
                AuditAction.UPDATE,
                resource_id=entity_id,
                resource_type="memory",
                details={"corrections": list(corrections.keys())},
            )
        return result

    async def get_memory(
        self,
        entity_id: str,
    ) -> MemoryRecord | None:
        """Get a specific memory record.

        Args:
            entity_id: The entity ID

        Returns:
            MemoryRecord or None
        """
        return await self._inspector.get_memory(self._user_context, entity_id)

    # ========== Preferences API ==========

    async def get_preferences(self) -> MemoryPreferences:
        """Get user preferences."""
        if self._preferences is None:
            self._preferences = MemoryPreferences.default_for_user(
                self._user_context.user_id,
                self._user_context.tenant_id,
            )
        return self._preferences

    async def update_preferences(self, preferences: MemoryPreferences) -> None:
        """Update user preferences.

        Args:
            preferences: The new preferences
        """
        self._preferences = preferences

    async def update_preferences_partial(self, updates: dict[str, Any]) -> bool:
        """Partially update user preferences.

        Args:
            updates: Dictionary of fields to update

        Returns:
            True if successful
        """
        prefs = await self.get_preferences()
        for key, value in updates.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        self._preferences = prefs
        return True

    async def add_blocked_topic(self, topic: str) -> bool:
        """Add a topic to the do-not-remember list."""
        prefs = await self.get_preferences()
        prefs.add_blocked_topic(topic)
        return True

    async def remove_blocked_topic(self, topic: str) -> bool:
        """Remove a topic from the do-not-remember list."""
        prefs = await self.get_preferences()
        return prefs.remove_blocked_topic(topic)

    async def add_blocked_pattern(self, pattern: str) -> bool:
        """Add a regex pattern to the do-not-remember list."""
        prefs = await self.get_preferences()
        return prefs.add_blocked_pattern(pattern)

    async def remove_blocked_pattern(self, pattern: str) -> bool:
        """Remove a pattern from the do-not-remember list."""
        prefs = await self.get_preferences()
        return prefs.remove_blocked_pattern(pattern)

    # ========== Privacy API ==========

    async def set_entity_privacy(
        self,
        entity_id: str,
        privacy_level: PrivacyLevel,
    ) -> bool:
        """Set privacy level for an entity.

        Args:
            entity_id: The entity ID
            privacy_level: The new privacy level

        Returns:
            True if successful
        """
        try:
            # Build full external ID if needed
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            # Verify ownership
            if metadata.get("owner_id") != self._user_context.user_id:
                if not self._user_context.is_admin():
                    return False

            metadata["privacy_level"] = privacy_level.value
            self._storage._db.update(entity_id, metadata=metadata)
            return True
        except Exception:
            return False

    async def get_entity_privacy(
        self,
        entity_id: str,
    ) -> PrivacyMetadata | None:
        """Get privacy metadata for an entity.

        Args:
            entity_id: The entity ID

        Returns:
            PrivacyMetadata or None
        """
        try:
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return None

            metadata = doc.get("metadata", {})
            if metadata.get("owner_id") and metadata.get("tenant_id"):
                return PrivacyMetadata.from_dict(metadata)
            return None
        except Exception:
            return None

    async def share_entity(
        self,
        entity_id: str,
        user_id: str,
    ) -> bool:
        """Share an entity with another user.

        Args:
            entity_id: The entity ID
            user_id: The user to share with

        Returns:
            True if successful
        """
        try:
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            # Verify ownership
            if metadata.get("owner_id") != self._user_context.user_id:
                if not self._user_context.is_admin():
                    return False

            shared_with = metadata.get("shared_with", [])
            if user_id not in shared_with:
                shared_with.append(user_id)
                metadata["shared_with"] = shared_with
                self._storage._db.update(entity_id, metadata=metadata)

            return True
        except Exception:
            return False

    async def unshare_entity(
        self,
        entity_id: str,
        user_id: str,
    ) -> bool:
        """Remove sharing from an entity.

        Args:
            entity_id: The entity ID
            user_id: The user to unshare from

        Returns:
            True if successful
        """
        try:
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            # Verify ownership
            if metadata.get("owner_id") != self._user_context.user_id:
                if not self._user_context.is_admin():
                    return False

            shared_with = metadata.get("shared_with", [])
            if user_id in shared_with:
                shared_with.remove(user_id)
                metadata["shared_with"] = shared_with
                self._storage._db.update(entity_id, metadata=metadata)

            return True
        except Exception:
            return False

    # ========== Consent API ==========

    async def grant_consent(
        self,
        entity_id: str,
        consent_type: str,
    ) -> bool:
        """Grant consent for an entity.

        Args:
            entity_id: The entity ID
            consent_type: Type of consent (storage, processing, sharing)

        Returns:
            True if successful
        """
        try:
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            consents = metadata.get("consents", {})
            consents[consent_type] = utc_now().isoformat()
            metadata["consents"] = consents
            self._storage._db.update(entity_id, metadata=metadata)
            return True
        except Exception:
            return False

    async def revoke_consent(
        self,
        entity_id: str,
        consent_type: str,
    ) -> bool:
        """Revoke consent for an entity.

        Args:
            entity_id: The entity ID
            consent_type: Type of consent to revoke

        Returns:
            True if successful
        """
        try:
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            consents = metadata.get("consents", {})
            if consent_type in consents:
                del consents[consent_type]
                metadata["consents"] = consents
                self._storage._db.update(entity_id, metadata=metadata)

            return True
        except Exception:
            return False

    async def check_consent(
        self,
        entity_id: str,
        consent_type: str,
    ) -> bool:
        """Check if consent is granted.

        Args:
            entity_id: The entity ID
            consent_type: Type of consent to check

        Returns:
            True if consent is granted
        """
        try:
            if "/" not in entity_id:
                entity_id = f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_id}"

            doc = self._storage._db.get(entity_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            consents = metadata.get("consents", {})
            return consent_type in consents
        except Exception:
            return False
