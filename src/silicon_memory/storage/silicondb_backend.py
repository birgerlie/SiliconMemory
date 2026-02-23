"""SiliconDB backend - all memory operations backed by SiliconDB."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import logging
import math
import random
import ast
from pathlib import Path
import threading
import time
from typing import Any, Callable
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now
from silicon_memory.core.claims import (
    AntiTrivialityPolicy,
    ClaimDecisionLabel,
    ClaimMergeCandidate,
    ClaimPrecisionJudge,
    build_provenance_contract,
    validate_provenance_contract,
)
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

logger = logging.getLogger(__name__)


@dataclass
class SiliconDBConfig:
    """Configuration for SiliconDB backend."""

    path: str | Path
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 8643
    language: str = "english"
    enable_graph: bool = True
    enable_async: bool = True
    auto_embedder: bool = True
    embedder_model: str = "base"
    retry_attempts: int = 3
    retry_base_ms: int = 100
    retry_max_ms: int = 2000
    request_timeout_s: float = 10.0
    max_inflight_mutations: int = 32
    idempotency_ttl_s: int = 600

    # --- Feature flags: leverage SiliconDB native capabilities ---
    use_evidence_links: bool = True           # Phase 1: evidence refs on triples
    use_native_entropy_rerank: bool = True    # Phase 2: delegate entropy reranking
    use_native_temporal_decay: bool = True    # Phase 2: delegate temporal decay
    skip_belief_surface_writes: bool = True   # Phase 2: stop writing belief_surface docs
    use_consistency_waits: bool = True        # Phase 3A: wait_for after each write
    use_event_stream: bool = True             # Phase 3B: event-driven architecture
    enable_raptor: bool = False               # Phase 4: RAPTOR hierarchical retrieval


class SiliconDBBackend:
    """Unified memory backend using SiliconDB.

    All memory operations (semantic, episodic, procedural, working) are
    stored in SiliconDB using different node_types and metadata.

    Memory types are distinguished by:
    - semantic: Triples with node_type="belief" plus belief_surface docs
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
    NODE_TYPE_BELIEF_SURFACE = "belief_surface"
    NODE_TYPE_EXPERIENCE = "experience"
    NODE_TYPE_PROCEDURE = "procedure"
    NODE_TYPE_WORKING = "working"
    NODE_TYPE_WORKING_INDEX = "working_index"
    NODE_TYPE_EXTRACTION_ITEM = "extraction_item"
    NODE_TYPE_REFLECTION_RUN = "reflection_run"
    NODE_TYPE_DREAM_RUN = "dream_run"

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
            from silicondb import SiliconDBClient
        except ImportError as e:
            raise ImportError(
                "SiliconDB is required. Install with: pip install silicondb"
            ) from e

        grpc_target = f"{config.grpc_host}:{config.grpc_port}"
        self._db = SiliconDBClient(grpc_target=grpc_target)
        self._config = config
        self._decay_config = decay_config or DecayConfig()
        self._user_context = user_context
        self._policy_engine = PolicyEngine()
        # Track working memory keys in-process (workaround for SiliconDB #125:
        # empty-query search with metadata filter returns no results)
        self._working_keys: set[str] = set()
        # Track recently written/read experience IDs so episodic retrieval can
        # fall back to direct gets when search indexing is temporarily stale.
        self._experience_external_ids: set[str] = set()
        # Track extraction journal item IDs for deterministic reflection fallback
        # when extraction-item search indexing is temporarily stale.
        self._extraction_external_ids: set[str] = set()
        self._mutation_semaphore = asyncio.Semaphore(max(1, config.max_inflight_mutations))
        self._idempotency_lock = threading.Lock()
        self._idempotency_seen: dict[str, float] = {}
        self._idempotency_last_gc = 0.0
        self._anti_triviality = AntiTrivialityPolicy()
        self._claim_precision_judge = ClaimPrecisionJudge()

    def _build_external_id(self, entity_type: str, entity_id: UUID | str) -> str:
        """Build the new external ID format: {tenant_id}/{user_id}/{type}-{uuid}."""
        return f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_type}-{entity_id}"

    def _get_user_prefix(self) -> str:
        """Get the prefix for the current user's documents."""
        return f"{self._user_context.tenant_id}/{self._user_context.user_id}/"

    def _can_access(
        self,
        metadata: dict[str, Any],
        permission: Permission = Permission.READ,
        _external_id: str = "",
    ) -> bool:
        """Check if current user can access a document based on its metadata."""
        # Extract privacy metadata
        owner_id = metadata.get("owner_id")
        tenant_id = metadata.get("tenant_id")
        privacy_level = metadata.get("privacy_level", "private")
        shared_with = metadata.get("shared_with", [])

        # Some triple query paths may not hydrate ownership metadata.
        # Fall back to tenant/user prefix guard when external_id is available.
        if not owner_id and not tenant_id:
            if _external_id:
                expected = f"{self._user_context.tenant_id}/{self._user_context.user_id}/"
                return _external_id.startswith(expected)
            return True

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

    @staticmethod
    def _normalize_tags(raw: Any) -> list[str]:
        """Normalize tags from list/set/string encodings."""
        if isinstance(raw, list | set | tuple):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list | set | tuple):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            return [s]
        return []

    @staticmethod
    def _normalize_dict(raw: Any) -> dict[str, Any]:
        """Normalize dict payloads that may be JSON-encoded."""
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    @staticmethod
    def _normalize_triplet(raw: Any) -> Triplet | None:
        """Normalize triplet payloads from dict/JSON."""
        parsed = SiliconDBBackend._normalize_dict(raw)
        subject = str(parsed.get("subject") or "").strip()
        predicate = str(parsed.get("predicate") or "").strip()
        object_value = str(parsed.get("object") or "").strip()
        if not (subject and predicate and object_value):
            return None
        return Triplet(subject=subject, predicate=predicate, object=object_value)

    @staticmethod
    def _triplet_key(subject: str, predicate: str, object_value: str) -> str:
        return (
            f"{subject.strip().lower()}|"
            f"{predicate.strip().lower()}|"
            f"{object_value.strip().lower()}"
        )

    @staticmethod
    def _norm_text(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @classmethod
    def _token_jaccard(cls, text_a: str, text_b: str) -> float:
        tokens_a = set(cls._norm_text(text_a).split())
        tokens_b = set(cls._norm_text(text_b).split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    @classmethod
    def _triplet_similarity(cls, left: Triplet, right: Triplet) -> float:
        left_sub = cls._norm_text(left.subject)
        left_pred = cls._norm_text(left.predicate)
        left_obj = cls._norm_text(left.object)
        right_sub = cls._norm_text(right.subject)
        right_pred = cls._norm_text(right.predicate)
        right_obj = cls._norm_text(right.object)

        subject_match = 1.0 if left_sub == right_sub else cls._token_jaccard(left_sub, right_sub)
        predicate_match = 1.0 if left_pred == right_pred else cls._token_jaccard(left_pred, right_pred)
        object_match = 1.0 if left_obj == right_obj else cls._token_jaccard(left_obj, right_obj)

        return (0.4 * subject_match) + (0.3 * predicate_match) + (0.3 * object_match)

    @staticmethod
    def _looks_reflection_generated(belief: Belief) -> bool:
        tags = {str(tag).lower() for tag in (belief.tags or set())}
        if tags & {"extracted", "consolidated", "hypothesis", "run_batch"}:
            return True
        source_id = str(belief.source.id if belief.source else "").strip().lower()
        if source_id in {
            "reflection_engine",
            "observation_consolidation",
            "batch_extract_reflect",
            "hypothesis_generation",
        }:
            return True
        if belief.source and hasattr(belief.source.type, "value"):
            return str(belief.source.type.value).strip().lower() == "reflection"
        return False

    @staticmethod
    def _canonical_claim_key_from_belief(belief: Belief) -> str:
        metadata = belief.metadata if isinstance(belief.metadata, dict) else {}
        canonical_id = str(metadata.get("canonical_claim_id") or "").strip()
        return canonical_id or str(belief.id)

    @staticmethod
    def _parse_source_type(value: Any) -> SourceType:
        """Parse source type enum from string/native value."""
        if isinstance(value, SourceType):
            return value
        if isinstance(value, str):
            norm = value.strip().lower()
            for item in SourceType:
                if item.value == norm:
                    return item
        return SourceType.OBSERVATION

    @staticmethod
    def _normalize_uuid_list(raw: Any) -> list[UUID]:
        """Normalize UUID lists from list/JSON/string payloads."""
        out: list[UUID] = []
        values: list[str] = []
        if isinstance(raw, list | tuple | set):
            values = [str(v) for v in raw]
        elif isinstance(raw, str):
            s = raw.strip()
            if not s:
                values = []
            else:
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        values = [str(v) for v in parsed]
                    else:
                        values = [s]
                except Exception:
                    values = [s]
        for value in values:
            try:
                out.append(UUID(value))
            except Exception:
                continue
        return out

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """Parse bools from native/string/int encodings."""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in {"true", "1", "yes", "y", "on"}:
                return True
            if norm in {"false", "0", "no", "n", "off", ""}:
                return False
        return default

    def _query_triples(self, **kwargs):
        """Compatibility wrapper for query_triples across client variants."""
        # Some SiliconDB client builds do not accept `min_probability`.
        kwargs.pop("min_probability", None)
        return self._db.query_triples(**kwargs)

    def _ensure_reliability_state(self) -> None:
        """Initialize reliability state for test backends that bypass __init__."""
        if not hasattr(self, "_config"):
            self._config = SiliconDBConfig(path="")
        if not hasattr(self, "_working_keys"):
            self._working_keys = set()
        if not hasattr(self, "_experience_external_ids"):
            self._experience_external_ids = set()
        if not hasattr(self, "_extraction_external_ids"):
            self._extraction_external_ids = set()
        if not hasattr(self, "_mutation_semaphore"):
            self._mutation_semaphore = asyncio.Semaphore(max(1, self._config.max_inflight_mutations))
        if not hasattr(self, "_idempotency_lock"):
            self._idempotency_lock = threading.Lock()
        if not hasattr(self, "_idempotency_seen"):
            self._idempotency_seen = {}
        if not hasattr(self, "_idempotency_last_gc"):
            self._idempotency_last_gc = 0.0
        if not hasattr(self, "_anti_triviality"):
            self._anti_triviality = AntiTrivialityPolicy()
        if not hasattr(self, "_claim_precision_judge"):
            self._claim_precision_judge = ClaimPrecisionJudge()

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """Classify transient transport errors that should be retried."""
        if isinstance(exc, TimeoutError | asyncio.TimeoutError):
            return True
        msg = str(exc).lower()
        transient_markers = (
            "deadline",
            "timeout",
            "temporarily unavailable",
            "unavailable",
            "resource exhausted",
            "connection reset",
            "connection refused",
            "try again",
        )
        return any(marker in msg for marker in transient_markers)

    def _retry_delay_seconds(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        base = max(0.001, self._config.retry_base_ms / 1000.0)
        cap = max(base, self._config.retry_max_ms / 1000.0)
        backoff = min(cap, base * (2 ** attempt))
        return backoff * (0.5 + random.random())

    async def _run_db(
        self,
        op_name: str,
        fn: Callable[[], Any],
        *,
        timeout_s: float | None = None,
        retry: bool = True,
    ) -> Any:
        """Run a synchronous DB call in a worker thread with retry policy."""
        self._ensure_reliability_state()
        timeout = timeout_s if timeout_s is not None else self._config.request_timeout_s
        attempts = self._config.retry_attempts if retry else 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
                if attempt >= attempts - 1 or not self._is_retryable_exception(exc):
                    break
                delay = self._retry_delay_seconds(attempt)
                logger.warning(
                    "Retrying %s after %s (attempt %d/%d, delay=%.3fs)",
                    op_name,
                    exc,
                    attempt + 1,
                    attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _make_idempotency_key(self, op_name: str, payload: dict[str, Any]) -> str:
        """Create deterministic idempotency key for write operations."""
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{op_name}:{digest}"

    def _idempotency_gc_locked(self, now: float) -> None:
        if now - self._idempotency_last_gc < 30.0:
            return
        cutoff = now - float(self._config.idempotency_ttl_s)
        stale = [k for k, ts in self._idempotency_seen.items() if ts < cutoff]
        for k in stale:
            self._idempotency_seen.pop(k, None)
        self._idempotency_last_gc = now

    def _is_duplicate_write(self, idem_key: str) -> bool:
        """Return True when this idempotency key was seen recently."""
        self._ensure_reliability_state()
        now = time.monotonic()
        with self._idempotency_lock:
            self._idempotency_gc_locked(now)
            ts = self._idempotency_seen.get(idem_key)
            if ts is None:
                return False
            return (now - ts) <= float(self._config.idempotency_ttl_s)

    def _mark_write_seen(self, idem_key: str) -> None:
        """Record successful write idempotency key."""
        self._ensure_reliability_state()
        now = time.monotonic()
        with self._idempotency_lock:
            self._idempotency_gc_locked(now)
            self._idempotency_seen[idem_key] = now

    async def find_claim_merge_candidates(
        self,
        belief: Belief,
        limit: int = 8,
    ) -> list[ClaimMergeCandidate]:
        """Recall scored merge candidates for a new belief claim."""
        if limit <= 0:
            return []

        scored: dict[UUID, ClaimMergeCandidate] = {}

        if belief.triplet:
            try:
                triples = self._query_triples(
                    subject=belief.triplet.subject,
                    predicate=belief.triplet.predicate,
                    k=max(limit * 30, 200),
                )
            except Exception:
                triples = []
            for triple in triples:
                metadata = self._rget(triple, "metadata") or {}
                if not self._can_access(
                    metadata,
                    _external_id=self._rget(triple, "external_id", ""),
                ):
                    continue
                existing = self._triple_to_belief(triple)
                if not existing or not existing.triplet:
                    continue
                if existing.id == belief.id:
                    continue
                similarity = self._triplet_similarity(belief.triplet, existing.triplet)
                candidate = ClaimMergeCandidate(
                    belief_id=existing.id,
                    score=similarity,
                    reason="subject_predicate_triplet_scan",
                    source="triple_scan",
                    subject=existing.triplet.subject,
                    predicate=existing.triplet.predicate,
                    object_value=existing.triplet.object,
                    confidence=existing.confidence,
                )
                prior = scored.get(existing.id)
                if prior is None or candidate.score > prior.score:
                    scored[existing.id] = candidate

        query_text = (
            belief.content.strip()
            if belief.content and belief.content.strip()
            else belief.triplet.as_text() if belief.triplet else ""
        )
        if query_text:
            try:
                search_results = self._search_by_type(
                    query_text,
                    {self.NODE_TYPE_BELIEF, self.NODE_TYPE_BELIEF_SURFACE},
                    target=max(limit * 20, 100),
                )
            except Exception:
                search_results = []
            for result in search_results:
                metadata = self._rget(result, "metadata") or {}
                if not self._can_access(
                    metadata,
                    _external_id=self._rget(result, "external_id", ""),
                ):
                    continue
                existing = self._search_result_to_belief(result)
                if not existing:
                    continue
                if existing.id == belief.id:
                    continue

                existing_text = (
                    existing.content
                    if existing.content.strip()
                    else existing.triplet.as_text() if existing.triplet else ""
                )
                lexical = self._token_jaccard(query_text, existing_text)
                if belief.triplet and existing.triplet:
                    lexical = max(lexical, self._triplet_similarity(belief.triplet, existing.triplet))
                candidate = ClaimMergeCandidate(
                    belief_id=existing.id,
                    score=lexical,
                    reason="surface_search_recall",
                    source="surface_search",
                    subject=existing.triplet.subject if existing.triplet else "",
                    predicate=existing.triplet.predicate if existing.triplet else "",
                    object_value=existing.triplet.object if existing.triplet else "",
                    confidence=existing.confidence,
                )
                prior = scored.get(existing.id)
                if prior is None or candidate.score > prior.score:
                    scored[existing.id] = candidate

        ranked = sorted(scored.values(), key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    # ========== Semantic Memory (Beliefs/Triplets) ==========

    async def commit_belief(self, belief: Belief) -> None:
        """Store a belief as a SiliconDB triple."""
        self._ensure_reliability_state()
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
            "content": belief.content or (belief.triplet.as_text() if belief.triplet else ""),
            "created_at": utc_now().isoformat(),
            **privacy_meta,
        }

        belief_meta = dict(belief.metadata) if isinstance(belief.metadata, dict) else {}
        if belief_meta:
            for key, value in belief_meta.items():
                if key in {"privacy_level", "owner_id", "tenant_id", "shared_with"}:
                    continue
                metadata.setdefault(key, value)

        tags_value = (
            list(belief.tags) if isinstance(belief.tags, set)
            else list(belief.tags or []) if isinstance(belief.tags, list | tuple)
            else ([belief.tags] if belief.tags else [])
        )
        evidence_for_value = [str(e) for e in belief.evidence_for]
        evidence_against_value = [str(e) for e in belief.evidence_against]
        metadata["tags"] = tags_value
        metadata["evidence_for"] = evidence_for_value
        metadata["evidence_against"] = evidence_against_value

        source_metadata = {}
        if isinstance(metadata.get("source_metadata"), dict):
            source_metadata.update(metadata["source_metadata"])

        if belief.source:
            metadata["source_id"] = belief.source.id
            metadata["source_type"] = (
                belief.source.type.value
                if hasattr(belief.source.type, "value")
                else str(belief.source.type)
            )
            metadata["source_reliability"] = float(belief.source.reliability)
            if belief.source.metadata:
                source_metadata.update(belief.source.metadata)
        if source_metadata:
            metadata["source_metadata"] = source_metadata

        claim_relation = str(
            metadata.get("claim_relation_label")
            or metadata.get("claim_relation")
            or "",
        ).strip().upper()
        canonical_claim_id = str(
            metadata.get("canonical_claim_id")
            or metadata.get("canonical_belief_id")
            or "",
        ).strip()
        canonical_promotion_allowed = True
        claim_relation_unknown = False
        claim_relation_unknown_reason = ""
        valid_claim_labels = {item.value for item in ClaimDecisionLabel}

        if self._looks_reflection_generated(belief):
            provenance = build_provenance_contract(belief)
            provenance_validation = validate_provenance_contract(provenance)
            anti_triviality = self._anti_triviality.evaluate(belief)
            metadata["provenance"] = provenance.as_dict()
            metadata["provenance_complete"] = provenance_validation.complete
            metadata["provenance_missing_fields"] = provenance_validation.missing_fields
            metadata["provenance_completeness_rate"] = round(provenance_validation.completeness_rate, 4)
            metadata["anti_triviality_score"] = anti_triviality.score
            metadata["anti_triviality_signals"] = anti_triviality.signals
            metadata["anti_triviality_block_reasons"] = anti_triviality.block_reasons

            claim_candidates: list[ClaimMergeCandidate] = []
            if not claim_relation and belief.triplet:
                claim_candidates = await self.find_claim_merge_candidates(belief, limit=8)
                decision = await self._claim_precision_judge.decide(
                    belief,
                    claim_candidates,
                    llm=None,
                )
                claim_relation = decision.label.value
                metadata["claim_decision_reason"] = decision.reason
                metadata["claim_decision_confidence"] = decision.confidence
                metadata["claim_decision_via"] = decision.via
                metadata["claim_candidate_count"] = len(claim_candidates)
                if decision.candidate_belief_id and not canonical_claim_id:
                    canonical_claim_id = str(decision.candidate_belief_id)

            if claim_relation and claim_relation not in valid_claim_labels:
                claim_relation_unknown = True
                claim_relation_unknown_reason = f"invalid_input_label:{claim_relation}"
                claim_relation = ClaimDecisionLabel.NEW.value
            if not claim_relation:
                claim_relation_unknown = True
                claim_relation_unknown_reason = claim_relation_unknown_reason or "missing_label"
                claim_relation = ClaimDecisionLabel.NEW.value

            canonical_claim_id = canonical_claim_id or str(belief.id)
            metadata["claim_relation_label"] = claim_relation
            metadata["canonical_claim_id"] = canonical_claim_id

            canonical_promotion_allowed = (
                provenance_validation.complete
                and anti_triviality.promotable
                and claim_relation != ClaimDecisionLabel.MERGE.value
            )
            metadata["canonical_promotion_allowed"] = canonical_promotion_allowed
            if not canonical_promotion_allowed:
                metadata["observation_only"] = True
                if "observation_only" not in tags_value:
                    tags_value.append("observation_only")
                metadata["tags"] = tags_value
        elif claim_relation:
            if claim_relation not in valid_claim_labels:
                claim_relation_unknown = True
                claim_relation_unknown_reason = f"invalid_input_label:{claim_relation}"
                claim_relation = ClaimDecisionLabel.NEW.value
            metadata["claim_relation_label"] = claim_relation
            metadata["canonical_claim_id"] = canonical_claim_id or str(belief.id)
            metadata["canonical_promotion_allowed"] = bool(
                metadata.get("canonical_promotion_allowed", True),
            )
        else:
            claim_relation_unknown = True
            claim_relation_unknown_reason = "missing_label"
        metadata.setdefault("claim_relation_label", ClaimDecisionLabel.NEW.value)
        metadata.setdefault("canonical_claim_id", str(belief.id))
        metadata.setdefault("canonical_promotion_allowed", True)
        if claim_relation_unknown:
            metadata["claim_relation_unknown"] = True
            metadata["claim_relation_unknown_reason"] = claim_relation_unknown_reason or "unknown"
        else:
            metadata.setdefault("claim_relation_unknown", False)

        if belief.temporal:
            metadata["observed_at"] = belief.temporal.observed_at.isoformat()
            if belief.temporal.valid_from:
                metadata["valid_from"] = belief.temporal.valid_from.isoformat()
            if belief.temporal.valid_until:
                metadata["valid_until"] = belief.temporal.valid_until.isoformat()
            if belief.temporal.last_verified:
                metadata["last_verified"] = belief.temporal.last_verified.isoformat()

        idem_key = self._make_idempotency_key(
            "commit_belief",
            {
                "external_id": external_id,
                "triplet": belief.triplet.as_dict() if belief.triplet else None,
                "content": belief.content,
                "confidence": belief.confidence,
                "status": belief.status.value,
                "claim_relation": metadata.get("claim_relation_label", ""),
                "canonical_claim_id": metadata.get("canonical_claim_id", ""),
            },
        )
        if self._is_duplicate_write(idem_key):
            logger.debug("Skipping duplicate commit_belief for %s", external_id)
            return

        should_store_as_triple = bool(
            belief.triplet and bool(metadata.get("canonical_promotion_allowed", True)),
        )

        if should_store_as_triple and belief.triplet:
            metadata["triplet"] = belief.triplet.as_dict()
            metadata["triplet_key"] = self._triplet_key(
                belief.triplet.subject,
                belief.triplet.predicate,
                belief.triplet.object,
            )
            # Build evidence refs for native evidence links
            evidence_kwargs: dict[str, Any] = {}
            if self._config.use_evidence_links:
                evidence_refs = self._build_evidence_refs(belief)
                if evidence_refs:
                    evidence_kwargs["evidence_refs"] = evidence_refs
                evidence_kwargs["link_confidence"] = belief.confidence
                if belief.source:
                    evidence_kwargs["link_source"] = belief.source.id

            # Store as triple
            async with self._mutation_semaphore:
                triple_result = await self._run_db(
                    "insert_triple",
                    lambda: self._db.insert_triple(
                        external_id=external_id,
                        subject=belief.triplet.subject,
                        predicate=belief.triplet.predicate,
                        object_value=belief.triplet.object,
                        probability=belief.confidence,
                        sources=sources,
                        metadata=metadata,
                        **evidence_kwargs,
                    ),
                    retry=True,
                )
                if self._config.use_consistency_waits:
                    seq = getattr(triple_result, "sequence", None) if triple_result else None
                    if seq is not None:
                        try:
                            self._db.wait_for(seq, consistency="indexed")
                        except Exception:
                            logger.debug("wait_for after insert_triple failed for %s", external_id)
                if not self._config.skip_belief_surface_writes:
                    await self._upsert_belief_surface(
                        belief=belief,
                        belief_external_id=external_id,
                        metadata=metadata,
                        sources=sources,
                    )
                self._mark_write_seen(idem_key)
                await self._write_extraction_item_for_belief(
                    belief=belief,
                    belief_external_id=external_id,
                    storage_type="triple",
                )
                if (
                    metadata.get("claim_relation_label") == ClaimDecisionLabel.CONTRADICTION.value
                    and metadata.get("canonical_claim_id")
                ):
                    try:
                        canonical_uuid = UUID(str(metadata["canonical_claim_id"]))
                    except Exception:
                        canonical_uuid = None
                    if canonical_uuid and canonical_uuid != belief.id:
                        canonical_ext_id = self._build_external_id("belief", canonical_uuid)
                        try:
                            await self._run_db(
                                "add_claim_contradiction_edge_forward",
                                lambda: self._db.add_edge(
                                    from_id=canonical_ext_id,
                                    to_id=external_id,
                                    edge_type="has_conflicting_claim_on",
                                ),
                                retry=True,
                            )
                            await self._run_db(
                                "add_claim_contradiction_edge_backward",
                                lambda: self._db.add_edge(
                                    from_id=external_id,
                                    to_id=canonical_ext_id,
                                    edge_type="has_conflicting_claim_on",
                                ),
                                retry=True,
                            )
                        except Exception:
                            logger.debug(
                                "Failed to add contradiction edges between %s and %s",
                                canonical_ext_id,
                                external_id,
                            )
                return
        # Store as document with belief content
        async with self._mutation_semaphore:
            await self._run_db(
                "ingest_belief",
                lambda: self._db.ingest(
                    external_id=external_id,
                    text=belief.content or "",
                    metadata=metadata,
                    node_type=self.NODE_TYPE_BELIEF,
                    probability=belief.confidence,
                    sources=list(sources.keys()) if sources else None,
                ),
                retry=True,
            )
            self._mark_write_seen(idem_key)
            await self._write_extraction_item_for_belief(
                belief=belief,
                belief_external_id=external_id,
                storage_type=(
                    "merge_observation"
                    if metadata.get("claim_relation_label") == ClaimDecisionLabel.MERGE.value
                    else "document"
                ),
            )

        if (
            metadata.get("claim_relation_label") == ClaimDecisionLabel.MERGE.value
            and metadata.get("canonical_claim_id")
        ):
            try:
                canonical_uuid = UUID(str(metadata["canonical_claim_id"]))
            except Exception:
                canonical_uuid = None
            if canonical_uuid and canonical_uuid != belief.id:
                try:
                    await self.update_belief_confidence(canonical_uuid, delta=0.03)
                except Exception:
                    logger.debug("Failed confidence bump for merged canonical claim %s", canonical_uuid)

    def _build_evidence_refs(self, belief: Belief) -> list[dict[str, Any]]:
        """Build evidence_refs list from belief's source metadata and evidence lists."""
        refs: list[dict[str, Any]] = []
        # From source metadata experiences
        if belief.source and belief.source.metadata:
            experience_ids = belief.source.metadata.get("experiences", [])
            for exp_id in experience_ids:
                exp_ext_id = self._build_external_id("experience", exp_id)
                refs.append({
                    "external_id": exp_ext_id,
                    "relationship": "derived_from",
                })
        # From evidence_for list
        for eid in belief.evidence_for:
            ext_id = self._build_external_id("experience", eid)
            if not any(r["external_id"] == ext_id for r in refs):
                refs.append({
                    "external_id": ext_id,
                    "relationship": "supported_by",
                })
        return refs

    async def get_evidence_for_belief(self, belief_id: UUID) -> list[Any]:
        """Get native evidence edges for a belief via resolve_edges.

        Returns native EvidenceEdge objects with effective_confidence,
        entropy, and risk_adjusted_confidence.
        """
        external_id = self._build_external_id("belief", belief_id)
        try:
            edges = await self._run_db(
                "resolve_edges",
                lambda: self._db.resolve_edges(external_id, k=50),
                retry=True,
            )
            return edges if edges else []
        except Exception:
            logger.debug("resolve_edges failed for %s, falling back to empty", external_id)
            return []

    def _build_belief_surface_text(self, belief: Belief, metadata: dict[str, Any]) -> str:
        """Create rich lexical text for triplet surface retrieval."""
        triplet = belief.triplet
        if not triplet:
            return belief.content or ""

        source_meta = self._normalize_dict(metadata.get("source_metadata"))
        embedding_text = str(source_meta.get("embedding_text") or "").strip()
        if embedding_text:
            return embedding_text

        parts = [f"{triplet.subject} | {triplet.predicate} | {triplet.object}"]
        content = (belief.content or "").strip()
        if content:
            parts.append(content)

        source_doc = source_meta.get("source_document")
        if isinstance(source_doc, dict):
            doc_id = str(source_doc.get("document_id") or source_doc.get("title") or "").strip()
            if doc_id:
                parts.append(f"doc: {doc_id}")
        elif isinstance(source_doc, str):
            doc_id = source_doc.strip()
            if doc_id:
                parts.append(f"doc: {doc_id}")

        date_meta = source_meta.get("date_normalized")
        if isinstance(date_meta, dict):
            canonical = str(date_meta.get("canonical") or "").strip()
            if canonical:
                parts.append(f"date: {canonical}")

        co_triplets = source_meta.get("co_triplet_keys")
        if isinstance(co_triplets, list):
            co_preview = "; ".join(str(item) for item in co_triplets[:5] if str(item).strip())
            if co_preview:
                parts.append(f"context: {co_preview}")

        return " | ".join(parts)

    async def _upsert_belief_surface(
        self,
        belief: Belief,
        belief_external_id: str,
        metadata: dict[str, Any],
        sources: dict[str, float] | None,
    ) -> None:
        """Upsert a semantic surface document for triplet retrieval."""
        if not belief.triplet:
            return

        surface_external_id = self._build_external_id("belief_surface", belief.id)
        surface_text = self._build_belief_surface_text(belief, metadata)
        surface_metadata = dict(metadata)
        surface_metadata["belief_external_id"] = belief_external_id
        surface_metadata["surface_type"] = "triplet_surface"
        surface_metadata["triplet"] = belief.triplet.as_dict()
        surface_metadata["triplet_key"] = self._triplet_key(
            belief.triplet.subject,
            belief.triplet.predicate,
            belief.triplet.object,
        )

        existing = await self._run_db(
            "get_belief_surface",
            lambda: self._db.get(surface_external_id),
            retry=True,
        )
        if existing:
            await self._run_db(
                "update_belief_surface",
                lambda: self._db.update(
                    surface_external_id,
                    text=surface_text,
                    metadata=surface_metadata,
                ),
                retry=True,
            )
            return

        await self._run_db(
            "ingest_belief_surface",
            lambda: self._db.ingest(
                external_id=surface_external_id,
                text=surface_text,
                metadata=surface_metadata,
                node_type=self.NODE_TYPE_BELIEF_SURFACE,
                probability=belief.confidence,
                sources=list(sources.keys()) if sources else None,
            ),
            retry=True,
        )

    async def _write_extraction_item_for_belief(
        self,
        belief: Belief,
        belief_external_id: str,
        storage_type: str,
    ) -> None:
        """Write/update extraction journal item for extracted beliefs."""
        tags = belief.tags or set()
        if "extracted" not in tags:
            return

        existing_meta: dict[str, Any] = {}
        has_existing_item = False
        extraction_external_id = self._build_external_id("extraction", belief.id)
        try:
            existing_doc = self._db.get(extraction_external_id)
            if existing_doc:
                has_existing_item = True
                raw = self._rget(existing_doc, "metadata") or {}
                if isinstance(raw, dict):
                    existing_meta = dict(raw)
        except Exception:
            existing_meta = {}

        source_doc_id = ""
        source_metadata = belief.source.metadata if belief.source else {}
        if isinstance(source_metadata, dict):
            source_doc = source_metadata.get("source_document")
            if isinstance(source_doc, dict):
                source_doc_id = str(source_doc.get("document_id") or source_doc.get("title") or "")
            elif isinstance(source_doc, str):
                source_doc_id = source_doc

        triplet_dict = belief.triplet.as_dict() if belief.triplet else {}
        merged_meta = {
            **existing_meta,
            **self._create_privacy_metadata(),
            "item_type": storage_type,
            "kind": (belief.metadata or {}).get("pattern_type", "belief"),
            "belief_id": str(belief.id),
            "belief_external_id": belief_external_id,
            "source_doc_id": source_doc_id,
            "triplet": json.dumps(triplet_dict) if triplet_dict else "",
            "content": belief.content or "",
            "confidence": str(belief.confidence),
            "tags": list(tags) if isinstance(tags, set) else tags,
            "reflection_processed": bool(existing_meta.get("reflection_processed", False)),
            "active": bool(existing_meta.get("active", True)),
            "created_at": existing_meta.get("created_at", utc_now().isoformat()),
            "last_updated_at": utc_now().isoformat(),
        }
        if isinstance(belief.metadata, dict):
            if belief.metadata.get("canonical_claim_id"):
                merged_meta["canonical_claim_id"] = str(belief.metadata.get("canonical_claim_id"))
            if belief.metadata.get("claim_relation_label") or belief.metadata.get("claim_relation"):
                merged_meta["claim_relation_label"] = str(
                    belief.metadata.get("claim_relation_label")
                    or belief.metadata.get("claim_relation"),
                )
            if "provenance_complete" in belief.metadata:
                merged_meta["provenance_complete"] = bool(belief.metadata.get("provenance_complete"))
            if "observation_only" in belief.metadata:
                merged_meta["observation_only"] = bool(belief.metadata.get("observation_only"))
        if belief.triplet:
            merged_meta["triplet_key"] = (
                f"{belief.triplet.subject.strip().lower()}|"
                f"{belief.triplet.predicate.strip().lower()}|"
                f"{belief.triplet.object.strip().lower()}"
            )

        text = (
            f"extraction item {merged_meta.get('kind')} "
            f"{belief.content or (belief.triplet.as_text() if belief.triplet else '')}"
        )
        if has_existing_item:
            await self._run_db(
                "update_extraction_item",
                lambda: self._db.update(
                    extraction_external_id,
                    text=text,
                    metadata=merged_meta,
                ),
                retry=True,
            )
            if not self._config.use_consistency_waits:
                self._extraction_external_ids.add(extraction_external_id)
            return

        await self._run_db(
            "ingest_extraction_item",
            lambda: self._db.ingest(
                external_id=extraction_external_id,
                text=text,
                metadata=merged_meta,
                node_type=self.NODE_TYPE_EXTRACTION_ITEM,
            ),
            retry=True,
        )
        if not self._config.use_consistency_waits:
            self._extraction_external_ids.add(extraction_external_id)
        return

    async def get_belief(self, belief_id: UUID) -> Belief | None:
        """Get a belief by ID."""
        external_id = self._build_external_id("belief", belief_id)

        # Try triple first
        triples = self._query_triples(k=1)
        for t in triples:
            if t.external_id == external_id:
                # Check access
                if not self._can_access(t.metadata or {}, _external_id=t.external_id):
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
        seen_canonical_ids: set[str] = set()

        # Use SiliconDB search for semantic similarity
        try:
            search_results = await self._run_db(
                "query_beliefs_search",
                lambda: self._weighted_search(query, k=limit * 4, weights_dict=search_weights),
                retry=True,
            )
        except Exception:
            search_results = []

        for r in search_results:
            node_type = self._rget(r, "node_type")
            if node_type not in {self.NODE_TYPE_BELIEF, self.NODE_TYPE_BELIEF_SURFACE}:
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
                canonical_key = self._canonical_claim_key_from_belief(belief)
                if canonical_key in seen_canonical_ids:
                    continue
                seen_canonical_ids.add(canonical_key)
                results.append(belief)

            if len(results) >= limit:
                break

        # Also query triples
        try:
            triples = await self._run_db(
                "query_beliefs_triples",
                lambda: self._query_triples(min_probability=min_confidence, k=limit * 2),
                retry=True,
            )
        except Exception:
            triples = []
        for t in triples:
            # Access control check
            if not self._can_access(
                self._rget(t, "metadata") or {},
                _external_id=self._rget(t, "external_id", ""),
            ):
                continue
            belief = self._triple_to_belief(t)
            if not belief:
                continue
            canonical_key = self._canonical_claim_key_from_belief(belief)
            if canonical_key in seen_canonical_ids:
                continue
            seen_canonical_ids.add(canonical_key)
            results.append(belief)

        return results[:limit]

    async def get_beliefs_by_entity(self, entity: str) -> list[Belief]:
        """Get all beliefs about an entity."""
        # Query triples where entity is subject or object
        try:
            as_subject = await self._run_db(
                "get_beliefs_by_entity_subject",
                lambda: self._query_triples(subject=entity, k=100),
                retry=True,
            )
        except Exception:
            as_subject = []
        try:
            as_object = await self._run_db(
                "get_beliefs_by_entity_object",
                lambda: self._query_triples(object_value=entity, k=100),
                retry=True,
            )
        except Exception:
            as_object = []

        beliefs = []
        seen_ids = set()

        for t in as_subject + as_object:
            # Access control check
            if not self._can_access(
                self._rget(t, "metadata") or {},
                _external_id=self._rget(t, "external_id", ""),
            ):
                continue
            belief = self._triple_to_belief(t)
            if belief and belief.id not in seen_ids:
                beliefs.append(belief)
                seen_ids.add(belief.id)

        return beliefs

    async def get_beliefs_by_tag(
        self,
        tag: str,
        limit: int = 100,
        min_confidence: float = 0.0,
    ) -> list[Belief]:
        """Return beliefs with a specific tag across triple/document storage."""
        if not tag:
            return []
        desired = tag.strip().lower()
        if not desired:
            return []

        results: list[Belief] = []
        seen: set[str] = set()

        try:
            triples = await self._run_db(
                "get_beliefs_by_tag_triples",
                lambda: self._query_triples(k=max(limit * 8, 500)),
                retry=True,
            )
        except Exception:
            triples = []
        for t in triples:
            if self._rget(t, "probability", 1.0) < min_confidence:
                continue
            metadata = self._rget(t, "metadata") or {}
            if not self._can_access(
                metadata,
                _external_id=self._rget(t, "external_id", ""),
            ):
                continue
            tags = self._normalize_tags(metadata.get("tags"))
            if desired not in {str(x).lower() for x in tags}:
                continue
            belief = self._triple_to_belief(t)
            if not belief:
                continue
            canonical_key = self._canonical_claim_key_from_belief(belief)
            if canonical_key in seen:
                continue
            seen.add(canonical_key)
            results.append(belief)
            if len(results) >= limit:
                return results

        try:
            docs = await self._run_db(
                "get_beliefs_by_tag_docs",
                lambda: self._search_by_type(
                    "the",
                    {self.NODE_TYPE_BELIEF, self.NODE_TYPE_BELIEF_SURFACE},
                    target=max(limit * 8, 500),
                ),
                retry=True,
            )
        except Exception:
            docs = []
        for r in docs:
            if self._rget(r, "probability", 1.0) < min_confidence:
                continue
            metadata = self._rget(r, "metadata") or {}
            if not self._can_access(metadata):
                continue
            tags = self._normalize_tags(metadata.get("tags"))
            if desired not in {str(x).lower() for x in tags}:
                continue
            belief = self._search_result_to_belief(r)
            if not belief:
                continue
            canonical_key = self._canonical_claim_key_from_belief(belief)
            if canonical_key in seen:
                continue
            seen.add(canonical_key)
            results.append(belief)
            if len(results) >= limit:
                break

        return results

    async def find_contradictions(self, belief: Belief) -> list[Belief]:
        """Find beliefs that contradict the given belief."""
        if not belief.triplet:
            return []

        # Use SiliconDB's contradiction detection
        try:
            contradictions = await self._run_db(
                "find_contradictions",
                lambda: self._db.detect_triple_contradictions(min_probability=0.0),
                retry=True,
            )
        except Exception:
            contradictions = []

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
            await self._run_db(
                "record_observation_confidence_update",
                lambda: self._db.record_observation(
                    external_id=external_id,
                    confirmed=delta > 0,
                    source="confidence_update",
                ),
                retry=True,
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
        seen_canonical_ids: set[str] = set()

        # Use native entropy reranking when enabled and entropy_weight > 0
        entropy_weight = (search_weights or {}).get("entropy_weight", 0)
        if self._config.use_native_entropy_rerank and entropy_weight > 0:
            try:
                search_results = self._db.search_with_entropy_rerank(
                    query=query,
                    k=limit * 4,
                    candidates=limit * 8,
                    lambda_param=entropy_weight,
                    samples=100,
                )
            except Exception:
                logger.debug("search_with_entropy_rerank unavailable, falling back")
                search_results = self._weighted_search(query, k=limit * 4, weights_dict=search_weights)
        else:
            search_results = self._weighted_search(query, k=limit * 4, weights_dict=search_weights)

        for r in search_results:
            node_type = self._rget(r, "node_type")
            if node_type not in {self.NODE_TYPE_BELIEF, self.NODE_TYPE_BELIEF_SURFACE}:
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
                canonical_key = self._canonical_claim_key_from_belief(belief)
                if canonical_key in seen_canonical_ids:
                    continue
                entropy = self._rget(r, "entropy", 0.0) or 0.0
                results.append((belief, entropy))
                seen_canonical_ids.add(canonical_key)
            if len(results) >= limit:
                break

        # Fallback for triple-backed beliefs (which may not be present in
        # document search results yet).
        if len(results) < limit:
            triples = self._query_triples(min_probability=min_confidence, k=limit * 2)
            for t in triples:
                if not self._can_access(
                    self._rget(t, "metadata") or {},
                    _external_id=self._rget(t, "external_id", ""),
                ):
                    continue
                belief = self._triple_to_belief(t)
                if not belief:
                    continue
                canonical_key = self._canonical_claim_key_from_belief(belief)
                if canonical_key in seen_canonical_ids:
                    continue
                results.append((belief, 0.0))
                seen_canonical_ids.add(canonical_key)
                if len(results) >= limit:
                    break

        return results

    # ========== Belief Consolidation (Monte Carlo + Graph) ==========

    async def build_cooccurrences(
        self,
        belief_ids: list[UUID],
        session_id: str | None = None,
    ) -> None:
        """Link beliefs that co-occur in the same evidence group.

        Creates bidirectional co-occurrence edges in SiliconDB's graph.
        Repeated calls strengthen the link.
        """
        external_ids = [
            self._build_external_id("belief", bid) for bid in belief_ids
        ]
        # Also include __search variants so both triple and doc nodes link
        all_ids = external_ids + [f"{eid}__search" for eid in external_ids]
        try:
            self._db.add_cooccurrences(all_ids, session_id)
        except Exception:
            # Fallback: try without __search variants
            self._db.add_cooccurrences(external_ids, session_id)

    async def monte_carlo_update(
        self,
        evidence: list[dict[str, Any]],
        samples: int = 10000,
        apply: bool = True,
    ) -> list[dict[str, Any]]:
        """Run Monte Carlo probability update on beliefs.

        Args:
            evidence: List of {"external_id": str, "confidence": float}
            samples: Number of MC samples (default 10k)
            apply: If True, also persist updated probabilities

        Returns:
            List of update results with probability, ciLower, ciUpper
        """
        if not evidence:
            return []
        if apply:
            return self._db.update_and_apply_probabilities(evidence, samples)
        return self._db.update_probabilities(evidence, samples)

    async def propagate_belief(
        self,
        belief_id: UUID,
        confidence: float,
        decay: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Propagate evidence through the co-occurrence graph.

        BFS from the source belief, spreading probability updates
        with exponential decay per hop.

        Returns:
            List of propagation updates with previous/new probability and delta.
        """
        external_id = self._build_external_id("belief", belief_id)
        return self._db.propagate(external_id, confidence, decay)

    async def detect_mc_contradictions(
        self,
        samples: int = 10000,
        min_conflict_score: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Detect contradictions between beliefs using Monte Carlo sampling.

        Returns beliefs that are statistically unlikely to be simultaneously true.
        """
        return self._db.detect_contradictions(samples, min_conflict_score)

    async def detect_triple_contradictions(
        self,
        min_probability: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Detect structural contradictions in subject-predicate-object triples.

        Finds cases where the same subject+predicate has conflicting objects.
        """
        return self._db.detect_triple_contradictions(min_probability)

    async def get_uncertain_beliefs(
        self,
        min_entropy: float = 0.5,
        k: int = 50,
    ) -> list[dict[str, Any]]:
        """Get beliefs with highest uncertainty (entropy near 1.0).

        These are beliefs close to 0.5 probability — the system doesn't
        know whether they're true or false. Useful for directing future
        reflection to gather more evidence.
        """
        return self._db.get_uncertain_beliefs(min_entropy, k)

    async def get_related_beliefs(
        self,
        belief_id: UUID,
        min_strength: float = 0.1,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """Get beliefs related by co-occurrence."""
        external_id = self._build_external_id("belief", belief_id)
        return self._db.get_related(external_id, min_strength, k)

    # ========== Episodic Memory (Experiences) ==========

    async def record_experience(self, experience: Experience) -> None:
        """Record an experience."""
        self._ensure_reliability_state()
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
            "context": experience.context or {},
            **privacy_meta,
        }

        if experience.causal_parent:
            metadata["causal_parent"] = str(experience.causal_parent)

        idem_key = self._make_idempotency_key(
            "record_experience",
            {
                "external_id": external_id,
                "content": experience.content,
                "occurred_at": metadata.get("occurred_at"),
                "session_id": experience.session_id,
                "sequence_id": experience.sequence_id,
            },
        )
        if self._is_duplicate_write(idem_key):
            logger.debug("Skipping duplicate record_experience for %s", external_id)
            return

        async with self._mutation_semaphore:
            result = await self._run_db(
                "ingest_experience",
                lambda: self._db.ingest(
                    external_id=external_id,
                    text=experience.content,
                    metadata=metadata,
                    node_type=self.NODE_TYPE_EXPERIENCE,
                ),
                retry=True,
            )
            self._mark_write_seen(idem_key)
            if self._config.use_consistency_waits:
                seq = getattr(result, "sequence", None) if result else None
                if seq is not None:
                    try:
                        self._db.wait_for(seq, consistency="indexed")
                    except Exception:
                        logger.debug("wait_for after experience ingest failed for %s", external_id)
            if not self._config.use_consistency_waits:
                self._experience_external_ids.add(external_id)

        # Add causal edge if parent exists
        if experience.causal_parent:
            parent_id = self._build_external_id("experience", experience.causal_parent)
            try:
                await self._run_db(
                    "add_causal_edge",
                    lambda: self._db.add_edge(
                        from_id=parent_id,
                        to_id=external_id,
                        edge_type="causes",
                    ),
                    retry=True,
                )
            except Exception:
                pass  # Parent may not exist yet

    async def get_experience(self, experience_id: UUID) -> Experience | None:
        """Get an experience by ID."""
        self._ensure_reliability_state()
        external_id = self._build_external_id("experience", experience_id)
        try:
            doc = await self._run_db(
                "get_experience",
                lambda: self._db.get(external_id),
                retry=True,
            )
            if doc:
                metadata = self._rget(doc, "metadata") or {}
                if not self._can_access(metadata, _external_id=external_id):
                    return None
                if not self._config.use_consistency_waits:
                    self._experience_external_ids.add(external_id)
                if isinstance(doc, dict):
                    return self._doc_to_experience(doc)
                return self._doc_to_experience(
                    {
                        "text": self._rget(doc, "text", ""),
                        "metadata": metadata,
                        "probability": self._rget(doc, "probability", 1.0),
                    },
                )
        except Exception:
            pass
        return None

    async def wait_for_experience_visibility(
        self,
        experience_ids: list[UUID],
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.1,
    ) -> bool:
        """Wait until newly ingested experiences are visible in search.

        When use_consistency_waits is enabled, individual writes already
        wait for indexed consistency, so this is a no-op.  Otherwise
        falls back to polling.
        """
        self._ensure_reliability_state()
        if not experience_ids:
            return True

        # With consistency waits, each write already waited for indexed —
        # no additional polling needed.
        if self._config.use_consistency_waits:
            return True

        # Legacy polling path
        pending = {
            self._build_external_id("experience", experience_id)
            for experience_id in experience_ids
        }
        deadline = time.monotonic() + max(0.1, timeout_s)
        poll_interval = max(0.01, poll_interval_s)
        target = max(500, len(pending) * 20)

        while pending and time.monotonic() < deadline:
            try:
                results = self._search_experiences_broad(target=target)
                for result in results:
                    external_id = self._rget(result, "external_id", "")
                    if external_id in pending:
                        pending.discard(external_id)
            except Exception:
                pass

            if pending:
                for external_id in list(pending):
                    doc = await self._get_experience_doc(external_id)
                    if doc:
                        pending.discard(external_id)

            if pending:
                await asyncio.sleep(poll_interval)

        if pending:
            logger.warning(
                "Timed out waiting for %d ingested experiences to become searchable",
                len(pending),
            )
        return not pending

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
        self._ensure_reliability_state()
        cutoff = utc_now() - timedelta(hours=hours)

        # With scan-based retrieval, x6 overfetch is unnecessary and can trigger
        # long-running gRPC scans on larger stores.
        results = await self._search_experiences_broad_async(target=max(limit, 200))
        experiences: list[Experience] = []
        seen_external_ids: set[str] = set()
        for r in results:
            if self._rget(r, "node_type") != self.NODE_TYPE_EXPERIENCE:
                continue
            external_id = self._rget(r, "external_id", "")
            if not external_id or external_id in seen_external_ids:
                continue
            seen_external_ids.add(external_id)
            # Access control check
            if not self._can_access(self._rget(r, "metadata") or {}, _external_id=external_id):
                continue
            exp = self._search_result_to_experience(r)
            if exp and exp.occurred_at >= cutoff:
                experiences.append(exp)
                if not self._config.use_consistency_waits:
                    self._experience_external_ids.add(external_id)

        # Top up via direct-id hydration when search indexing lags.
        # Skipped when use_consistency_waits is enabled (no lag).
        if (
            not self._config.use_consistency_waits
            and len(experiences) < limit
            and self._experience_external_ids
        ):
            for external_id in list(self._experience_external_ids):
                if external_id in seen_external_ids:
                    continue
                exp = await self._hydrate_experience_from_external_id(external_id)
                if not exp:
                    continue
                seen_external_ids.add(external_id)
                if exp.occurred_at >= cutoff:
                    experiences.append(exp)

        # Sort by recency
        experiences.sort(key=lambda e: e.occurred_at, reverse=True)
        return experiences[:limit]

    async def get_unprocessed_experiences(self, limit: int = 100) -> list[Experience]:
        """Get unprocessed experiences for reflection."""
        self._ensure_reliability_state()
        results = await self._search_experiences_broad_async(target=max(limit, 200))
        experiences: list[Experience] = []
        seen_external_ids: set[str] = set()
        for r in results:
            if self._rget(r, "node_type") != self.NODE_TYPE_EXPERIENCE:
                continue
            external_id = self._rget(r, "external_id", "")
            if not external_id or external_id in seen_external_ids:
                continue
            seen_external_ids.add(external_id)
            # Access control check
            metadata = self._rget(r, "metadata") or {}
            if not self._can_access(metadata, _external_id=external_id):
                continue
            exp = self._search_result_to_experience(r)
            if exp and not exp.processed:
                experiences.append(exp)
                if not self._config.use_consistency_waits:
                    self._experience_external_ids.add(external_id)

        # Top up via direct-id hydration — skipped with consistency waits.
        if (
            not self._config.use_consistency_waits
            and len(experiences) < limit
            and self._experience_external_ids
        ):
            for external_id in list(self._experience_external_ids):
                if external_id in seen_external_ids:
                    continue
                exp = await self._hydrate_experience_from_external_id(external_id)
                if not exp:
                    continue
                seen_external_ids.add(external_id)
                if not exp.processed:
                    experiences.append(exp)

        experiences.sort(key=lambda e: e.occurred_at)
        return experiences[:limit]

    async def _update_experience_metadata(
        self,
        external_id: str,
        metadata_patch: dict[str, Any],
        op_name: str,
    ) -> None:
        """Merge a metadata patch into an experience document."""
        self._ensure_reliability_state()

        def _op() -> Any:
            doc = self._db.get(external_id)
            if not doc:
                return None
            metadata = dict(self._rget(doc, "metadata") or {})
            metadata.update(metadata_patch)
            return self._db.update(external_id, metadata=metadata)

        async with self._mutation_semaphore:
            await self._run_db(op_name, _op, retry=True)

    async def mark_experiences_processed(self, experience_ids: list[UUID]) -> None:
        """Mark experiences as processed."""
        for eid in experience_ids:
            external_id = self._build_external_id("experience", eid)
            try:
                await self._update_experience_metadata(
                    external_id,
                    {"processed": True},
                    "mark_experience_processed",
                )
            except Exception:
                pass

    # ========== Procedural Memory (Procedures) ==========

    async def commit_procedure(self, procedure: Procedure) -> None:
        """Store a procedure."""
        self._ensure_reliability_state()
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

        idem_key = self._make_idempotency_key(
            "commit_procedure",
            {
                "external_id": external_id,
                "name": procedure.name,
                "description": procedure.description,
                "steps": procedure.steps,
                "trigger": procedure.trigger,
                "confidence": procedure.confidence,
            },
        )
        if self._is_duplicate_write(idem_key):
            logger.debug("Skipping duplicate commit_procedure for %s", external_id)
            return

        async with self._mutation_semaphore:
            await self._run_db(
                "ingest_procedure",
                lambda: self._db.ingest(
                    external_id=external_id,
                    text="\n".join(text_parts),
                    metadata=metadata,
                    node_type=self.NODE_TYPE_PROCEDURE,
                    probability=procedure.confidence,
                ),
                retry=True,
            )
            self._mark_write_seen(idem_key)

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
        try:
            results = await self._run_db(
                "find_applicable_procedures",
                lambda: self._weighted_search(context, k=limit * 4, weights_dict=search_weights),
                retry=True,
            )
        except Exception:
            results = []

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
                await self._run_db(
                    "record_procedure_observation",
                    lambda: self._db.record_observation(external_id, confirmed=success),
                    retry=True,
                )

            await self._run_db(
                "update_procedure_outcome",
                lambda: self._db.update(external_id, metadata=metadata),
                retry=True,
            )
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
        result = None
        try:
            result = await self._run_db(
                "set_working_update",
                lambda: self._db.update(
                    external_id=external_id,
                    text=str(value),
                    metadata=metadata,
                ),
                retry=True,
            )
        except Exception:
            # Document doesn't exist, create it
            result = await self._run_db(
                "set_working_ingest",
                lambda: self._db.ingest(
                    external_id=external_id,
                    text=str(value),
                    metadata=metadata,
                    node_type=self.NODE_TYPE_WORKING,
                ),
                retry=True,
            )

        if self._config.use_consistency_waits:
            seq = getattr(result, "sequence", None) if result else None
            if seq is not None:
                try:
                    self._db.wait_for(seq, consistency="indexed")
                except Exception:
                    logger.debug("wait_for after set_working failed for %s", key)
        if not self._config.use_consistency_waits:
            self._working_keys.add(key)
            await self._persist_working_keys_index()

    def _working_index_external_id(self) -> str:
        """External ID for persisted working-key index."""
        return self._build_external_id("working-index", "keys")

    async def _persist_working_keys_index(self) -> None:
        """Persist working keys so they survive process restarts."""
        self._ensure_reliability_state()
        external_id = self._working_index_external_id()
        keys_sorted = sorted(self._working_keys)
        metadata = {
            "keys": keys_sorted,
            "updated_at": utc_now().isoformat(),
            **self._create_privacy_metadata(),
        }
        try:
            await self._run_db(
                "persist_working_index_update",
                lambda: self._db.update(
                    external_id=external_id,
                    text=",".join(keys_sorted),
                    metadata=metadata,
                ),
                retry=True,
            )
        except Exception:
            try:
                await self._run_db(
                    "persist_working_index_ingest",
                    lambda: self._db.ingest(
                        external_id=external_id,
                        text=",".join(keys_sorted),
                        metadata=metadata,
                        node_type=self.NODE_TYPE_WORKING_INDEX,
                    ),
                    retry=True,
                )
            except Exception:
                pass

    async def _load_working_keys_from_index(self) -> None:
        """Load persisted working keys when in-memory index is empty."""
        self._ensure_reliability_state()
        external_id = self._working_index_external_id()
        try:
            doc = await self._run_db(
                "load_working_index",
                lambda: self._db.get(external_id),
                retry=True,
            )
        except Exception:
            return
        if not doc:
            return
        metadata = self._rget(doc, "metadata") or {}
        keys_raw = metadata.get("keys")
        if isinstance(keys_raw, list | set | tuple):
            keys = [str(k).strip() for k in keys_raw if str(k).strip()]
        elif isinstance(keys_raw, str):
            parsed = self._normalize_tags(keys_raw)
            keys = [str(k).strip() for k in parsed if str(k).strip()]
            if not keys and keys_raw.strip():
                keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        else:
            keys = []
        self._working_keys.update(keys)

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
                    await self._run_db(
                        "delete_expired_working",
                        lambda: self._db.delete(external_id),
                        retry=True,
                    )
                    if not self._config.use_consistency_waits:
                        self._working_keys.discard(key)
                        await self._persist_working_keys_index()
                    return None

            if not self._config.use_consistency_waits:
                self._working_keys.add(key)
            return metadata.get("value")
        except Exception:
            return None

    async def delete_working(self, key: str) -> bool:
        """Delete a working memory value."""
        external_id = self._build_external_id("working", key)
        try:
            await self._run_db(
                "delete_working",
                lambda: self._db.delete(external_id),
                retry=True,
            )
            if not self._config.use_consistency_waits:
                self._working_keys.discard(key)
                await self._persist_working_keys_index()
            return True
        except Exception:
            return False

    async def get_all_working(self) -> dict[str, Any]:
        """Get all non-expired working memory values.

        When use_consistency_waits is enabled, uses scan() instead of
        in-process key tracking (since writes wait for indexed, scan is
        reliable). Otherwise falls back to key set iteration.
        """
        self._ensure_reliability_state()
        now = utc_now()

        if self._config.use_consistency_waits:
            # Use native scan — no shadow set needed
            result: dict[str, Any] = {}
            try:
                prefix = self._get_user_prefix()
                docs = self._db.scan(node_type=self.NODE_TYPE_WORKING)
                for doc in docs:
                    external_id = self._rget(doc, "external_id", "")
                    if not external_id.startswith(prefix):
                        continue
                    metadata = self._rget(doc, "metadata") or {}
                    if not self._can_access(metadata, _external_id=external_id):
                        continue
                    key = metadata.get("key")
                    if not key:
                        continue
                    expires_at_str = metadata.get("expires_at")
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if now > expires_at:
                            # Expired — clean up
                            try:
                                await self._run_db(
                                    "cleanup_scan_expired",
                                    lambda e=external_id: self._db.delete(e),
                                    retry=True,
                                )
                            except Exception:
                                pass
                            continue
                    value = metadata.get("value")
                    if value is not None:
                        result[key] = value
            except Exception:
                logger.debug("scan(node_type=working) failed, falling back to key set")
                # Fall through to legacy path below
                if result:
                    return result

        # Legacy path: in-process key tracking
        if not self._config.use_consistency_waits:
            if not self._working_keys:
                await self._load_working_keys_from_index()

            result = {}
            expired_keys = []
            for key in list(self._working_keys):
                value = await self.get_working(key)
                if value is not None:
                    result[key] = value
                else:
                    expired_keys.append(key)
            for key in expired_keys:
                self._working_keys.discard(key)
            if expired_keys:
                await self._persist_working_keys_index()

        return result

    async def cleanup_expired_working(self) -> int:
        """Clean up expired working memory entries. Returns count deleted."""
        now = utc_now()
        deleted = 0

        if self._config.use_consistency_waits:
            # Use scan — no shadow set needed
            try:
                prefix = self._get_user_prefix()
                docs = self._db.scan(node_type=self.NODE_TYPE_WORKING)
                for doc in docs:
                    external_id = self._rget(doc, "external_id", "")
                    if not external_id.startswith(prefix):
                        continue
                    metadata = self._rget(doc, "metadata") or {}
                    expires_at_str = metadata.get("expires_at")
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if now > expires_at:
                            try:
                                await self._run_db(
                                    "cleanup_expired_working_delete",
                                    lambda e=external_id: self._db.delete(e),
                                    retry=True,
                                )
                                deleted += 1
                            except Exception:
                                pass
            except Exception:
                logger.debug("scan for cleanup_expired_working failed")
            return deleted

        # Legacy path: in-process key tracking
        expired_keys = []
        for key in list(self._working_keys):
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
                        await self._run_db(
                            "cleanup_expired_working_delete",
                            lambda: self._db.delete(external_id),
                            retry=True,
                        )
                        expired_keys.append(key)
                        deleted += 1
            except Exception:
                pass

        for key in expired_keys:
            self._working_keys.discard(key)
        if expired_keys:
            await self._persist_working_keys_index()

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

        # Build evidence summary — prefer native edges when available
        evidence_summary: dict[UUID, dict[str, Any]] = {}
        if self._config.use_evidence_links:
            for b in beliefs:
                edges = await self.get_evidence_for_belief(b.id)
                evidence_summary[b.id] = {
                    "for": len([e for e in edges if getattr(e, "relationship", "") != "contradicted_by"]),
                    "against": len([e for e in edges if getattr(e, "relationship", "") == "contradicted_by"]),
                    "edges": len(edges),
                }
        else:
            evidence_summary = {
                b.id: {
                    "for": len(b.evidence_for),
                    "against": len(b.evidence_against),
                }
                for b in beliefs
            }

        # Calculate total confidence — delegate temporal decay to SiliconDB
        # when use_native_temporal_decay is enabled (decay already applied
        # via SearchWeights.temporal during query_beliefs).
        if beliefs:
            if self._config.use_native_temporal_decay:
                total_confidence = sum(b.confidence for b in beliefs) / len(beliefs)
            else:
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

        # Build recall results — temporal decay is delegated to SiliconDB
        # via SearchWeights when use_native_temporal_decay is enabled.
        fact_results = []
        for b, ent in facts_with_entropy:
            if self._config.use_native_temporal_decay:
                conf = b.confidence
            else:
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

        # Post-retrieval entropy reranking — delegate to SiliconDB when
        # use_native_entropy_rerank is enabled; otherwise apply in Python.
        entropy_weight = (search_weights or {}).get("entropy_weight", 0)
        if entropy_weight > 0:
            if self._config.use_native_entropy_rerank:
                # Native entropy reranking already applied by
                # search_with_entropy_rerank in _query_beliefs_with_entropy
                pass
            else:
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

    def _search_by_type(
        self,
        query: str,
        node_types: set[str],
        target: int,
        page_size: int = 1000,
        max_scan: int = 50000,
        search_weights: dict[str, Any] | None = None,
    ) -> list:
        """Paginated search and node_type post-filtering."""
        text_weight = 1.0
        vector_weight = 0.0
        if search_weights:
            text_weight = search_weights.get("text", search_weights.get("text_weight", 1.0))
            vector_weight = search_weights.get("vector", search_weights.get("vector_weight", 0.0))

        matched: list = []
        seen: set[str] = set()

        if hasattr(self._db, "search_paginated"):
            for offset in range(0, max_scan, page_size):
                if len(matched) >= target:
                    break
                try:
                    results = self._db.search_paginated(
                        query=query,
                        k=page_size,
                        offset=offset,
                        text_weight=text_weight,
                        vector_weight=vector_weight,
                    )
                except Exception:
                    break
                if not results:
                    break
                for r in results:
                    if self._rget(r, "node_type") not in node_types:
                        continue
                    ext_id = self._rget(r, "external_id", "")
                    if ext_id in seen:
                        continue
                    seen.add(ext_id)
                    matched.append(r)
                    if len(matched) >= target:
                        break
                if len(results) < page_size:
                    break
            return matched

        results = self._weighted_search(query, k=max(target * 4, 200), weights_dict=search_weights)
        for r in results:
            if self._rget(r, "node_type") not in node_types:
                continue
            ext_id = self._rget(r, "external_id", "")
            if ext_id in seen:
                continue
            seen.add(ext_id)
            matched.append(r)
            if len(matched) >= target:
                break
        return matched

    @staticmethod
    def _experience_probe_queries() -> tuple[str, ...]:
        """Generic probe queries for broad episodic recall."""
        return (
            "document",
            "message",
            "meeting",
            "chat",
            "email",
            "report",
            "status",
            "update",
            "task",
            "project",
            "issue",
            "1",
            "2",
            "3",
        )

    def _search_experiences_broad(
        self,
        target: int,
        search_weights: dict[str, Any] | None = None,
    ) -> list:
        """Search experiences across multiple probes and de-duplicate results."""
        wanted = max(1, target)
        merged: list = []
        seen_external_ids: set[str] = set()
        per_probe_target = max(50, min(max(wanted // 2, 50), wanted))
        user_prefix = self._get_user_prefix()

        # Prefer scan-by-node-type when available: it avoids search-index lag
        # and gives deterministic pagination over stored documents.
        if hasattr(self._db, "scan"):
            page_size = min(1000, max(100, wanted))
            scan_cap = max(2000, min(50000, wanted * 20))
            for offset in range(0, scan_cap, page_size):
                if len(merged) >= wanted:
                    break
                try:
                    page = self._db.scan(
                        node_type=self.NODE_TYPE_EXPERIENCE,
                        limit=page_size,
                        offset=offset,
                    )
                except Exception:
                    break
                if not page:
                    break
                for result in page:
                    external_id = self._rget(result, "external_id", "")
                    if not external_id or external_id in seen_external_ids:
                        continue
                    if user_prefix and not external_id.startswith(user_prefix):
                        continue
                    seen_external_ids.add(external_id)
                    merged.append(result)
                    if len(merged) >= wanted:
                        break
                if len(page) < page_size:
                    break

        # If scan produced user-visible experiences, prefer it over search
        # fallbacks. This avoids expensive index probes and keeps retrieval
        # sane even when search indexing is stale.
        if merged:
            return merged[:wanted]

        for probe in self._experience_probe_queries():
            if len(merged) >= wanted:
                break
            try:
                probe_results = self._search_by_type(
                    probe,
                    {self.NODE_TYPE_EXPERIENCE},
                    target=per_probe_target,
                    search_weights=search_weights,
                )
            except Exception:
                continue
            for result in probe_results:
                external_id = self._rget(result, "external_id", "")
                if not external_id or external_id in seen_external_ids:
                    continue
                seen_external_ids.add(external_id)
                merged.append(result)
                if len(merged) >= wanted:
                    break
        return merged

    async def _search_experiences_broad_async(
        self,
        target: int,
        search_weights: dict[str, Any] | None = None,
    ) -> list:
        """Async wrapper for broad episodic search with timeout/retry."""
        try:
            return await self._run_db(
                "search_experiences_broad",
                lambda: self._search_experiences_broad(
                    target=target,
                    search_weights=search_weights,
                ),
                retry=True,
            )
        except Exception:
            return []

    async def _get_experience_doc(self, external_id: str) -> dict[str, Any] | None:
        """Fetch an experience document by external id with access checks."""
        try:
            doc = await self._run_db(
                "get_experience_doc",
                lambda e=external_id: self._db.get(e),
                retry=True,
            )
        except Exception:
            return None
        if not doc:
            return None
        if not isinstance(doc, dict):
            doc = {
                "text": self._rget(doc, "text", ""),
                "metadata": self._rget(doc, "metadata") or {},
                "probability": self._rget(doc, "probability", 1.0),
            }
        metadata = self._rget(doc, "metadata") or {}
        if not self._can_access(metadata, _external_id=external_id):
            return None
        return doc

    async def _hydrate_experience_from_external_id(self, external_id: str) -> Experience | None:
        """Load and convert a single indexed experience."""
        doc = await self._get_experience_doc(external_id)
        if not doc:
            return None
        metadata = self._rget(doc, "metadata") or {}
        if metadata.get("experience_id") and not self._config.use_consistency_waits:
            self._experience_external_ids.add(external_id)
        return self._doc_to_experience(doc)

    # ========== RAPTOR Hierarchical Retrieval ==========

    async def build_raptor_tree(
        self,
        cluster_size: int = 10,
        max_levels: int = 5,
    ) -> dict[str, Any]:
        """Build a RAPTOR hierarchical tree over the document store.

        Wraps SiliconDB's native build_raptor_tree() with the LLM
        provider as summarizer callback. Expensive (GPU k-means) —
        run async, off-request-path.
        """
        try:
            result = await self._run_db(
                "build_raptor_tree",
                lambda: self._db.build_raptor_tree(
                    cluster_size=cluster_size,
                    max_levels=max_levels,
                ),
                retry=True,
            )
            return result if isinstance(result, dict) else {"status": "built"}
        except Exception as e:
            logger.warning("build_raptor_tree failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def search_raptor_hybrid(
        self,
        query: str,
        k: int = 10,
        tree_boost: float = 0.3,
    ) -> list[RecallResult]:
        """Search using RAPTOR hierarchical retrieval.

        Returns multi-granularity results: summaries for overview,
        leaves for detail.
        """
        try:
            results = await self._run_db(
                "search_raptor_hybrid",
                lambda: self._db.search_raptor_hybrid(
                    query=query,
                    k=k,
                    tree_boost=tree_boost,
                ),
                retry=True,
            )
        except Exception:
            logger.debug("search_raptor_hybrid unavailable, falling back to regular search")
            return []

        recall_results: list[RecallResult] = []
        if not results:
            return recall_results

        for r in results:
            text = self._rget(r, "text", "")
            score = self._rget(r, "score", 0.0) or 0.0
            level = self._rget(r, "level", 0) or 0
            metadata = self._rget(r, "metadata") or {}
            if not self._can_access(metadata, _external_id=self._rget(r, "external_id", "")):
                continue
            recall_results.append(RecallResult(
                content=text,
                confidence=score,
                source=None,
                memory_type="semantic" if level == 0 else f"raptor_L{level}",
                relevance_score=score,
            ))

        return recall_results

    async def get_raptor_stats(self) -> dict[str, Any]:
        """Get RAPTOR tree health stats."""
        try:
            stats = self._db.get_raptor_stats()
            return stats if isinstance(stats, dict) else {}
        except Exception:
            return {}

    # ========== Event Stream + Observability ==========

    async def get_event_stats(self) -> dict[str, Any]:
        """Get event stream health stats from SiliconDB.

        Returns buffer utilization, match rate, trigger counts.
        Exposed via /api/v1/status for monitoring.
        """
        try:
            stats = self._db.event_stats()
            return stats if isinstance(stats, dict) else {}
        except Exception:
            return {}

    async def replay_mutations(
        self,
        from_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Replay mutation events for audit/transparency.

        Returns structured EventEnvelope objects for "what changed since X".
        """
        try:
            kwargs: dict[str, Any] = {"limit": limit}
            if from_time:
                kwargs["from_time"] = from_time.isoformat()
            kwargs["event_types"] = [
                "ingest.document.accepted",
                "triple.inserted",
                "belief.observed",
            ]
            events = self._db.replay_events(**kwargs)
            return events if isinstance(events, list) else []
        except Exception:
            return []

    async def detect_triple_contradictions(
        self,
        min_probability: float = 0.3,
    ) -> list[Any]:
        """Detect structural contradictions in subject-predicate-object triples."""
        if hasattr(self._db, "detect_triple_contradictions_async"):
            return await self._db.detect_triple_contradictions_async(min_probability=min_probability)
        try:
            return await asyncio.to_thread(
                self._db.detect_triple_contradictions, min_probability
            )
        except Exception:
            return []

    async def get_uncertain_beliefs(
        self,
        min_entropy: float = 0.5,
        k: int = 50,
    ) -> list[Any]:
        """Get high-entropy beliefs for active learning and reflection."""
        if hasattr(self._db, "get_uncertain_beliefs_async"):
            return await self._db.get_uncertain_beliefs_async(min_entropy=min_entropy, k=k)
        if hasattr(self._db, "get_uncertain_beliefs"):
            try:
                return await asyncio.to_thread(self._db.get_uncertain_beliefs, min_entropy, k)
            except Exception:
                return []

        # Fallback when direct uncertainty API is unavailable.
        candidates = self._search_by_type("the", {self.NODE_TYPE_BELIEF}, target=max(k * 6, 200))
        ranked: list[tuple[float, Any]] = []
        for c in candidates:
            entropy = self._rget(c, "entropy")
            if entropy is None:
                p = float(self._rget(c, "probability", 0.5))
                p = max(1e-6, min(1.0 - 1e-6, p))
                entropy = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
            if entropy >= min_entropy:
                ranked.append((float(entropy), c))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:k]]

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
        """Convert SiliconDB TripleResult to Belief.

        Note: SiliconDB's query_triples CAPI does not return metadata or
        sources in results.  We recover the belief UUID from external_id
        and use safe defaults for the remaining fields.
        """
        try:
            metadata = t.metadata or {}

            # Recover belief_id: prefer metadata, then parse from external_id
            belief_id_str = metadata.get("belief_id")
            if belief_id_str:
                belief_id = UUID(belief_id_str)
            else:
                ext_id = getattr(t, "external_id", "") or ""
                # external_id format: {tenant}/{user}/belief-{uuid}
                if "/belief-" in ext_id:
                    belief_id = UUID(ext_id.split("/belief-", 1)[1])
                else:
                    belief_id = uuid4()

            triplet = Triplet(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object_value,
            )

            source = None
            source_id = str(metadata.get("source_id") or "").strip()
            source_type = self._parse_source_type(metadata.get("source_type"))
            source_meta = self._normalize_dict(metadata.get("source_metadata"))
            source_reliability_raw = metadata.get("source_reliability")
            try:
                source_reliability = float(source_reliability_raw)
            except Exception:
                source_reliability = None

            if source_id:
                source = Source(
                    id=source_id,
                    type=source_type,
                    reliability=source_reliability if source_reliability is not None else 0.5,
                    metadata=source_meta,
                )
            elif t.sources:
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

            # Parse status safely — default to PROVISIONAL when metadata absent
            status_str = metadata.get("status", "provisional")
            try:
                status = BeliefStatus(status_str)
            except ValueError:
                status = BeliefStatus.PROVISIONAL

            return Belief(
                id=belief_id,
                content=str(metadata.get("content", "")),
                triplet=triplet,
                confidence=t.probability,
                source=source,
                status=status,
                tags=self._normalize_tags(metadata.get("tags", [])),
                temporal=temporal,
                evidence_for=self._normalize_uuid_list(metadata.get("evidence_for") or []),
                evidence_against=self._normalize_uuid_list(metadata.get("evidence_against") or []),
                metadata=self._normalize_dict(metadata),
            )
        except Exception as e:
            logger.debug("Failed to convert triple to belief: %s", e)
            return None

    def _doc_to_belief(self, doc: dict) -> Belief | None:
        """Convert SiliconDB document to Belief."""
        try:
            metadata = doc.get("metadata", {})
            status_str = metadata.get("status", "provisional")
            try:
                status = BeliefStatus(status_str)
            except ValueError:
                status = BeliefStatus.PROVISIONAL
            source = None
            source_id = str(metadata.get("source_id") or "").strip()
            if source_id:
                try:
                    source_reliability = float(metadata.get("source_reliability", 0.5))
                except Exception:
                    source_reliability = 0.5
                source = Source(
                    id=source_id,
                    type=self._parse_source_type(metadata.get("source_type")),
                    reliability=source_reliability,
                    metadata=self._normalize_dict(metadata.get("source_metadata")),
                )
            temporal = None
            observed_at = str(metadata.get("observed_at") or "").strip()
            if observed_at:
                temporal = TemporalContext(
                    observed_at=datetime.fromisoformat(observed_at),
                    valid_from=datetime.fromisoformat(metadata["valid_from"]) if metadata.get("valid_from") else None,
                    valid_until=datetime.fromisoformat(metadata["valid_until"]) if metadata.get("valid_until") else None,
                    last_verified=datetime.fromisoformat(metadata["last_verified"]) if metadata.get("last_verified") else None,
                )
            triplet = self._normalize_triplet(metadata.get("triplet"))
            return Belief(
                id=UUID(metadata.get("belief_id", str(uuid4()))),
                content=doc.get("text", "") or str(metadata.get("content", "")),
                triplet=triplet,
                confidence=doc.get("probability", 1.0),
                source=source,
                status=status,
                tags=self._normalize_tags(metadata.get("tags", [])),
                temporal=temporal,
                evidence_for=self._normalize_uuid_list(metadata.get("evidence_for") or []),
                evidence_against=self._normalize_uuid_list(metadata.get("evidence_against") or []),
                metadata=self._normalize_dict(metadata),
            )
        except Exception as e:
            logger.debug("Failed to convert doc to belief: %s", e)
            return None

    def _search_result_to_belief(self, r) -> Belief | None:
        """Convert SiliconDB SearchResult to Belief."""
        try:
            metadata = self._rget(r, "metadata") or {}
            status_str = metadata.get("status", "provisional")
            try:
                status = BeliefStatus(status_str)
            except ValueError:
                status = BeliefStatus.PROVISIONAL
            source = None
            source_id = str(metadata.get("source_id") or "").strip()
            if source_id:
                try:
                    source_reliability = float(metadata.get("source_reliability", 0.5))
                except Exception:
                    source_reliability = 0.5
                source = Source(
                    id=source_id,
                    type=self._parse_source_type(metadata.get("source_type")),
                    reliability=source_reliability,
                    metadata=self._normalize_dict(metadata.get("source_metadata")),
                )
            temporal = None
            observed_at = str(metadata.get("observed_at") or "").strip()
            if observed_at:
                temporal = TemporalContext(
                    observed_at=datetime.fromisoformat(observed_at),
                    valid_from=datetime.fromisoformat(metadata["valid_from"]) if metadata.get("valid_from") else None,
                    valid_until=datetime.fromisoformat(metadata["valid_until"]) if metadata.get("valid_until") else None,
                    last_verified=datetime.fromisoformat(metadata["last_verified"]) if metadata.get("last_verified") else None,
                )
            triplet = self._normalize_triplet(metadata.get("triplet"))
            return Belief(
                id=UUID(metadata.get("belief_id", str(uuid4()))),
                content=self._rget(r, "text", "") or str(metadata.get("content", "")),
                triplet=triplet,
                confidence=self._rget(r, "probability", 1.0),
                source=source,
                status=status,
                tags=self._normalize_tags(metadata.get("tags", [])),
                temporal=temporal,
                evidence_for=self._normalize_uuid_list(metadata.get("evidence_for") or []),
                evidence_against=self._normalize_uuid_list(metadata.get("evidence_against") or []),
                metadata=self._normalize_dict(metadata),
            )
        except Exception as e:
            logger.debug("Failed to convert search result to belief: %s", e)
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
                context=self._normalize_dict(metadata.get("context")),
                processed=self._as_bool(metadata.get("processed"), default=False),
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
                context=self._normalize_dict(metadata.get("context")),
                processed=self._as_bool(metadata.get("processed"), default=False),
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

    async def update_belief_status(
        self,
        belief_id: UUID,
        new_status: BeliefStatus,
        reason: str = "",
    ) -> bool:
        """Update lifecycle status metadata on belief documents."""
        external_id = self._build_external_id("belief", belief_id)
        updated = False

        for candidate_id in (external_id, f"{external_id}__search"):
            try:
                doc = await self._run_db(
                    "get_belief_doc_for_status_update",
                    lambda: self._db.get(candidate_id),
                    retry=True,
                )
                if not doc:
                    continue
                metadata = dict(doc.get("metadata", {}))
                metadata["status"] = new_status.value
                if reason:
                    metadata["status_reason"] = reason
                await self._run_db(
                    "update_belief_status_doc",
                    lambda: self._db.update(candidate_id, metadata=metadata),
                    retry=True,
                )
                updated = True
            except Exception:
                continue

        return updated

    async def mark_beliefs_reflection_processed(self, external_ids: list[str]) -> None:
        """Mark extracted belief docs as processed by reflection."""
        if not external_ids:
            return
        stamped_at = utc_now().isoformat()
        seen: set[str] = set()
        for external_id in external_ids:
            if not external_id or external_id in seen:
                continue
            seen.add(external_id)
            try:
                existing_meta: dict[str, Any] = {}
                try:
                    existing_doc = self._db.get(external_id)
                    if existing_doc:
                        raw_meta = self._rget(existing_doc, "metadata") or {}
                        if isinstance(raw_meta, dict):
                            existing_meta = dict(raw_meta)
                except Exception:
                    existing_meta = {}

                merged_meta = {
                    **existing_meta,
                    "reflection_processed": True,
                    "reflection_processed_at": stamped_at,
                }
                async with self._mutation_semaphore:
                    await self._run_db(
                        "mark_belief_reflection_processed",
                        lambda: self._db.update(
                            external_id,
                            metadata=merged_meta,
                        ),
                        retry=True,
                    )
                    self._mark_write_seen(
                        self._make_idempotency_key(
                            "mark_belief_reflection_processed",
                            {"external_id": external_id, "ts": stamped_at},
                        )
                    )
            except Exception:
                logger.debug("Unable to mark reflection processed for %s", external_id)

    async def get_unprocessed_extraction_items(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return unprocessed extraction journal items (doc + triple sources)."""
        self._ensure_reliability_state()
        user_prefix = self._get_user_prefix()
        results = self._search_by_type(
            "",
            {self.NODE_TYPE_EXTRACTION_ITEM},
            target=max(50, limit * 4),
        )
        items: list[dict[str, Any]] = []
        seen_external_ids: set[str] = set()
        for r in results:
            external_id = self._rget(r, "external_id", "")
            if not external_id.startswith(f"{user_prefix}extraction-"):
                continue
            metadata = self._rget(r, "metadata") or {}
            if not self._can_access(metadata):
                continue
            if not metadata.get("belief_id"):
                continue
            if self._as_bool(metadata.get("reflection_processed"), default=False):
                continue
            if not self._as_bool(metadata.get("active"), default=True):
                continue
            if external_id in seen_external_ids:
                continue
            seen_external_ids.add(external_id)
            items.append({"external_id": external_id, "metadata": metadata})
            if len(items) >= limit:
                break

        if len(items) >= limit:
            return items

        # Direct-ID fallback from locally tracked extraction items written/read in
        # this process. Skipped when use_consistency_waits is enabled (no lag).
        if not self._config.use_consistency_waits and self._extraction_external_ids:
            for extraction_external_id in list(self._extraction_external_ids):
                if not extraction_external_id.startswith(f"{user_prefix}extraction-"):
                    continue
                if extraction_external_id in seen_external_ids:
                    continue
                try:
                    doc = await self._run_db(
                        "get_extraction_item_indexed",
                        lambda e=extraction_external_id: self._db.get(e),
                        retry=True,
                    )
                except Exception:
                    continue
                metadata = self._rget(doc, "metadata") or {}
                if not self._can_access(metadata, _external_id=extraction_external_id):
                    continue
                if not metadata.get("belief_id"):
                    continue
                if self._as_bool(metadata.get("reflection_processed"), default=False):
                    continue
                if not self._as_bool(metadata.get("active"), default=True):
                    continue
                seen_external_ids.add(extraction_external_id)
                items.append({"external_id": extraction_external_id, "metadata": metadata})
                if len(items) >= limit:
                    return items

        # Top-up fallback when extraction-item search is partial/stale: derive extraction IDs
        # from known beliefs and fetch directly by external_id.
        belief_external_ids: set[str] = set()
        try:
            triples = self._query_triples(k=max(limit * 8, 500))
            for t in triples:
                ext_id = self._rget(t, "external_id", "")
                if ext_id.startswith(f"{user_prefix}belief-"):
                    belief_external_ids.add(ext_id)
        except Exception:
            pass

        try:
            belief_docs = self._search_by_type(
                "the",
                {self.NODE_TYPE_BELIEF},
                target=max(limit * 8, 500),
            )
            for r in belief_docs:
                ext_id = self._rget(r, "external_id", "")
                if ext_id.startswith(f"{user_prefix}belief-"):
                    belief_external_ids.add(ext_id)
        except Exception:
            pass

        for belief_external_id in belief_external_ids:
            belief_suffix = belief_external_id.rsplit("belief-", 1)[-1]
            extraction_external_id = f"{user_prefix}extraction-{belief_suffix}"
            if extraction_external_id in seen_external_ids:
                continue
            try:
                doc = await self._run_db(
                    "get_extraction_item_fallback",
                    lambda e=extraction_external_id: self._db.get(e),
                    retry=True,
                )
            except Exception:
                continue
            metadata = self._rget(doc, "metadata") or {}
            if not self._can_access(metadata):
                continue
            if not metadata.get("belief_id"):
                continue
            if self._as_bool(metadata.get("reflection_processed"), default=False):
                continue
            if not self._as_bool(metadata.get("active"), default=True):
                continue
            seen_external_ids.add(extraction_external_id)
            items.append({"external_id": extraction_external_id, "metadata": metadata})
            if len(items) >= limit:
                break
        return items

    async def mark_extraction_items_processed(
        self,
        extraction_external_ids: list[str],
        run_id: str = "",
    ) -> None:
        """Mark extraction journal items as processed by reflection."""
        if not extraction_external_ids:
            return
        stamped_at = utc_now().isoformat()
        for external_id in list(dict.fromkeys(extraction_external_ids)):
            if not external_id:
                continue
            try:
                existing_meta: dict[str, Any] = {}
                try:
                    existing_doc = self._db.get(external_id)
                    if existing_doc:
                        raw_meta = self._rget(existing_doc, "metadata") or {}
                        if isinstance(raw_meta, dict):
                            existing_meta = dict(raw_meta)
                except Exception:
                    existing_meta = {}

                merged_meta = {
                    **existing_meta,
                    "reflection_processed": True,
                    "reflection_processed_at": stamped_at,
                    "reflection_run_id": run_id,
                }
                async with self._mutation_semaphore:
                    await self._run_db(
                        "mark_extraction_item_processed",
                        lambda: self._db.update(
                            external_id,
                            metadata=merged_meta,
                        ),
                        retry=True,
                    )
            except Exception:
                logger.debug("Unable to mark extraction item processed for %s", external_id)

    async def record_reflection_run(
        self,
        run_id: str,
        status: str,
        metrics: dict[str, Any],
    ) -> None:
        """Store a reflection run journal record."""
        external_id = self._build_external_id("reflection-run", run_id)
        metadata = {
            **self._create_privacy_metadata(),
            "run_id": run_id,
            "status": status,
            "run_metrics": json.dumps(metrics),
            "created_at": utc_now().isoformat(),
        }
        text = f"reflection run {run_id} status={status}"
        try:
            await self._run_db(
                "ingest_reflection_run",
                lambda: self._db.ingest(
                    external_id=external_id,
                    text=text,
                    metadata=metadata,
                    node_type=self.NODE_TYPE_REFLECTION_RUN,
                ),
                retry=True,
            )
        except Exception:
            logger.debug("Failed to record reflection run %s", run_id)

    async def record_dream_run(
        self,
        run_id: str,
        status: str,
        metrics: dict[str, Any],
    ) -> None:
        """Store a dream run journal record."""
        external_id = self._build_external_id("dream-run", run_id)
        metadata = {
            **self._create_privacy_metadata(),
            "run_id": run_id,
            "status": status,
            "run_metrics": json.dumps(metrics),
            "created_at": utc_now().isoformat(),
        }
        text = f"dream run {run_id} status={status}"
        try:
            await self._run_db(
                "ingest_dream_run",
                lambda: self._db.ingest(
                    external_id=external_id,
                    text=text,
                    metadata=metadata,
                    node_type=self.NODE_TYPE_DREAM_RUN,
                ),
                retry=True,
            )
        except Exception:
            logger.debug("Failed to record dream run %s", run_id)

    async def get_beliefs_from_experience(self, experience_id: UUID) -> list[Belief]:
        """Return beliefs that cite the given experience as evidence."""
        beliefs = await self.query_beliefs(str(experience_id), limit=200, min_confidence=0.0, include_contested=True)
        matched: list[Belief] = []
        for belief in beliefs:
            evidence_for = {str(e) for e in (belief.evidence_for or [])}
            if str(experience_id) in evidence_for:
                matched.append(belief)
        return matched

    async def get_unextracted_experiences(self, limit: int = 10000) -> list[Experience]:
        """Get experiences where extracted is false or missing."""
        self._ensure_reliability_state()
        experiences: list[Experience] = []
        seen_external_ids: set[str] = set()
        results = await self._search_experiences_broad_async(target=max(limit, 200))
        for r in results:
            external_id = self._rget(r, "external_id", "")
            if not external_id or external_id in seen_external_ids:
                continue
            seen_external_ids.add(external_id)
            metadata = self._rget(r, "metadata") or {}
            if self._as_bool(metadata.get("extracted"), default=False):
                continue
            if not self._can_access(metadata, _external_id=external_id):
                continue
            exp = self._search_result_to_experience(r)
            if exp:
                experiences.append(exp)
                if not self._config.use_consistency_waits:
                    self._experience_external_ids.add(external_id)
                if len(experiences) >= limit:
                    break

        # Fallback hydration — skipped when consistency waits are enabled.
        if (
            not self._config.use_consistency_waits
            and len(experiences) < limit
            and self._experience_external_ids
        ):
            for external_id in list(self._experience_external_ids):
                if external_id in seen_external_ids:
                    continue
                doc = await self._get_experience_doc(external_id)
                if not doc:
                    continue
                seen_external_ids.add(external_id)
                metadata = self._rget(doc, "metadata") or {}
                if self._as_bool(metadata.get("extracted"), default=False):
                    continue
                exp = self._doc_to_experience(doc)
                if exp:
                    experiences.append(exp)
                    if len(experiences) >= limit:
                        break
        experiences.sort(key=lambda e: e.occurred_at)
        return experiences

    async def mark_experiences_extracted(self, experience_ids: list[UUID]) -> None:
        """Mark experiences as extracted."""
        for eid in experience_ids:
            external_id = self._build_external_id("experience", eid)
            try:
                await self._update_experience_metadata(
                    external_id,
                    {"extracted": True},
                    "mark_experience_extracted",
                )
            except Exception:
                continue

    async def count_extraction_progress(self) -> dict[str, int]:
        """Count extracted and unextracted experiences."""
        self._ensure_reliability_state()
        extracted = 0
        unextracted = 0
        seen_external_ids: set[str] = set()
        # Cap full-store scans in dev loops to keep progress checks responsive.
        results = await self._search_experiences_broad_async(target=2000)
        for r in results:
            external_id = self._rget(r, "external_id", "")
            if not external_id or external_id in seen_external_ids:
                continue
            seen_external_ids.add(external_id)
            metadata = self._rget(r, "metadata") or {}
            if not self._can_access(metadata, _external_id=external_id):
                continue
            if self._as_bool(metadata.get("extracted"), default=False):
                extracted += 1
            else:
                unextracted += 1

        # Fallback from shadow set — skipped when consistency waits are enabled.
        if not self._config.use_consistency_waits:
            for external_id in list(self._experience_external_ids):
                if external_id in seen_external_ids:
                    continue
                doc = await self._get_experience_doc(external_id)
                if not doc:
                    continue
                seen_external_ids.add(external_id)
                metadata = self._rget(doc, "metadata") or {}
                if not self._can_access(metadata, _external_id=external_id):
                    continue
                if self._as_bool(metadata.get("extracted"), default=False):
                    extracted += 1
                else:
                    unextracted += 1
        return {"extracted": extracted, "unextracted": unextracted, "total": extracted + unextracted}

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

        idem_key = self._make_idempotency_key(
            "commit_decision",
            {"external_id": ext_id, "decision": decision.to_dict()},
        )
        if self._is_duplicate_write(idem_key):
            logger.debug("Skipping duplicate commit_decision for %s", ext_id)
            return

        async with self._mutation_semaphore:
            await self._run_db(
                "ingest_decision",
                lambda: self._db.ingest(
                    external_id=ext_id,
                    text=text,
                    node_type="decision",
                    metadata=metadata,
                ),
                retry=True,
            )
            self._mark_write_seen(idem_key)

        # Create graph edges for assumptions
        for assumption in decision.assumptions:
            belief_ext_id = str(assumption.belief_id)
            try:
                await self._run_db(
                    "add_decision_assumption_edge",
                    lambda: self._db.add_edge(ext_id, belief_ext_id, "assumes"),
                    retry=True,
                )
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
        external_id = self._build_external_id("decision", decision_id)

        # Direct lookup by external_id is the most reliable path, especially
        # immediately after writes when search indexes may lag.
        try:
            doc = await self._run_db(
                "get_decision_doc",
                lambda: self._db.get(external_id),
                retry=True,
            )
            if doc:
                metadata = doc.get("metadata", {})
                decision_data = metadata.get("decision_data")
                if decision_data:
                    return Decision.from_dict(json.loads(decision_data))
        except Exception:
            pass

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
        idem_key = self._make_idempotency_key(
            "record_decision_outcome",
            {"decision_id": str(decision_id), "outcome": outcome},
        )
        if self._is_duplicate_write(idem_key):
            return True

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
                async with self._mutation_semaphore:
                    await self._run_db(
                        "update_decision_outcome",
                        lambda: self._db.update(ext_id, metadata=metadata),
                        retry=True,
                    )
                self._mark_write_seen(idem_key)
                return True
        return False

    async def update_decision_status(
        self,
        decision_id: UUID,
        status: DecisionStatus,
        reason: str | None = None,
    ) -> bool:
        """Persist decision status transition and optional reason."""
        import json

        decision = await self.get_decision(decision_id)
        if not decision:
            return False
        decision.status = status
        if reason:
            decision.metadata["status_reason"] = reason
            decision.metadata["status_updated_at"] = utc_now().isoformat()
        idem_key = self._make_idempotency_key(
            "update_decision_status",
            {
                "decision_id": str(decision_id),
                "status": status.value,
                "reason": reason or "",
            },
        )
        if self._is_duplicate_write(idem_key):
            return True

        results = self._db.search(
            query=str(decision_id),
            k=5,
            filter={"node_type": "decision"},
        )
        for r in results:
            metadata = r.metadata or {} if hasattr(r, "metadata") else r.get("metadata", {})
            if metadata.get("decision_id") != str(decision_id):
                continue
            ext_id = r.external_id if hasattr(r, "external_id") else r.get("external_id", "")
            metadata["decision_data"] = json.dumps(decision.to_dict())
            metadata["status"] = status.value
            if reason:
                metadata["status_reason"] = reason
            async with self._mutation_semaphore:
                await self._run_db(
                    "update_decision_status",
                    lambda: self._db.update(ext_id, metadata=metadata),
                    retry=True,
                )
            self._mark_write_seen(idem_key)
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
                await self._run_db(
                    "mark_decision_superseded",
                    lambda: self._db.update(ext_id, metadata=metadata),
                    retry=True,
                )
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
