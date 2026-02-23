"""Tests for SiliconDB native capabilities refactor (Phases 1-5).

Validates evidence links, search delegation, consistency waits,
event stream, RAPTOR retrieval, and predicate management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import UUID, uuid4

import pytest

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    RecallResult,
    Source,
    SourceType,
    TemporalContext,
    Triplet,
)
from silicon_memory.security.types import UserContext
from silicon_memory.storage.silicondb_backend import SiliconDBBackend, SiliconDBConfig
from silicon_memory.retrieval.salience import PROFILES, SalienceProfile


# ========== Fixtures ==========

def _make_user_context() -> UserContext:
    return UserContext(user_id="user1", tenant_id="tenant1")


def _make_config(**overrides) -> SiliconDBConfig:
    defaults = {
        "path": "/tmp/test_db",
        "use_evidence_links": True,
        "use_native_entropy_rerank": True,
        "use_native_temporal_decay": True,
        "skip_belief_surface_writes": True,
        "use_consistency_waits": True,
        "use_event_stream": True,
        "enable_raptor": False,
    }
    defaults.update(overrides)
    return SiliconDBConfig(**defaults)


def _make_belief(
    belief_id: UUID | None = None,
    content: str = "test belief",
    confidence: float = 0.8,
    subject: str = "Alice",
    predicate: str = "works_at",
    obj: str = "Acme",
    source_experiences: list[str] | None = None,
) -> Belief:
    bid = belief_id or uuid4()
    source_metadata = {}
    if source_experiences:
        source_metadata["experiences"] = source_experiences
    return Belief(
        id=bid,
        content=content,
        triplet=Triplet(subject=subject, predicate=predicate, object=obj),
        confidence=confidence,
        source=Source(
            id="reflection_engine",
            type=SourceType.REFLECTION,
            reliability=0.7,
            metadata=source_metadata,
        ),
        status=BeliefStatus.PROVISIONAL,
        evidence_for=[],
        evidence_against=[],
        tags=set(),
        metadata={},
    )


def _make_backend(config: SiliconDBConfig | None = None) -> SiliconDBBackend:
    """Create a SiliconDBBackend with mocked SiliconDB client."""
    cfg = config or _make_config()
    user_ctx = _make_user_context()

    # SiliconDBClient is imported lazily in __init__, mock at the silicondb module level
    mock_db = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_db)
    mock_module = MagicMock()
    mock_module.SiliconDBClient = mock_client_cls

    import sys
    original = sys.modules.get("silicondb")
    sys.modules["silicondb"] = mock_module
    try:
        backend = SiliconDBBackend(cfg, user_ctx)
    finally:
        if original is not None:
            sys.modules["silicondb"] = original
        else:
            sys.modules.pop("silicondb", None)

    return backend


# ========== Phase 1: Evidence Links ==========


class TestEvidenceLinks:
    """Test evidence_refs on triples."""

    def test_build_evidence_refs_from_source_experiences(self):
        backend = _make_backend()
        belief = _make_belief(source_experiences=["exp-001", "exp-002"])
        refs = backend._build_evidence_refs(belief)
        assert len(refs) == 2
        assert refs[0]["relationship"] == "derived_from"
        assert "experience" in refs[0]["external_id"]

    def test_build_evidence_refs_empty_when_no_experiences(self):
        backend = _make_backend()
        belief = _make_belief()
        refs = backend._build_evidence_refs(belief)
        assert refs == []

    def test_build_evidence_refs_deduplicates(self):
        backend = _make_backend()
        exp_id = str(uuid4())
        belief = _make_belief(source_experiences=[exp_id])
        belief.evidence_for = [UUID(exp_id)]
        refs = backend._build_evidence_refs(belief)
        # exp_id appears in both source.metadata.experiences and evidence_for
        # but should only appear once
        ext_ids = [r["external_id"] for r in refs]
        assert len(ext_ids) == len(set(ext_ids))

    @pytest.mark.asyncio
    async def test_get_evidence_for_belief_calls_resolve_edges(self):
        backend = _make_backend()
        bid = uuid4()
        mock_edges = [MagicMock(external_id="edge1"), MagicMock(external_id="edge2")]
        backend._db.resolve_edges = MagicMock(return_value=mock_edges)

        edges = await backend.get_evidence_for_belief(bid)
        assert len(edges) == 2
        backend._db.resolve_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_evidence_for_belief_returns_empty_on_error(self):
        backend = _make_backend()
        backend._db.resolve_edges = MagicMock(side_effect=Exception("not found"))

        edges = await backend.get_evidence_for_belief(uuid4())
        assert edges == []

    @pytest.mark.asyncio
    async def test_commit_belief_passes_evidence_kwargs(self):
        backend = _make_backend()
        belief = _make_belief(source_experiences=["exp-001"])

        # Track what insert_triple was called with
        captured_kwargs: dict[str, Any] = {}

        def capture_insert(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(sequence=42)

        backend._db.insert_triple = capture_insert
        backend._db.get = MagicMock(return_value=None)
        backend._db.wait_for = MagicMock()
        backend._db.ingest = MagicMock(return_value=MagicMock(sequence=1))
        # Make _run_db call the lambda directly (skip threading)
        async def direct_run_db(op_name, fn, *, timeout_s=None, retry=True):
            return fn()
        backend._run_db = direct_run_db
        # Skip complex claim logic
        backend._looks_reflection_generated = MagicMock(return_value=False)

        await backend.commit_belief(belief)

        assert "evidence_refs" in captured_kwargs
        assert captured_kwargs["link_confidence"] == belief.confidence
        assert captured_kwargs["link_source"] == "reflection_engine"

    @pytest.mark.asyncio
    async def test_commit_belief_skips_evidence_when_disabled(self):
        cfg = _make_config(use_evidence_links=False)
        backend = _make_backend(cfg)
        belief = _make_belief(source_experiences=["exp-001"])

        captured_kwargs: dict[str, Any] = {}

        def capture_insert(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(sequence=42)

        backend._db.insert_triple = capture_insert
        backend._db.get = MagicMock(return_value=None)
        backend._db.wait_for = MagicMock()
        backend._db.ingest = MagicMock(return_value=MagicMock(sequence=1))
        async def direct_run_db(op_name, fn, *, timeout_s=None, retry=True):
            return fn()
        backend._run_db = direct_run_db
        backend._looks_reflection_generated = MagicMock(return_value=False)

        await backend.commit_belief(belief)

        assert "evidence_refs" not in captured_kwargs


# ========== Phase 2: Search Delegation ==========


class TestSearchDelegation:
    """Test native entropy reranking, temporal decay delegation, and
    belief surface write gating."""

    @pytest.mark.asyncio
    async def test_commit_belief_skips_surface_write(self):
        backend = _make_backend()
        belief = _make_belief()

        backend._db.insert_triple = MagicMock(return_value=MagicMock(sequence=42))
        backend._db.get = MagicMock(return_value=None)
        backend._db.wait_for = MagicMock()
        backend._db.ingest = MagicMock(return_value=MagicMock(sequence=1))
        async def direct_run_db(op_name, fn, *, timeout_s=None, retry=True):
            return fn()
        backend._run_db = direct_run_db
        backend._looks_reflection_generated = MagicMock(return_value=False)
        backend._upsert_belief_surface = AsyncMock()

        await backend.commit_belief(belief)

        # Should NOT be called when skip_belief_surface_writes=True
        backend._upsert_belief_surface.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_belief_writes_surface_when_flag_off(self):
        cfg = _make_config(skip_belief_surface_writes=False)
        backend = _make_backend(cfg)
        belief = _make_belief()

        backend._db.insert_triple = MagicMock(return_value=MagicMock(sequence=42))
        backend._db.get = MagicMock(return_value=None)
        backend._db.wait_for = MagicMock()
        backend._db.ingest = MagicMock(return_value=MagicMock(sequence=1))
        async def direct_run_db(op_name, fn, *, timeout_s=None, retry=True):
            return fn()
        backend._run_db = direct_run_db
        backend._looks_reflection_generated = MagicMock(return_value=False)
        backend._upsert_belief_surface = AsyncMock()

        await backend.commit_belief(belief)

        backend._upsert_belief_surface.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_skips_python_decay_when_native(self):
        backend = _make_backend()

        # Provide facts with temporal context
        now = datetime.now(timezone.utc)
        beliefs = [
            (
                _make_belief(),
                0.1,  # entropy
            ),
        ]
        beliefs[0][0].temporal = TemporalContext(observed_at=now)

        backend._query_beliefs_with_entropy = AsyncMock(return_value=beliefs)
        backend.query_experiences = AsyncMock(return_value=[])
        backend.find_applicable_procedures = AsyncMock(return_value=[])
        backend.get_all_working = AsyncMock(return_value={})

        result = await backend.recall("test query")

        # Confidence should be unchanged (no decay applied)
        assert result["facts"][0].confidence == beliefs[0][0].confidence

    @pytest.mark.asyncio
    async def test_entropy_rerank_uses_native_when_enabled(self):
        backend = _make_backend()

        # Mock the native search
        backend._db.search_with_entropy_rerank = MagicMock(return_value=[])
        backend._db.search = MagicMock(return_value=[])

        await backend._query_beliefs_with_entropy(
            "test", limit=10,
            search_weights={"entropy_weight": 0.5},
        )

        backend._db.search_with_entropy_rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_entropy_rerank_falls_back_when_unavailable(self):
        cfg = _make_config(use_native_entropy_rerank=True)
        backend = _make_backend(cfg)

        # Simulate native call failing
        backend._db.search_with_entropy_rerank = MagicMock(side_effect=Exception("unavailable"))
        backend._db.search = MagicMock(return_value=[])

        # Should not raise
        results = await backend._query_beliefs_with_entropy(
            "test", limit=10,
            search_weights={"entropy_weight": 0.5},
        )
        assert isinstance(results, list)

    def test_deep_recall_profile_has_tree_boost(self):
        profile = PROFILES["deep_recall"]
        assert profile.tree_boost > 0
        weights = profile.to_search_weights()
        assert "tree_boost" in weights


# ========== Phase 3A: Consistency Waits ==========


class TestConsistencyWaits:
    """Test consistency waits and shadow set elimination."""

    @pytest.mark.asyncio
    async def test_record_experience_calls_wait_for(self):
        backend = _make_backend()

        from silicon_memory.core.types import Experience
        exp = Experience(
            id=uuid4(),
            content="test experience",
            occurred_at=datetime.now(timezone.utc),
        )

        # Mock ingest to return result with sequence
        result_mock = MagicMock(sequence=42)
        backend._db.ingest = MagicMock(return_value=result_mock)
        backend._db.wait_for = MagicMock()

        await backend.record_experience(exp)

        backend._db.wait_for.assert_called_once_with(42, consistency="indexed")

    @pytest.mark.asyncio
    async def test_record_experience_skips_shadow_set_with_waits(self):
        backend = _make_backend()

        from silicon_memory.core.types import Experience
        exp = Experience(
            id=uuid4(),
            content="test experience",
            occurred_at=datetime.now(timezone.utc),
        )

        result_mock = MagicMock(sequence=42)
        backend._db.ingest = MagicMock(return_value=result_mock)
        backend._db.wait_for = MagicMock()

        await backend.record_experience(exp)

        # Shadow set should NOT be populated when consistency waits are on
        assert len(backend._experience_external_ids) == 0

    @pytest.mark.asyncio
    async def test_wait_for_visibility_is_noop_with_waits(self):
        backend = _make_backend()

        result = await backend.wait_for_experience_visibility([uuid4()])
        assert result is True

    @pytest.mark.asyncio
    async def test_get_all_working_uses_scan(self):
        backend = _make_backend()

        prefix = backend._get_user_prefix()
        mock_docs = [
            MagicMock(
                external_id=f"{prefix}working-key1",
                metadata={
                    "key": "key1",
                    "value": "value1",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                    "owner_id": "user1",
                    "tenant_id": "tenant1",
                },
                node_type="working",
            ),
        ]
        # Make _rget work with mock objects
        for doc in mock_docs:
            doc.metadata = doc.metadata
            doc.node_type = "working"

        backend._db.scan = MagicMock(return_value=mock_docs)

        result = await backend.get_all_working()

        backend._db.scan.assert_called_once_with(node_type="working")

    @pytest.mark.asyncio
    async def test_set_working_calls_wait_for(self):
        backend = _make_backend()

        result_mock = MagicMock(sequence=99)
        backend._db.update = MagicMock(return_value=result_mock)
        backend._db.wait_for = MagicMock()

        await backend.set_working("key1", "value1")

        backend._db.wait_for.assert_called_once_with(99, consistency="indexed")


# ========== Phase 3B: Event-Driven Architecture ==========


class TestEventDrivenArchitecture:
    """Test event stream integration."""

    @pytest.mark.asyncio
    async def test_get_event_stats(self):
        backend = _make_backend()
        backend._db.event_stats = MagicMock(return_value={
            "buffer_utilization": 0.42,
            "match_rate": 0.95,
        })

        stats = await backend.get_event_stats()
        assert stats["buffer_utilization"] == 0.42

    @pytest.mark.asyncio
    async def test_get_event_stats_returns_empty_on_error(self):
        backend = _make_backend()
        backend._db.event_stats = MagicMock(side_effect=Exception("unavailable"))

        stats = await backend.get_event_stats()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_replay_mutations(self):
        backend = _make_backend()
        mock_events = [
            {"event_type": "ingest.document.accepted", "sequence": 1},
            {"event_type": "triple.inserted", "sequence": 2},
        ]
        backend._db.replay_events = MagicMock(return_value=mock_events)

        events = await backend.replay_mutations(limit=10)
        assert len(events) == 2
        backend._db.replay_events.assert_called_once()

    def test_reflection_worker_has_event_support(self):
        from silicon_memory.server.workers import ReflectionWorker
        from silicon_memory.server.config import ServerConfig

        config = ServerConfig()
        pool = MagicMock()
        worker = ReflectionWorker(pool, config)

        assert hasattr(worker, "event_stream_active")
        assert worker.event_stream_active is False

    def test_status_response_includes_event_stream(self):
        from silicon_memory.server.schemas import StatusResponse
        resp = StatusResponse(
            status="ok",
            version="0.1.0",
            uptime_seconds=10.0,
            active_users=1,
            mode="full",
            event_stream={"buffer_utilization": 0.5},
        )
        assert resp.event_stream is not None
        assert resp.event_stream["buffer_utilization"] == 0.5


# ========== Phase 4: RAPTOR ==========


class TestRaptorRetrieval:
    """Test RAPTOR hierarchical retrieval."""

    @pytest.mark.asyncio
    async def test_build_raptor_tree(self):
        backend = _make_backend()
        backend._db.build_raptor_tree = MagicMock(return_value={
            "status": "built",
            "levels": 3,
            "clusters": 15,
        })

        result = await backend.build_raptor_tree(cluster_size=10, max_levels=5)
        assert result["status"] == "built"
        backend._db.build_raptor_tree.assert_called_once_with(
            cluster_size=10, max_levels=5,
        )

    @pytest.mark.asyncio
    async def test_search_raptor_hybrid(self):
        backend = _make_backend()
        mock_results = [
            MagicMock(
                text="summary of cluster",
                score=0.95,
                level=1,
                metadata={"owner_id": "user1", "tenant_id": "tenant1"},
                external_id="tenant1/user1/belief-123",
            ),
            MagicMock(
                text="leaf document",
                score=0.88,
                level=0,
                metadata={"owner_id": "user1", "tenant_id": "tenant1"},
                external_id="tenant1/user1/belief-456",
            ),
        ]
        backend._db.search_raptor_hybrid = MagicMock(return_value=mock_results)

        results = await backend.search_raptor_hybrid("test query", k=10, tree_boost=0.3)
        assert len(results) == 2
        assert results[0].memory_type == "raptor_L1"
        assert results[1].memory_type == "semantic"

    @pytest.mark.asyncio
    async def test_search_raptor_returns_empty_on_error(self):
        backend = _make_backend()
        backend._db.search_raptor_hybrid = MagicMock(side_effect=Exception("no tree"))

        results = await backend.search_raptor_hybrid("test", k=10)
        assert results == []

    def test_deep_recall_profile_exists(self):
        assert "deep_recall" in PROFILES
        profile = PROFILES["deep_recall"]
        assert profile.tree_boost > 0
        assert profile.graph_proximity_weight > 0

    def test_salience_profile_tree_boost_in_weights(self):
        profile = SalienceProfile(tree_boost=0.3)
        weights = profile.to_search_weights()
        assert weights["tree_boost"] == 0.3

    def test_salience_profile_tree_boost_zero_excluded(self):
        profile = SalienceProfile(tree_boost=0.0)
        weights = profile.to_search_weights()
        assert "tree_boost" not in weights

    def test_salience_total_weight_includes_tree_boost(self):
        profile = SalienceProfile(tree_boost=0.3)
        assert profile.total_weight > 1.0  # Default sum + 0.3


# ========== Phase 5: Predicate Management ==========


class TestPredicateManagement:
    """Test predicate alias registration."""

    @pytest.mark.asyncio
    async def test_consolidate_registers_aliases(self):
        from silicon_memory.reflection.predicate_consolidator import PredicateConsolidator

        mock_memory = MagicMock()
        mock_db = MagicMock()
        mock_memory._backend._db = mock_db

        # 70 predicates > threshold of 60
        predicates = [f"pred_{i}" for i in range(70)]
        mock_db.all_predicates = MagicMock(return_value=predicates)
        mock_db.query_triples = MagicMock(return_value=[])

        # Simulate rewrite
        mock_rw = MagicMock()
        mock_rw.rewritten_count = 5
        mock_rw.merged_count = 3
        mock_rw.surviving_count = 67
        mock_db.rewrite_predicates = MagicMock(return_value=mock_rw)
        mock_db.register_predicate_aliases = MagicMock()

        consolidator = PredicateConsolidator(mock_memory, MagicMock(), threshold=60)

        # Mock LLM to return mappings
        consolidator._llm_evaluate = AsyncMock(return_value={
            "employed_by": "works_at",
            "works_for": "works_at",
        })

        result = await consolidator.consolidate_predicates()

        # register_predicate_aliases should be called with grouped aliases
        mock_db.register_predicate_aliases.assert_called_once()
        call_args = mock_db.register_predicate_aliases.call_args[0][0]
        assert "works_at" in call_args

        # rewrite_predicates should be called with the mapping
        mock_db.rewrite_predicates.assert_called_once()


# ========== Feature Flag Backward Compatibility ==========


class TestFeatureFlagCompat:
    """Test that disabling flags preserves old behavior."""

    def test_default_config_has_flags_enabled(self):
        cfg = _make_config()
        assert cfg.use_evidence_links is True
        assert cfg.use_native_entropy_rerank is True
        assert cfg.use_native_temporal_decay is True
        assert cfg.skip_belief_surface_writes is True
        assert cfg.use_consistency_waits is True
        assert cfg.use_event_stream is True
        assert cfg.enable_raptor is False

    @pytest.mark.asyncio
    async def test_legacy_mode_uses_shadow_sets(self):
        cfg = _make_config(use_consistency_waits=False)
        backend = _make_backend(cfg)

        from silicon_memory.core.types import Experience
        exp = Experience(
            id=uuid4(),
            content="test",
            occurred_at=datetime.now(timezone.utc),
        )

        backend._db.ingest = MagicMock(return_value=None)

        await backend.record_experience(exp)

        # Should populate shadow set in legacy mode
        assert len(backend._experience_external_ids) == 1

    @pytest.mark.asyncio
    async def test_legacy_mode_polls_for_visibility(self):
        cfg = _make_config(use_consistency_waits=False)
        backend = _make_backend(cfg)

        # Make search return the expected experience
        exp_id = uuid4()
        ext_id = backend._build_external_id("experience", exp_id)

        mock_result = MagicMock()
        mock_result.external_id = ext_id
        mock_result.node_type = "experience"
        mock_result.metadata = {}

        backend._search_experiences_broad = MagicMock(return_value=[mock_result])

        result = await backend.wait_for_experience_visibility(
            [exp_id], timeout_s=0.1, poll_interval_s=0.01,
        )
        # Should have found it via polling
        assert result is True
