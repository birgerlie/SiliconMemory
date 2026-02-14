"""Tests for SM-3: Salience-Weighted Retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import types

import pytest

from silicon_memory.retrieval.salience import (
    PROFILES,
    SalienceProfile,
)
from silicon_memory.memory.silicondb_router import RecallContext


# ============================================================================
# Unit tests: SalienceProfile
# ============================================================================


class TestSalienceProfile:
    """Test SalienceProfile dataclass."""

    def test_default_profile(self):
        """Test default SalienceProfile values."""
        profile = SalienceProfile()
        assert profile.vector_weight == 0.3
        assert profile.text_weight == 0.1
        assert profile.temporal_weight == 0.15
        assert profile.temporal_half_life_days == 30.0
        assert profile.confidence_weight == 0.2
        assert profile.graph_proximity_weight == 0.15
        assert profile.entropy_weight == 0.1
        assert profile.entropy_direction == "prefer_low"

    def test_custom_profile(self):
        """Test custom SalienceProfile values."""
        profile = SalienceProfile(
            vector_weight=0.5,
            text_weight=0.3,
            temporal_weight=0.1,
            confidence_weight=0.05,
            graph_proximity_weight=0.05,
            entropy_weight=0.0,
            entropy_direction="prefer_high",
        )
        assert profile.vector_weight == 0.5
        assert profile.entropy_direction == "prefer_high"

    def test_invalid_entropy_direction(self):
        """Test that invalid entropy direction raises error."""
        with pytest.raises(ValueError, match="entropy_direction"):
            SalienceProfile(entropy_direction="invalid")

    def test_total_weight(self):
        """Test total_weight property."""
        profile = SalienceProfile()
        assert profile.total_weight == pytest.approx(1.0)

    def test_to_search_weights(self):
        """Test conversion to search weights dict."""
        profile = SalienceProfile(
            vector_weight=0.3,
            text_weight=0.1,
            temporal_weight=0.15,
            temporal_half_life_days=30,
            confidence_weight=0.2,
            graph_proximity_weight=0.15,
            entropy_weight=0.1,
            entropy_direction="prefer_low",
        )
        weights = profile.to_search_weights()
        assert weights["vector"] == 0.3
        assert weights["text"] == 0.1
        assert weights["temporal"] == 0.15
        assert weights["temporal_half_life_hours"] == 30 * 24
        assert weights["confidence"] == 0.2
        assert weights["graph_proximity"] == 0.15
        assert weights["entropy_weight"] == 0.1
        assert weights["entropy_direction"] == "prefer_low"


# ============================================================================
# Unit tests: Preset profiles
# ============================================================================


class TestPresetProfiles:
    """Test preset salience profiles."""

    def test_profiles_exist(self):
        """Test that all expected profiles exist."""
        assert "decision_support" in PROFILES
        assert "exploration" in PROFILES
        assert "context_recall" in PROFILES

    def test_decision_support_profile(self):
        """Test decision_support profile prioritizes confidence."""
        profile = PROFILES["decision_support"]
        assert profile.confidence_weight >= 0.2
        assert profile.entropy_direction == "prefer_low"

    def test_exploration_profile(self):
        """Test exploration profile prioritizes diversity and novelty."""
        profile = PROFILES["exploration"]
        assert profile.vector_weight >= 0.3
        assert profile.entropy_weight >= 0.1
        assert profile.entropy_direction == "prefer_high"
        assert profile.temporal_half_life_days >= 100

    def test_context_recall_profile(self):
        """Test context_recall profile prioritizes recency and proximity."""
        profile = PROFILES["context_recall"]
        assert profile.temporal_weight >= 0.2
        assert profile.graph_proximity_weight >= 0.2
        assert profile.temporal_half_life_days <= 14

    def test_different_profiles_produce_different_configs(self):
        """Test that profiles produce distinct search configurations."""
        decision_weights = PROFILES["decision_support"].to_search_weights()
        explore_weights = PROFILES["exploration"].to_search_weights()
        context_weights = PROFILES["context_recall"].to_search_weights()

        # They should differ in at least some weights
        assert decision_weights != explore_weights
        assert explore_weights != context_weights
        assert decision_weights != context_weights

    def test_all_profiles_are_valid(self):
        """Test that all profiles have valid entropy direction."""
        for name, profile in PROFILES.items():
            assert profile.entropy_direction in ("prefer_low", "prefer_high"), (
                f"Profile '{name}' has invalid entropy_direction"
            )

    def test_all_profiles_weights_roughly_sum_to_one(self):
        """Test that all profile weights sum to approximately 1.0."""
        for name, profile in PROFILES.items():
            total = profile.total_weight
            assert 0.8 <= total <= 1.2, (
                f"Profile '{name}' weights sum to {total}"
            )


# ============================================================================
# Unit tests: RecallContext with salience_profile
# ============================================================================


class TestRecallContextSalience:
    """Test RecallContext with salience_profile."""

    def test_default_no_profile(self):
        """Test that RecallContext defaults to no salience profile."""
        ctx = RecallContext(query="test")
        assert ctx.salience_profile is None

    def test_string_profile(self):
        """Test RecallContext with string profile name."""
        ctx = RecallContext(query="test", salience_profile="decision_support")
        assert ctx.salience_profile == "decision_support"

    def test_object_profile(self):
        """Test RecallContext with SalienceProfile object."""
        profile = SalienceProfile(vector_weight=0.5)
        ctx = RecallContext(query="test", salience_profile=profile)
        assert isinstance(ctx.salience_profile, SalienceProfile)
        assert ctx.salience_profile.vector_weight == 0.5


# ============================================================================
# Integration tests: Router recall with salience
# ============================================================================


class TestRecallWithSalience:
    """Test that recall() properly handles salience profiles."""

    def _make_mock_memory(self):
        """Create a mock SiliconMemory for testing recall."""
        from silicon_memory.memory.silicondb_router import SiliconMemory

        mock_backend = AsyncMock()
        mock_backend.recall = AsyncMock(return_value={
            "facts": [],
            "experiences": [],
            "procedures": [],
            "working_context": {},
            "total_items": 0,
            "query": "test",
            "as_of": "2025-01-01T00:00:00",
        })
        mock_backend.get_working = AsyncMock(return_value=None)
        mock_backend.query_beliefs = AsyncMock(return_value=[])

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")
        memory.recall = types.MethodType(SiliconMemory.recall, memory)
        memory._resolve_context_seeds = types.MethodType(
            SiliconMemory._resolve_context_seeds, memory,
        )
        return memory

    async def test_recall_without_profile_unchanged(self):
        """Test that recall without profile works as before (backward compatible)."""
        memory = self._make_mock_memory()

        ctx = RecallContext(query="Python")
        await memory.recall(ctx)

        call_kwargs = memory._backend.recall.call_args[1]
        assert "search_weights" not in call_kwargs
        assert call_kwargs["query"] == "Python"

    async def test_recall_with_string_profile(self):
        """Test recall with a string profile name resolves to weights."""
        memory = self._make_mock_memory()

        ctx = RecallContext(query="test", salience_profile="decision_support")
        await memory.recall(ctx)

        call_kwargs = memory._backend.recall.call_args[1]
        assert "search_weights" in call_kwargs
        weights = call_kwargs["search_weights"]
        expected = PROFILES["decision_support"].to_search_weights()
        assert weights == expected

    async def test_recall_with_profile_object(self):
        """Test recall with a SalienceProfile object."""
        memory = self._make_mock_memory()

        custom = SalienceProfile(vector_weight=0.9, text_weight=0.1)
        ctx = RecallContext(query="test", salience_profile=custom)
        await memory.recall(ctx)

        call_kwargs = memory._backend.recall.call_args[1]
        assert "search_weights" in call_kwargs
        assert call_kwargs["search_weights"]["vector"] == 0.9

    async def test_recall_with_unknown_string_profile(self):
        """Test that unknown profile name falls back to no weights."""
        memory = self._make_mock_memory()

        ctx = RecallContext(query="test", salience_profile="nonexistent")
        await memory.recall(ctx)

        call_kwargs = memory._backend.recall.call_args[1]
        assert "search_weights" not in call_kwargs

    async def test_different_profiles_produce_different_weights(self):
        """Test that different profiles result in different backend calls."""
        for profile_name in ("decision_support", "exploration", "context_recall"):
            memory = self._make_mock_memory()
            ctx = RecallContext(query="test", salience_profile=profile_name)
            await memory.recall(ctx)

            call_kwargs = memory._backend.recall.call_args[1]
            weights = call_kwargs["search_weights"]
            expected = PROFILES[profile_name].to_search_weights()
            assert weights == expected, f"Profile {profile_name} mismatch"


# ============================================================================
# Integration tests: Backend search_weights wiring (SDB-1)
# ============================================================================


class TestBackendSearchWeightsWiring:
    """Verify that backend query methods receive and use search_weights."""

    async def test_recall_passes_search_weights_to_query_methods(self):
        """Backend.recall() should forward search_weights to all query methods."""
        from silicon_memory.storage.silicondb_backend import SiliconDBBackend

        backend = MagicMock(spec=SiliconDBBackend)
        backend._query_beliefs_with_entropy = AsyncMock(return_value=[])
        backend.query_experiences = AsyncMock(return_value=[])
        backend.find_applicable_procedures = AsyncMock(return_value=[])
        backend.get_all_working = AsyncMock(return_value={})
        backend._decay_config = MagicMock()
        backend.recall = types.MethodType(SiliconDBBackend.recall, backend)

        weights = {"vector": 0.4, "text": 0.2, "temporal": 0.1,
                    "confidence": 0.2, "graph_proximity": 0.1,
                    "temporal_half_life_hours": 720,
                    "entropy_weight": 0.0, "entropy_direction": "prefer_low"}

        await backend.recall("test query", search_weights=weights)

        # _query_beliefs_with_entropy should receive search_weights
        bkw = backend._query_beliefs_with_entropy.call_args[1]
        assert bkw["search_weights"] == weights

        # query_experiences should receive search_weights
        ekw = backend.query_experiences.call_args[1]
        assert ekw["search_weights"] == weights

        # find_applicable_procedures should receive search_weights
        pkw = backend.find_applicable_procedures.call_args[1]
        assert pkw["search_weights"] == weights


# ============================================================================
# Integration tests: Entropy reranking (SDB-2)
# ============================================================================


class TestEntropyReranking:
    """Test post-retrieval entropy reranking in recall()."""

    def test_entropy_reranking_prefer_high(self):
        """prefer_high should surface high-entropy (uncertain) results."""
        from silicon_memory.core.types import RecallResult
        from silicon_memory.storage.silicondb_backend import SiliconDBBackend

        results = [
            RecallResult(
                content="certain fact", confidence=0.9, source=None,
                memory_type="semantic", relevance_score=0.8, entropy=0.05,
            ),
            RecallResult(
                content="uncertain fact", confidence=0.5, source=None,
                memory_type="semantic", relevance_score=0.7, entropy=0.65,
            ),
        ]

        reranked = SiliconDBBackend._apply_entropy_reranking(
            results, entropy_weight=0.5, entropy_direction="prefer_high",
        )

        # The uncertain fact (high entropy) should now rank first
        assert reranked[0].content == "uncertain fact"

    def test_entropy_reranking_prefer_low(self):
        """prefer_low should surface low-entropy (confident) results."""
        from silicon_memory.core.types import RecallResult
        from silicon_memory.storage.silicondb_backend import SiliconDBBackend

        results = [
            RecallResult(
                content="uncertain fact", confidence=0.5, source=None,
                memory_type="semantic", relevance_score=0.8, entropy=0.65,
            ),
            RecallResult(
                content="certain fact", confidence=0.9, source=None,
                memory_type="semantic", relevance_score=0.7, entropy=0.05,
            ),
        ]

        reranked = SiliconDBBackend._apply_entropy_reranking(
            results, entropy_weight=0.5, entropy_direction="prefer_low",
        )

        # The certain fact (low entropy) should now rank first
        assert reranked[0].content == "certain fact"

    def test_entropy_reranking_zero_weight_noop(self):
        """Entropy weight of 0 should not change ordering."""
        from silicon_memory.core.types import RecallResult
        from silicon_memory.storage.silicondb_backend import SiliconDBBackend

        results = [
            RecallResult(
                content="a", confidence=0.9, source=None,
                memory_type="semantic", relevance_score=0.8, entropy=0.65,
            ),
            RecallResult(
                content="b", confidence=0.5, source=None,
                memory_type="semantic", relevance_score=0.3, entropy=0.05,
            ),
        ]
        original_scores = [r.relevance_score for r in results]

        reranked = SiliconDBBackend._apply_entropy_reranking(
            results, entropy_weight=0.0, entropy_direction="prefer_low",
        )

        # Scores should be unchanged (0 weight → no adjustment)
        for r, orig in zip(reranked, original_scores):
            assert r.relevance_score == pytest.approx(orig)

    def test_entropy_in_to_search_weights(self):
        """SalienceProfile.to_search_weights() should include entropy config."""
        profile = SalienceProfile(entropy_weight=0.15, entropy_direction="prefer_high")
        weights = profile.to_search_weights()
        assert weights["entropy_weight"] == 0.15
        assert weights["entropy_direction"] == "prefer_high"


# ============================================================================
# Integration tests: Graph context nodes (SDB-3)
# ============================================================================


class TestGraphContextNodes:
    """Test graph_context_nodes support in recall pipeline."""

    def test_recall_context_has_graph_context_nodes(self):
        """RecallContext should accept graph_context_nodes."""
        ctx = RecallContext(
            query="test",
            graph_context_nodes=["node-a", "node-b"],
        )
        assert ctx.graph_context_nodes == ["node-a", "node-b"]

    def test_recall_context_defaults_to_none(self):
        """graph_context_nodes should default to None."""
        ctx = RecallContext(query="test")
        assert ctx.graph_context_nodes is None

    async def test_graph_context_nodes_passed_to_backend(self):
        """Explicit graph_context_nodes should flow into search_weights."""
        from silicon_memory.memory.silicondb_router import SiliconMemory

        mock_backend = AsyncMock()
        mock_backend.recall = AsyncMock(return_value={
            "facts": [], "experiences": [], "procedures": [],
            "working_context": {}, "total_items": 0,
            "query": "test", "as_of": "2025-01-01T00:00:00",
        })
        mock_backend.get_working = AsyncMock(return_value=None)
        mock_backend.query_beliefs = AsyncMock(return_value=[])

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")
        memory.recall = types.MethodType(SiliconMemory.recall, memory)
        memory._resolve_context_seeds = types.MethodType(
            SiliconMemory._resolve_context_seeds, memory,
        )

        # Use context_recall which has graph_proximity_weight=0.25
        ctx = RecallContext(
            query="test",
            salience_profile="context_recall",
            graph_context_nodes=["seed-1", "seed-2"],
        )
        await memory.recall(ctx)

        call_kwargs = mock_backend.recall.call_args[1]
        sw = call_kwargs["search_weights"]
        assert sw["graph_context_nodes"] == ["seed-1", "seed-2"]

    async def test_auto_resolve_seeds_when_none_provided(self):
        """When graph_proximity > 0 and no seeds given, auto-resolve from working memory."""
        from silicon_memory.memory.silicondb_router import SiliconMemory
        from silicon_memory.core.types import Belief, BeliefStatus
        from uuid import uuid4

        mock_belief = MagicMock(spec=Belief)
        mock_belief.id = uuid4()

        mock_backend = AsyncMock()
        mock_backend.recall = AsyncMock(return_value={
            "facts": [], "experiences": [], "procedures": [],
            "working_context": {}, "total_items": 0,
            "query": "test", "as_of": "2025-01-01T00:00:00",
        })
        mock_backend.get_working = AsyncMock(return_value="machine learning")
        mock_backend.query_beliefs = AsyncMock(return_value=[mock_belief])
        mock_backend._build_external_id = MagicMock(
            return_value=f"t1/u1/belief-{mock_belief.id}",
        )

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")
        memory.recall = types.MethodType(SiliconMemory.recall, memory)
        memory._resolve_context_seeds = types.MethodType(
            SiliconMemory._resolve_context_seeds, memory,
        )

        ctx = RecallContext(
            query="test",
            salience_profile="context_recall",  # graph_proximity=0.25
        )
        await memory.recall(ctx)

        call_kwargs = mock_backend.recall.call_args[1]
        sw = call_kwargs["search_weights"]
        assert "graph_context_nodes" in sw
        assert len(sw["graph_context_nodes"]) == 1
        assert f"belief-{mock_belief.id}" in sw["graph_context_nodes"][0]
