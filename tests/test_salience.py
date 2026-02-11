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
        )
        weights = profile.to_search_weights()
        assert weights["vector"] == 0.3
        assert weights["text"] == 0.1
        assert weights["temporal"] == 0.15
        assert weights["temporal_half_life_hours"] == 30 * 24
        assert weights["confidence"] == 0.2
        assert weights["graph_proximity"] == 0.15


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

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")
        memory.recall = types.MethodType(SiliconMemory.recall, memory)
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
