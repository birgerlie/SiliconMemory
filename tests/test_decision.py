"""Tests for SM-1: Decision Records."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from silicon_memory.core.decision import (
    Alternative,
    Assumption,
    Decision,
    DecisionStatus,
)
from silicon_memory.core.utils import utc_now


# ============================================================================
# Unit tests: Decision dataclass
# ============================================================================


class TestDecisionDataclass:
    """Test Decision dataclass creation and serialization."""

    def test_decision_creation_defaults(self):
        """Test Decision with default values."""
        d = Decision(title="Use PostgreSQL", description="Chose Postgres for main DB")
        assert d.title == "Use PostgreSQL"
        assert d.status == DecisionStatus.ACTIVE
        assert d.outcome is None
        assert d.revision_of is None
        assert d.node_type == "decision"
        assert isinstance(d.id, UUID)

    def test_decision_with_assumptions(self):
        """Test Decision with assumptions."""
        belief_id = uuid4()
        assumption = Assumption(
            belief_id=belief_id,
            description="Team is familiar with SQL",
            confidence_at_decision=0.8,
            is_critical=True,
        )
        d = Decision(
            title="Use PostgreSQL",
            description="Chose Postgres for main DB",
            assumptions=[assumption],
        )
        assert len(d.assumptions) == 1
        assert d.assumptions[0].is_critical
        assert d.assumptions[0].confidence_at_decision == 0.8

    def test_decision_with_alternatives(self):
        """Test Decision with alternatives."""
        alt = Alternative(
            title="MongoDB",
            description="NoSQL document store",
            rejection_reason="Team lacks NoSQL experience",
        )
        d = Decision(
            title="Use PostgreSQL",
            alternatives=[alt],
        )
        assert len(d.alternatives) == 1
        assert d.alternatives[0].rejection_reason == "Team lacks NoSQL experience"

    def test_decision_serialization_roundtrip(self):
        """Test to_dict() / from_dict() roundtrip."""
        belief_id = uuid4()
        d = Decision(
            title="Use PostgreSQL",
            description="Chose Postgres for main DB",
            decided_by="user-123",
            session_id="sess-abc",
            belief_snapshot_id="snap-001",
            assumptions=[
                Assumption(
                    belief_id=belief_id,
                    description="Team knows SQL",
                    confidence_at_decision=0.85,
                    is_critical=True,
                )
            ],
            alternatives=[
                Alternative(
                    title="MongoDB",
                    description="NoSQL option",
                    rejection_reason="No experience",
                    beliefs_supporting=[belief_id],
                )
            ],
            tags={"database", "architecture"},
            metadata={"priority": "high"},
        )

        data = d.to_dict()
        restored = Decision.from_dict(data)

        assert restored.title == d.title
        assert restored.description == d.description
        assert restored.decided_by == d.decided_by
        assert restored.session_id == d.session_id
        assert restored.belief_snapshot_id == d.belief_snapshot_id
        assert len(restored.assumptions) == 1
        assert restored.assumptions[0].belief_id == belief_id
        assert restored.assumptions[0].is_critical
        assert len(restored.alternatives) == 1
        assert restored.alternatives[0].title == "MongoDB"
        assert "database" in restored.tags

    def test_decision_status_transitions(self):
        """Test decision status transitions."""
        d = Decision(title="Test")
        assert d.status == DecisionStatus.ACTIVE

        d.status = DecisionStatus.REVISIT_SUGGESTED
        assert d.status == DecisionStatus.REVISIT_SUGGESTED

        d.status = DecisionStatus.REVISED
        assert d.status == DecisionStatus.REVISED

        d.status = DecisionStatus.SUPERSEDED
        assert d.status == DecisionStatus.SUPERSEDED


# ============================================================================
# Unit tests: Decision storage via router
# ============================================================================


class TestDecisionRouter:
    """Test decision methods on the router."""

    @pytest.mark.asyncio
    async def test_commit_decision_stores_with_snapshot(self):
        """Test that commit_decision creates belief snapshot and stores."""
        mock_backend = AsyncMock()
        mock_backend.snapshot_beliefs = AsyncMock(return_value={"snapshot_id": "snap-001"})
        mock_backend.commit_decision = AsyncMock()

        belief_id = uuid4()
        decision = Decision(
            title="Use PostgreSQL",
            assumptions=[
                Assumption(
                    belief_id=belief_id,
                    description="Team knows SQL",
                    confidence_at_decision=0.85,
                    is_critical=True,
                )
            ],
        )

        # Create a minimal mock of SiliconMemory
        from silicon_memory.memory.silicondb_router import SiliconMemory

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory.commit_decision = SiliconMemory.commit_decision.__get__(memory, SiliconMemory)

        snapshot_id = await memory.commit_decision(decision)

        assert snapshot_id == "snap-001"
        assert decision.belief_snapshot_id == "snap-001"
        mock_backend.snapshot_beliefs.assert_awaited_once_with([str(belief_id)])
        mock_backend.commit_decision.assert_awaited_once_with(decision)

    @pytest.mark.asyncio
    async def test_commit_decision_no_assumptions_no_snapshot(self):
        """Test that commit_decision with no assumptions skips snapshot."""
        mock_backend = AsyncMock()
        mock_backend.commit_decision = AsyncMock()

        decision = Decision(title="Simple decision")

        from silicon_memory.memory.silicondb_router import SiliconMemory

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory.commit_decision = SiliconMemory.commit_decision.__get__(memory, SiliconMemory)

        snapshot_id = await memory.commit_decision(decision)

        assert snapshot_id is None
        mock_backend.snapshot_beliefs.assert_not_awaited()
        mock_backend.commit_decision.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_decisions_returns_by_similarity(self):
        """Test that recall_decisions delegates to backend."""
        expected = [Decision(title="Found decision")]
        mock_backend = AsyncMock()
        mock_backend.recall_decisions = AsyncMock(return_value=expected)

        from silicon_memory.memory.silicondb_router import SiliconMemory

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory.recall_decisions = SiliconMemory.recall_decisions.__get__(memory, SiliconMemory)

        results = await memory.recall_decisions("database", k=5)

        assert len(results) == 1
        assert results[0].title == "Found decision"

    @pytest.mark.asyncio
    async def test_record_outcome(self):
        """Test recording a decision outcome."""
        mock_backend = AsyncMock()
        mock_backend.record_decision_outcome = AsyncMock(return_value=True)

        from silicon_memory.memory.silicondb_router import SiliconMemory

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory.record_outcome = SiliconMemory.record_outcome.__get__(memory, SiliconMemory)

        decision_id = uuid4()
        result = await memory.record_outcome(decision_id, "Worked well")

        assert result is True
        mock_backend.record_decision_outcome.assert_awaited_once_with(decision_id, "Worked well")

    @pytest.mark.asyncio
    async def test_revise_decision_lifecycle(self):
        """Test full revision lifecycle: create → revise → original superseded."""
        original_id = uuid4()
        new_decision = Decision(title="Use CockroachDB instead")

        mock_backend = AsyncMock()
        mock_backend.revise_decision = AsyncMock(return_value=new_decision)

        from silicon_memory.memory.silicondb_router import SiliconMemory

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory.revise_decision = SiliconMemory.revise_decision.__get__(memory, SiliconMemory)

        result = await memory.revise_decision(original_id, new_decision)

        assert result is not None
        assert result.title == "Use CockroachDB instead"
        mock_backend.revise_decision.assert_awaited_once_with(original_id, new_decision)


# ============================================================================
# Unit tests: Reflection flags changed assumptions
# ============================================================================


class TestReflectionDecisionReview:
    """Test that reflection reviews active decisions for assumption drift."""

    @pytest.mark.asyncio
    async def test_reflection_flags_drifted_assumption(self):
        """Test that reflection flags decisions with drifted critical assumptions."""
        from silicon_memory.core.types import Belief

        belief_id = uuid4()
        decision = Decision(
            title="Use PostgreSQL",
            status=DecisionStatus.ACTIVE,
            assumptions=[
                Assumption(
                    belief_id=belief_id,
                    description="Team knows SQL",
                    confidence_at_decision=0.9,
                    is_critical=True,
                )
            ],
        )

        # Current belief confidence dropped significantly
        current_belief = Belief(
            id=belief_id,
            content="Team knows SQL",
            confidence=0.4,  # Dropped from 0.9 to 0.4 (drift = 0.5)
        )

        mock_memory = AsyncMock()
        mock_memory.recall_decisions = AsyncMock(return_value=[decision])
        mock_memory.get_belief = AsyncMock(return_value=current_belief)
        mock_memory._backend = AsyncMock()
        mock_memory._backend.record_decision_outcome = AsyncMock(return_value=True)

        from silicon_memory.reflection.engine import ReflectionEngine
        from silicon_memory.reflection.types import ReflectionConfig

        engine = ReflectionEngine.__new__(ReflectionEngine)
        engine._memory = mock_memory
        engine._config = ReflectionConfig()

        await engine._review_active_decisions()

        # Decision should have been flagged
        assert decision.status == DecisionStatus.REVISIT_SUGGESTED

    @pytest.mark.asyncio
    async def test_reflection_ignores_non_critical_drift(self):
        """Test that non-critical assumptions are ignored during review."""
        from silicon_memory.core.types import Belief

        belief_id = uuid4()
        decision = Decision(
            title="Use PostgreSQL",
            status=DecisionStatus.ACTIVE,
            assumptions=[
                Assumption(
                    belief_id=belief_id,
                    description="Nice to have SQL",
                    confidence_at_decision=0.9,
                    is_critical=False,  # Not critical
                )
            ],
        )

        current_belief = Belief(id=belief_id, content="test", confidence=0.3)

        mock_memory = AsyncMock()
        mock_memory.recall_decisions = AsyncMock(return_value=[decision])
        mock_memory.get_belief = AsyncMock(return_value=current_belief)
        mock_memory._backend = AsyncMock()

        from silicon_memory.reflection.engine import ReflectionEngine
        from silicon_memory.reflection.types import ReflectionConfig

        engine = ReflectionEngine.__new__(ReflectionEngine)
        engine._memory = mock_memory
        engine._config = ReflectionConfig()

        await engine._review_active_decisions()

        # Should remain ACTIVE
        assert decision.status == DecisionStatus.ACTIVE
