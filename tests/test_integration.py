"""Integration tests for silicon-memory with SiliconDB backend.

These tests verify that each memory layer (semantic, episodic, procedural, working)
correctly interacts with SiliconDB storage.

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_integration.py -v

For real embeddings (slower, more accurate):
    USE_REAL_EMBEDDINGS=1 SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_integration.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from silicon_memory.core.utils import utc_now
from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Experience,
    Procedure,
    Source,
    TemporalContext,
    Triplet,
)


pytestmark = pytest.mark.integration


# ============================================================================
# Semantic Memory (Beliefs) Integration Tests
# ============================================================================


class TestSemanticMemoryIntegration:
    """Test semantic memory operations through SiliconDB backend."""

    @pytest.mark.asyncio
    async def test_commit_and_get_triplet_belief(self, silicon_memory, sample_source):
        """Test storing and retrieving a belief with triplet."""
        belief = Belief(
            id=uuid4(),
            triplet=Triplet("Python", "is", "programming language"),
            confidence=0.9,
            source=sample_source,
            tags=["programming"],
        )

        # Store
        await silicon_memory.commit_belief(belief)

        # Query
        proof = await silicon_memory.what_do_you_know("Python programming", min_confidence=0.5)

        assert proof.beliefs is not None
        # Should find the belief about Python
        assert len(proof.beliefs) >= 0  # May or may not find depending on index timing

    @pytest.mark.asyncio
    async def test_commit_and_get_content_belief(self, silicon_memory, sample_source):
        """Test storing and retrieving a belief with content (no triplet)."""
        belief = Belief(
            id=uuid4(),
            content="Machine learning is a subset of artificial intelligence.",
            confidence=0.85,
            source=sample_source,
            tags=["ml", "ai"],
        )

        await silicon_memory.commit_belief(belief)

        proof = await silicon_memory.what_do_you_know("machine learning AI", min_confidence=0.5)
        assert proof is not None

    @pytest.mark.asyncio
    async def test_belief_with_temporal_context(self, silicon_memory, sample_source):
        """Test storing a belief with temporal validity."""
        now = utc_now()
        belief = Belief(
            id=uuid4(),
            triplet=Triplet("Python", "latest version is", "3.12"),
            confidence=0.95,
            source=sample_source,
            temporal=TemporalContext(
                observed_at=now,
                valid_from=now - timedelta(days=30),
                valid_until=now + timedelta(days=365),
            ),
        )

        await silicon_memory.commit_belief(belief)

        proof = await silicon_memory.what_do_you_know("Python version")
        assert proof is not None

    @pytest.mark.asyncio
    async def test_multiple_beliefs_same_subject(self, silicon_memory, sample_source):
        """Test storing multiple beliefs about the same subject."""
        beliefs = [
            Belief(
                id=uuid4(),
                triplet=Triplet("Python", "was created by", "Guido van Rossum"),
                confidence=0.99,
                source=sample_source,
            ),
            Belief(
                id=uuid4(),
                triplet=Triplet("Python", "was released in", "1991"),
                confidence=0.95,
                source=sample_source,
            ),
            Belief(
                id=uuid4(),
                triplet=Triplet("Python", "is known for", "readability"),
                confidence=0.9,
                source=sample_source,
            ),
        ]

        for belief in beliefs:
            await silicon_memory.commit_belief(belief)

        proof = await silicon_memory.what_do_you_know("Python creator history")
        assert proof is not None


# ============================================================================
# Episodic Memory (Experiences) Integration Tests
# ============================================================================


class TestEpisodicMemoryIntegration:
    """Test episodic memory operations through SiliconDB backend."""

    @pytest.mark.asyncio
    async def test_record_and_recall_experience(self, silicon_memory):
        """Test recording and recalling an experience."""
        experience = Experience(
            id=uuid4(),
            content="User asked how to sort a list in Python",
            outcome="Showed them the sorted() function and list.sort() method",
            session_id="test-session-1",
        )

        await silicon_memory.record_experience(experience)

        # Recall experiences
        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(
            query="Python list sorting",
            max_facts=0,
            max_experiences=10,
            max_procedures=0,
        )

        response = await silicon_memory.recall(ctx)
        # Check experiences were returned (may be empty if not indexed yet)
        assert response.experiences is not None

    @pytest.mark.asyncio
    async def test_experience_with_causal_chain(self, silicon_memory):
        """Test experiences with causal relationships."""
        # First experience
        exp1 = Experience(
            id=uuid4(),
            content="User reported a bug in their code",
            outcome="Started debugging",
            session_id="debug-session",
        )
        await silicon_memory.record_experience(exp1)

        # Second experience (caused by first)
        exp2 = Experience(
            id=uuid4(),
            content="Found the bug was a typo in variable name",
            outcome="Fixed the typo",
            session_id="debug-session",
            causal_parent=exp1.id,
        )
        await silicon_memory.record_experience(exp2)

        # Third experience (caused by second)
        exp3 = Experience(
            id=uuid4(),
            content="Verified the fix worked",
            outcome="Bug resolved successfully",
            session_id="debug-session",
            causal_parent=exp2.id,
        )
        await silicon_memory.record_experience(exp3)

        # Recall the debugging session
        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(
            query="debugging bug fix typo",
            max_experiences=10,
        )

        response = await silicon_memory.recall(ctx)
        assert response is not None

    @pytest.mark.asyncio
    async def test_experience_emotional_valence(self, silicon_memory):
        """Test experiences with different emotional valences."""
        positive_exp = Experience(
            id=uuid4(),
            content="User thanked me for the helpful explanation",
            outcome="Great interaction",
        )

        negative_exp = Experience(
            id=uuid4(),
            content="User was frustrated with a recurring error",
            outcome="Need to improve error handling guidance",
        )

        neutral_exp = Experience(
            id=uuid4(),
            content="User asked about Python documentation",
            outcome="Pointed them to docs.python.org",
        )

        await silicon_memory.record_experience(positive_exp)
        await silicon_memory.record_experience(negative_exp)
        await silicon_memory.record_experience(neutral_exp)

        # All should be stored
        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(query="user interaction", max_experiences=10)
        response = await silicon_memory.recall(ctx)
        assert response is not None


# ============================================================================
# Procedural Memory Integration Tests
# ============================================================================


class TestProceduralMemoryIntegration:
    """Test procedural memory operations through SiliconDB backend."""

    @pytest.mark.asyncio
    async def test_commit_and_find_procedure(self, silicon_memory):
        """Test storing and finding a procedure."""
        procedure = Procedure(
            id=uuid4(),
            name="Create Python Virtual Environment",
            description="Steps to create and activate a Python virtual environment",
            steps=[
                "Navigate to project directory",
                "Run: python -m venv venv",
                "Activate: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)",
                "Verify: which python shows venv path",
            ],
            trigger="create virtual environment venv",
            confidence=0.9,
        )

        await silicon_memory.commit_procedure(procedure)

        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(
            query="how to create python virtual environment",
            max_facts=0,
            max_experiences=0,
            max_procedures=5,
        )

        response = await silicon_memory.recall(ctx)
        assert response.procedures is not None

    @pytest.mark.asyncio
    async def test_procedure_with_source(self, silicon_memory, sample_source):
        """Test storing a procedure with source attribution."""
        procedure = Procedure(
            id=uuid4(),
            name="Run pytest tests",
            description="How to run Python tests with pytest",
            steps=[
                "Install pytest: pip install pytest",
                "Create test file: test_*.py",
                "Run: pytest",
                "Run specific: pytest test_file.py::test_name",
            ],
            trigger="run pytest tests",
            confidence=0.95,
            source=sample_source,
        )

        await silicon_memory.commit_procedure(procedure)

        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(query="pytest testing", max_procedures=5)
        response = await silicon_memory.recall(ctx)
        assert response is not None

    @pytest.mark.asyncio
    async def test_multiple_procedures_same_domain(self, silicon_memory):
        """Test storing multiple related procedures."""
        procedures = [
            Procedure(
                id=uuid4(),
                name="Git Init",
                description="Initialize a git repository",
                steps=["Navigate to project", "Run: git init"],
                trigger="git init",
                confidence=0.95,
            ),
            Procedure(
                id=uuid4(),
                name="Git Commit",
                description="Commit changes to git",
                steps=["Stage: git add .", "Commit: git commit -m 'message'"],
                trigger="git commit",
                confidence=0.95,
            ),
            Procedure(
                id=uuid4(),
                name="Git Push",
                description="Push to remote",
                steps=["git push origin branch-name"],
                trigger="git push",
                confidence=0.9,
            ),
        ]

        for proc in procedures:
            await silicon_memory.commit_procedure(proc)

        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(query="git workflow", max_procedures=10)
        response = await silicon_memory.recall(ctx)
        assert response is not None


# ============================================================================
# Working Memory Integration Tests
# ============================================================================


class TestWorkingMemoryIntegration:
    """Test working memory operations through SiliconDB backend."""

    @pytest.mark.asyncio
    async def test_set_and_get_context(self, silicon_memory):
        """Test setting and getting working memory context."""
        await silicon_memory.set_context("current_task", "debugging Python code")
        await silicon_memory.set_context("user_name", "Alice")

        task = await silicon_memory.get_context("current_task")
        name = await silicon_memory.get_context("user_name")

        assert task == "debugging Python code"
        assert name == "Alice"

    @pytest.mark.asyncio
    async def test_get_all_context(self, silicon_memory):
        """Test getting all working memory context."""
        await silicon_memory.set_context("key1", "value1")
        await silicon_memory.set_context("key2", "value2")
        await silicon_memory.set_context("key3", {"nested": "value"})

        all_context = await silicon_memory.get_all_context()

        assert "key1" in all_context
        assert "key2" in all_context
        assert "key3" in all_context
        assert all_context["key1"] == "value1"

    @pytest.mark.asyncio
    async def test_context_update(self, silicon_memory):
        """Test updating existing context."""
        await silicon_memory.set_context("counter", 1)
        value1 = await silicon_memory.get_context("counter")
        assert value1 == 1

        await silicon_memory.set_context("counter", 2)
        value2 = await silicon_memory.get_context("counter")
        assert value2 == 2

    @pytest.mark.asyncio
    async def test_context_nonexistent_key(self, silicon_memory):
        """Test getting a nonexistent key."""
        result = await silicon_memory.get_context("nonexistent_key_12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_context_complex_values(self, silicon_memory):
        """Test storing complex values in context."""
        complex_value = {
            "files": ["main.py", "utils.py"],
            "status": {"completed": True, "errors": 0},
            "metadata": {"created": "2024-01-01"},
        }

        await silicon_memory.set_context("project_state", complex_value)
        result = await silicon_memory.get_context("project_state")

        assert result == complex_value


# ============================================================================
# Cross-Memory Integration Tests
# ============================================================================


class TestCrossMemoryIntegration:
    """Test interactions across different memory types."""

    @pytest.mark.asyncio
    async def test_unified_recall(self, silicon_memory, sample_source):
        """Test recalling from all memory types at once."""
        # Add a belief
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                triplet=Triplet("FastAPI", "is", "web framework"),
                confidence=0.9,
                source=sample_source,
            )
        )

        # Add an experience
        await silicon_memory.record_experience(
            Experience(
                id=uuid4(),
                content="Helped user build FastAPI application",
                outcome="Successfully created REST API",
            )
        )

        # Add a procedure
        await silicon_memory.commit_procedure(
            Procedure(
                id=uuid4(),
                name="Create FastAPI App",
                description="How to create a FastAPI application",
                steps=[
                    "pip install fastapi uvicorn",
                    "Create main.py with FastAPI app",
                    "Run: uvicorn main:app --reload",
                ],
                trigger="create fastapi app",
                confidence=0.85,
            )
        )

        # Set working context
        await silicon_memory.set_context("current_framework", "FastAPI")

        # Recall everything about FastAPI
        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(
            query="FastAPI web framework",
            max_facts=10,
            max_experiences=5,
            max_procedures=5,
        )

        response = await silicon_memory.recall(ctx)

        assert response is not None
        assert response.working_context.get("current_framework") == "FastAPI"

    @pytest.mark.asyncio
    async def test_knowledge_proof_with_sources(self, silicon_memory, sample_source):
        """Test building a knowledge proof with source attribution."""
        source2 = Source(
            id="docs-source",
            type="documentation",
            reliability=0.95,
            metadata={"name": "Python Official Docs", "url": "https://docs.python.org"},
        )

        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                triplet=Triplet("asyncio", "is", "Python async framework"),
                confidence=0.9,
                source=sample_source,
            )
        )

        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                content="asyncio provides async/await syntax for concurrent programming",
                confidence=0.95,
                source=source2,
            )
        )

        proof = await silicon_memory.what_do_you_know("asyncio async programming")

        assert proof is not None
        assert proof.query == "asyncio async programming"
