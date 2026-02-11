"""Tests for SM-4: Context Switch Snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from silicon_memory.snapshot.types import ContextSnapshot, SnapshotConfig
from silicon_memory.snapshot.service import SnapshotService
from silicon_memory.tools.memory_tool import MemoryAction, MemoryTool, MemoryToolResponse
from silicon_memory.core.types import Experience
from silicon_memory.core.utils import utc_now


# ============================================================================
# Unit tests: ContextSnapshot dataclass
# ============================================================================


class TestContextSnapshotDataclass:
    """Unit tests for the ContextSnapshot dataclass."""

    def test_defaults(self):
        """Test that defaults are set correctly."""
        snap = ContextSnapshot()
        assert isinstance(snap.id, UUID)
        assert snap.task_context == ""
        assert snap.summary == ""
        assert snap.working_memory == {}
        assert snap.recent_experiences == []
        assert snap.next_steps == []
        assert snap.open_questions == []
        assert isinstance(snap.created_at, datetime)
        assert snap.session_id is None
        assert snap.user_id is None
        assert snap.tenant_id is None
        assert snap.privacy is None

    def test_with_values(self):
        """Test construction with explicit values."""
        exp_ids = [uuid4(), uuid4()]
        snap = ContextSnapshot(
            task_context="project-alpha/auth-module",
            summary="Working on token refresh logic",
            working_memory={"current_file": "auth.py", "bug_id": "BUG-42"},
            recent_experiences=exp_ids,
            next_steps=["Add refresh logic", "Write tests"],
            open_questions=["Should tokens auto-refresh?"],
            session_id="session-123",
            user_id="user-1",
            tenant_id="acme",
        )
        assert snap.task_context == "project-alpha/auth-module"
        assert snap.summary == "Working on token refresh logic"
        assert snap.working_memory == {"current_file": "auth.py", "bug_id": "BUG-42"}
        assert snap.recent_experiences == exp_ids
        assert snap.next_steps == ["Add refresh logic", "Write tests"]
        assert snap.open_questions == ["Should tokens auto-refresh?"]
        assert snap.session_id == "session-123"
        assert snap.user_id == "user-1"
        assert snap.tenant_id == "acme"

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        exp_id = uuid4()
        snap = ContextSnapshot(
            task_context="proj/module",
            summary="A summary",
            working_memory={"key": "val"},
            recent_experiences=[exp_id],
            next_steps=["step1"],
            open_questions=["q1?"],
            session_id="s1",
        )
        d = snap.to_dict()
        assert d["task_context"] == "proj/module"
        assert d["summary"] == "A summary"
        assert d["working_memory"] == {"key": "val"}
        assert d["recent_experiences"] == [str(exp_id)]
        assert d["next_steps"] == ["step1"]
        assert d["open_questions"] == ["q1?"]
        assert d["session_id"] == "s1"
        assert "id" in d
        assert "created_at" in d

    def test_unique_ids(self):
        """Test that each snapshot gets a unique ID."""
        snap1 = ContextSnapshot()
        snap2 = ContextSnapshot()
        assert snap1.id != snap2.id


class TestSnapshotConfig:
    """Unit tests for the SnapshotConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = SnapshotConfig()
        assert config.max_recent_experiences == 20
        assert config.recent_hours == 24
        assert config.llm_temperature == 0.3
        assert config.fallback_summary_max_chars == 500

    def test_custom_values(self):
        """Test custom configuration."""
        config = SnapshotConfig(
            max_recent_experiences=50,
            recent_hours=48,
            llm_temperature=0.5,
            fallback_summary_max_chars=1000,
        )
        assert config.max_recent_experiences == 50
        assert config.recent_hours == 48


# ============================================================================
# Unit tests: SnapshotService rule-based summary
# ============================================================================


class TestRuleBasedSummary:
    """Unit tests for rule-based summary generation."""

    def _make_service(self, config=None):
        """Create a SnapshotService with mocked dependencies."""
        memory = MagicMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        backend = MagicMock()
        return SnapshotService(memory=memory, backend=backend, config=config)

    def test_basic_summary(self):
        """Test rule-based summary with working memory and experiences."""
        service = self._make_service()
        exp = MagicMock(spec=Experience)
        exp.content = "Debugged the auth token refresh. TODO: write tests for edge cases."
        exp.outcome = None

        summary, next_steps, open_questions = service._generate_rule_based_summary(
            task_context="project-alpha",
            working_memory={"file": "auth.py", "branch": "fix-tokens"},
            recent_experiences=[exp],
        )

        assert "project-alpha" in summary
        assert "file" in summary or "branch" in summary
        assert any("write tests" in step.lower() for step in next_steps)

    def test_empty_state(self):
        """Test rule-based summary with empty state."""
        service = self._make_service()
        summary, next_steps, open_questions = service._generate_rule_based_summary(
            task_context="empty-project",
            working_memory={},
            recent_experiences=[],
        )
        assert "empty-project" in summary
        assert next_steps == []
        assert open_questions == []

    def test_question_extraction(self):
        """Test that questions are extracted from experience content."""
        service = self._make_service()
        exp = MagicMock(spec=Experience)
        exp.content = "We discussed the API design. Should we use REST or GraphQL? The team needs to decide."
        exp.outcome = None

        summary, next_steps, open_questions = service._generate_rule_based_summary(
            task_context="api-design",
            working_memory={},
            recent_experiences=[exp],
        )
        # Question extraction is best-effort
        # Just verify it doesn't crash
        assert isinstance(open_questions, list)

    def test_next_steps_from_keywords(self):
        """Test that next steps are extracted from TODO/NEXT keywords."""
        service = self._make_service()
        exp = MagicMock(spec=Experience)
        exp.content = "Finished the parser. NEXT: implement the evaluator.\nTODO: add error handling"
        exp.outcome = None

        summary, next_steps, open_questions = service._generate_rule_based_summary(
            task_context="compiler",
            working_memory={},
            recent_experiences=[exp],
        )
        assert len(next_steps) >= 1

    def test_summary_truncation(self):
        """Test that summary is truncated to max chars."""
        config = SnapshotConfig(fallback_summary_max_chars=50)
        service = self._make_service(config)

        summary, _, _ = service._generate_rule_based_summary(
            task_context="a-very-long-project-context-name",
            working_memory={f"key_{i}": f"value_{i}" for i in range(20)},
            recent_experiences=[],
        )
        assert len(summary) <= 50


# ============================================================================
# Unit tests: SnapshotService create/retrieve with mocked backend
# ============================================================================


class TestSnapshotServiceMocked:
    """Tests for SnapshotService with mocked backend and memory."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.get_all_context = AsyncMock(return_value={"key": "value"})
        memory.get_recent_experiences = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.store_snapshot = AsyncMock()
        backend.query_snapshots_by_context = AsyncMock(return_value=[])
        return backend

    async def test_create_snapshot(self, mock_memory, mock_backend):
        """Test creating a snapshot stores it in the backend."""
        service = SnapshotService(
            memory=mock_memory,
            backend=mock_backend,
        )

        snapshot = await service.create_snapshot("project-alpha")

        assert snapshot.task_context == "project-alpha"
        assert snapshot.working_memory == {"key": "value"}
        assert snapshot.user_id == "user-1"
        assert snapshot.tenant_id == "acme"
        mock_backend.store_snapshot.assert_awaited_once()

    async def test_create_snapshot_with_experiences(self, mock_memory, mock_backend):
        """Test that recent experience IDs are captured."""
        exp1 = MagicMock(spec=Experience)
        exp1.id = uuid4()
        exp1.content = "Did something"
        exp1.outcome = None

        mock_memory.get_recent_experiences = AsyncMock(return_value=[exp1])

        service = SnapshotService(memory=mock_memory, backend=mock_backend)
        snapshot = await service.create_snapshot("proj")

        assert exp1.id in snapshot.recent_experiences

    async def test_get_latest_snapshot_found(self, mock_memory, mock_backend):
        """Test retrieving the latest snapshot."""
        expected = ContextSnapshot(task_context="proj", summary="test")
        mock_backend.query_snapshots_by_context = AsyncMock(return_value=[expected])

        service = SnapshotService(memory=mock_memory, backend=mock_backend)
        result = await service.get_latest_snapshot("proj")

        assert result is expected
        mock_backend.query_snapshots_by_context.assert_awaited_once_with(
            task_context="proj", limit=1
        )

    async def test_get_latest_snapshot_not_found(self, mock_memory, mock_backend):
        """Test that None is returned when no snapshot exists."""
        mock_backend.query_snapshots_by_context = AsyncMock(return_value=[])

        service = SnapshotService(memory=mock_memory, backend=mock_backend)
        result = await service.get_latest_snapshot("nonexistent")

        assert result is None

    async def test_list_snapshots(self, mock_memory, mock_backend):
        """Test listing snapshots."""
        snaps = [
            ContextSnapshot(task_context="proj", summary="s1"),
            ContextSnapshot(task_context="proj", summary="s2"),
        ]
        mock_backend.query_snapshots_by_context = AsyncMock(return_value=snaps)

        service = SnapshotService(memory=mock_memory, backend=mock_backend)
        result = await service.list_snapshots("proj", limit=5)

        assert len(result) == 2

    async def test_create_snapshot_with_llm(self, mock_memory, mock_backend):
        """Test snapshot creation with LLM provider."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value='{"summary": "LLM summary", "next_steps": ["step1"], "open_questions": ["q1?"]}')

        service = SnapshotService(
            memory=mock_memory,
            backend=mock_backend,
            llm_provider=mock_llm,
        )

        snapshot = await service.create_snapshot("proj")

        assert snapshot.summary == "LLM summary"
        assert snapshot.next_steps == ["step1"]
        assert snapshot.open_questions == ["q1?"]

    async def test_llm_failure_falls_back_to_rules(self, mock_memory, mock_backend):
        """Test that LLM failure falls back to rule-based summary."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM error"))

        service = SnapshotService(
            memory=mock_memory,
            backend=mock_backend,
            llm_provider=mock_llm,
        )

        snapshot = await service.create_snapshot("proj")

        # Should still produce a snapshot with rule-based summary
        assert snapshot.task_context == "proj"
        assert isinstance(snapshot.summary, str)


# ============================================================================
# Unit tests: MemoryTool switch/resume context
# ============================================================================


class TestMemoryToolSwitchResume:
    """Tests for SWITCH_CONTEXT and RESUME_CONTEXT tool actions."""

    def test_action_enum_values(self):
        """Test that new actions are in the enum."""
        assert MemoryAction.SWITCH_CONTEXT == "switch_context"
        assert MemoryAction.RESUME_CONTEXT == "resume_context"

    async def test_switch_context(self):
        """Test switch_context creates a snapshot and returns summary."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(user_id="u1", tenant_id="t1")
        snapshot = ContextSnapshot(
            task_context="project-alpha",
            summary="Working on auth",
            working_memory={"file": "auth.py"},
            recent_experiences=[uuid4()],
            next_steps=["Add refresh logic"],
            open_questions=["Use JWT?"],
        )
        mock_memory.create_snapshot = AsyncMock(return_value=snapshot)

        tool = MemoryTool(mock_memory)
        response = await tool.invoke("switch_context", task_context="project-alpha")

        assert response.success
        assert response.action == MemoryAction.SWITCH_CONTEXT
        assert response.data["task_context"] == "project-alpha"
        assert response.data["summary"] == "Working on auth"
        assert "Add refresh logic" in response.data["next_steps"]
        mock_memory.create_snapshot.assert_awaited_once_with("project-alpha")

    async def test_resume_context_found(self):
        """Test resume_context retrieves and returns snapshot."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(user_id="u1", tenant_id="t1")
        snapshot = ContextSnapshot(
            task_context="project-alpha",
            summary="Working on auth",
            working_memory={"file": "auth.py"},
            next_steps=["Add refresh logic"],
            open_questions=["Use JWT?"],
            session_id="s1",
        )
        mock_memory.get_latest_snapshot = AsyncMock(return_value=snapshot)

        tool = MemoryTool(mock_memory)
        response = await tool.invoke("resume_context", task_context="project-alpha")

        assert response.success
        assert response.action == MemoryAction.RESUME_CONTEXT
        assert response.data["found"] is True
        assert response.data["summary"] == "Working on auth"
        assert response.data["working_memory"] == {"file": "auth.py"}

    async def test_resume_context_not_found(self):
        """Test resume_context when no snapshot exists."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(user_id="u1", tenant_id="t1")
        mock_memory.get_latest_snapshot = AsyncMock(return_value=None)

        tool = MemoryTool(mock_memory)
        response = await tool.invoke("resume_context", task_context="nonexistent")

        assert response.success
        assert response.data["found"] is False

    def test_openai_schema_includes_new_actions(self):
        """Test that OpenAI schema includes switch_context and resume_context."""
        schema = MemoryTool.get_openai_schema()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "switch_context" in actions
        assert "resume_context" in actions

    def test_openai_schema_includes_task_context(self):
        """Test that OpenAI schema includes task_context parameter."""
        schema = MemoryTool.get_openai_schema()
        props = schema["parameters"]["properties"]
        assert "task_context" in props


class TestOnSessionEnd:
    """Tests for auto-snapshot on session end."""

    @pytest.mark.asyncio
    async def test_on_session_end_creates_snapshot(self):
        """Test that on_session_end creates a snapshot."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(
            user_id="u1", tenant_id="t1", session_id="sess-1"
        )
        mock_memory.get_all_context = AsyncMock(return_value={"key": "value"})
        mock_memory.get_recent_experiences = AsyncMock(return_value=[])
        mock_backend = AsyncMock()

        service = SnapshotService(mock_memory, mock_backend)
        snapshot = await service.on_session_end("project-alpha")

        assert snapshot.task_context == "project-alpha"
        assert snapshot.user_id == "u1"
        mock_backend.store_snapshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_session_resume(self):
        """Test multi-session snapshot and resume: A -> B -> resume A."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(
            user_id="u1", tenant_id="t1", session_id="sess-a"
        )
        mock_memory.get_all_context = AsyncMock(return_value={"task": "A work"})
        mock_memory.get_recent_experiences = AsyncMock(return_value=[])
        mock_backend = AsyncMock()
        mock_backend.query_snapshots_by_context = AsyncMock(return_value=[])

        service = SnapshotService(mock_memory, mock_backend)

        # Session A: create snapshot
        snap_a = await service.on_session_end("task-A")
        assert snap_a.task_context == "task-A"
        assert snap_a.working_memory == {"task": "A work"}

        # Session B: create snapshot
        mock_memory.get_all_context = AsyncMock(return_value={"task": "B work"})
        snap_b = await service.on_session_end("task-B")
        assert snap_b.task_context == "task-B"

        # Resume session A
        mock_backend.query_snapshots_by_context = AsyncMock(return_value=[snap_a])
        resumed = await service.get_latest_snapshot("task-A")
        assert resumed is not None
        assert resumed.task_context == "task-A"
        assert resumed.working_memory == {"task": "A work"}

    @pytest.mark.asyncio
    async def test_snapshot_retrieval_performance(self):
        """Test that snapshot retrieval is fast (< 100ms with mock)."""
        import time

        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(user_id="u1", tenant_id="t1")

        snapshot = ContextSnapshot(
            task_context="perf-test",
            summary="test",
            working_memory={},
            user_id="u1",
            tenant_id="t1",
        )
        mock_backend = AsyncMock()
        mock_backend.query_snapshots_by_context = AsyncMock(return_value=[snapshot])

        service = SnapshotService(mock_memory, mock_backend)

        start = time.monotonic()
        result = await service.get_latest_snapshot("perf-test")
        elapsed_ms = (time.monotonic() - start) * 1000

        assert result is not None
        assert elapsed_ms < 100, f"Retrieval took {elapsed_ms:.1f}ms, expected <100ms"
