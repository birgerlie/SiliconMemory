"""Tests for SM-5: Passive Ingestion Adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from silicon_memory.ingestion.types import (
    IngestionAdapter,
    IngestionConfig,
    IngestionResult,
)
from silicon_memory.ingestion.meeting import (
    MeetingTranscriptAdapter,
    MeetingTranscriptConfig,
    TranscriptSegment,
)
from silicon_memory.core.types import Experience, Procedure
from silicon_memory.core.utils import utc_now


# ============================================================================
# Unit tests: IngestionResult dataclass
# ============================================================================


class TestIngestionResult:
    """Unit tests for IngestionResult."""

    def test_defaults(self):
        """Test default values."""
        result = IngestionResult()
        assert result.experiences_created == 0
        assert result.entities_resolved == 0
        assert result.decisions_detected == 0
        assert result.action_items_detected == 0
        assert result.errors == []
        assert result.details == {}
        assert result.source_type == ""
        assert isinstance(result.ingested_at, datetime)

    def test_success_property(self):
        """Test success property."""
        # Success when experiences created
        result = IngestionResult(experiences_created=3)
        assert result.success is True

        # Success when no errors and no experiences (empty input)
        result = IngestionResult()
        assert result.success is True

        # Not success when errors and no experiences
        result = IngestionResult(errors=["error"])
        assert result.success is False

    def test_has_errors_property(self):
        """Test has_errors property."""
        assert IngestionResult().has_errors is False
        assert IngestionResult(errors=["e"]).has_errors is True

    def test_summary(self):
        """Test summary generation."""
        result = IngestionResult(
            source_type="meeting_transcript",
            experiences_created=5,
            entities_resolved=3,
            action_items_detected=2,
            errors=["Parse warning"],
        )
        summary = result.summary()
        assert "meeting_transcript" in summary
        assert "5 experiences" in summary
        assert "3 entities" in summary
        assert "2 action items" in summary
        assert "1 errors" in summary


# ============================================================================
# Unit tests: IngestionConfig
# ============================================================================


class TestIngestionConfig:
    """Unit tests for IngestionConfig."""

    def test_defaults(self):
        config = IngestionConfig()
        assert config.max_segment_length == 2000
        assert config.min_segment_length == 50
        assert config.llm_temperature == 0.3
        assert config.fallback_to_heuristic is True


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestIngestionProtocol:
    """Test that MeetingTranscriptAdapter satisfies IngestionAdapter protocol."""

    def test_protocol_satisfaction(self):
        """MeetingTranscriptAdapter should be an IngestionAdapter."""
        adapter = MeetingTranscriptAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        """Test source_type property."""
        adapter = MeetingTranscriptAdapter()
        assert adapter.source_type == "meeting_transcript"


# ============================================================================
# Unit tests: TranscriptSegment
# ============================================================================


class TestTranscriptSegment:
    """Unit tests for TranscriptSegment."""

    def test_defaults(self):
        seg = TranscriptSegment()
        assert seg.content == ""
        assert seg.speakers == []
        assert seg.start_time is None
        assert seg.end_time is None
        assert seg.topic is None
        assert seg.segment_index == 0


# ============================================================================
# Unit tests: MeetingTranscriptConfig
# ============================================================================


class TestMeetingTranscriptConfig:
    """Unit tests for MeetingTranscriptConfig."""

    def test_defaults(self):
        config = MeetingTranscriptConfig()
        assert config.segment_by_topic is True
        assert config.time_block_minutes == 5
        assert config.extract_action_items is True
        assert config.resolve_entities is True
        assert config.auto_create_speakers is True
        # Inherited from IngestionConfig
        assert config.max_segment_length == 2000
        assert config.min_segment_length == 50

    def test_inherits_from_ingestion_config(self):
        assert issubclass(MeetingTranscriptConfig, IngestionConfig)


# ============================================================================
# Unit tests: Transcript parsing (3 formats)
# ============================================================================


class TestTranscriptParsing:
    """Test parsing of all three transcript formats."""

    def _adapter(self):
        return MeetingTranscriptAdapter()

    def test_parse_timestamped_format(self):
        """Test parsing [HH:MM:SS] Speaker: text format."""
        transcript = (
            "[00:00:05] Alice: Welcome everyone to the standup.\n"
            "[00:00:15] Bob: Thanks Alice. I worked on the API yesterday.\n"
            "[00:00:30] Alice: Great. Any blockers?\n"
            "[00:00:45] Bob: None so far.\n"
        )
        lines = self._adapter()._parse_transcript_lines(transcript)
        assert len(lines) == 4
        assert lines[0]["timestamp"] == "00:00:05"
        assert lines[0]["speaker"] == "Alice"
        assert "Welcome" in lines[0]["content"]
        assert lines[1]["speaker"] == "Bob"

    def test_parse_speaker_only_format(self):
        """Test parsing Speaker: text format."""
        transcript = (
            "Alice: Welcome everyone.\n"
            "Bob: I worked on the API.\n"
            "Alice: Any blockers?\n"
        )
        lines = self._adapter()._parse_transcript_lines(transcript)
        assert len(lines) == 3
        assert lines[0]["speaker"] == "Alice"
        assert lines[0]["timestamp"] is None
        assert lines[1]["speaker"] == "Bob"

    def test_parse_raw_text_format(self):
        """Test parsing raw text (no speakers/timestamps)."""
        transcript = (
            "The meeting started with a discussion about the API.\n"
            "Several approaches were considered.\n"
            "The team decided to use REST.\n"
        )
        lines = self._adapter()._parse_transcript_lines(transcript)
        assert len(lines) == 3
        assert lines[0]["speaker"] is None
        assert lines[0]["timestamp"] is None
        assert "API" in lines[0]["content"]

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        lines = self._adapter()._parse_transcript_lines("")
        assert lines == []

    def test_parse_timestamped_with_continuation(self):
        """Test that lines without timestamps continue previous entry."""
        transcript = (
            "[00:01:00] Alice: This is a long statement\n"
            "that continues on the next line.\n"
            "[00:02:00] Bob: Got it.\n"
        )
        lines = self._adapter()._parse_transcript_lines(transcript)
        assert len(lines) == 2
        assert "continues on the next line" in lines[0]["content"]


# ============================================================================
# Unit tests: Heuristic segmentation
# ============================================================================


class TestHeuristicSegmentation:
    """Test heuristic segmentation strategies."""

    def _adapter(self, config=None):
        return MeetingTranscriptAdapter(config=config)

    def test_segment_by_speaker(self):
        """Test segmentation by speaker turns."""
        lines = [
            {"timestamp": None, "speaker": "Alice", "content": "Hello"},
            {"timestamp": None, "speaker": "Alice", "content": "How are you?"},
            {"timestamp": None, "speaker": "Bob", "content": "I'm good, thanks."},
            {"timestamp": None, "speaker": "Bob", "content": "Let's discuss the API."},
            {"timestamp": None, "speaker": "Alice", "content": "Sure, let's do that."},
        ]
        adapter = self._adapter()
        segments = adapter._segment_by_speaker(lines)
        assert len(segments) >= 2  # At least Alice, Bob (may merge short segments)

    def test_segment_by_time(self):
        """Test segmentation by time blocks."""
        lines = [
            {"timestamp": "00:00:00", "speaker": "Alice", "content": "Start"},
            {"timestamp": "00:02:00", "speaker": "Bob", "content": "Topic 1"},
            {"timestamp": "00:06:00", "speaker": "Alice", "content": "Topic 2"},
            {"timestamp": "00:11:00", "speaker": "Bob", "content": "Topic 3"},
        ]
        adapter = self._adapter(MeetingTranscriptConfig(time_block_minutes=5))
        segments = adapter._segment_by_time(lines)
        assert len(segments) >= 2  # At least 2 time blocks

    def test_segment_by_paragraphs(self):
        """Test segmentation of raw text by paragraphs."""
        lines = [
            {"timestamp": None, "speaker": None, "content": "First paragraph start."},
            {"timestamp": None, "speaker": None, "content": "First paragraph end."},
            {"timestamp": None, "speaker": None, "content": ""},
            {"timestamp": None, "speaker": None, "content": "Second paragraph."},
        ]
        adapter = self._adapter(MeetingTranscriptConfig(min_segment_length=5))
        segments = adapter._segment_by_paragraphs(lines)
        assert len(segments) >= 1

    def test_segment_heuristic_auto_detection(self):
        """Test that heuristic auto-detects format."""
        adapter = self._adapter()

        # Timestamped lines
        ts_lines = [
            {"timestamp": "00:01:00", "speaker": "Alice", "content": "Hello"},
            {"timestamp": "00:06:00", "speaker": "Bob", "content": "World"},
        ]
        segments = adapter._segment_heuristic(ts_lines)
        assert len(segments) >= 1

        # Speaker-only lines
        speaker_lines = [
            {"timestamp": None, "speaker": "Alice", "content": "Hello there"},
            {"timestamp": None, "speaker": "Bob", "content": "Hi Alice"},
        ]
        segments = adapter._segment_heuristic(speaker_lines)
        assert len(segments) >= 1

        # Raw lines
        raw_lines = [
            {"timestamp": None, "speaker": None, "content": "Some text"},
        ]
        segments = adapter._segment_heuristic(raw_lines)
        assert len(segments) >= 1

    def test_empty_lines(self):
        """Test segmentation of empty input."""
        adapter = self._adapter()
        assert adapter._segment_heuristic([]) == []


# ============================================================================
# Unit tests: Action item extraction
# ============================================================================


class TestActionItemExtraction:
    """Test heuristic action item extraction."""

    def _adapter(self):
        return MeetingTranscriptAdapter()

    def test_action_keyword_extraction(self):
        """Test extraction of ACTION: and TODO: items."""
        segments = [
            TranscriptSegment(
                content="We discussed the plan. ACTION: Deploy by Friday.\nTODO: Update docs.",
                segment_index=0,
            ),
        ]
        items = self._adapter()._extract_action_items_heuristic(segments)
        assert len(items) == 2
        assert any("Deploy" in item["action"] for item in items)
        assert any("Update docs" in item["action"] for item in items)

    def test_will_should_pattern(self):
        """Test extraction of '[Name] will/should [verb]' patterns."""
        segments = [
            TranscriptSegment(
                content="Alice will review the PR by EOD.\nBob should update the tests.",
                speakers=["Alice", "Bob"],
                segment_index=0,
            ),
        ]
        items = self._adapter()._extract_action_items_heuristic(segments)
        assert len(items) >= 2
        assert any(item.get("owner") == "Alice" for item in items)
        assert any(item.get("owner") == "Bob" for item in items)

    def test_no_action_items(self):
        """Test no extraction when no patterns match."""
        segments = [
            TranscriptSegment(
                content="The meeting went well. Everyone agreed on the approach.",
                segment_index=0,
            ),
        ]
        items = self._adapter()._extract_action_items_heuristic(segments)
        assert items == []


# ============================================================================
# Integration tests: Full pipeline with mocked memory
# ============================================================================


class TestMeetingIngestionPipeline:
    """Integration tests for the full meeting ingestion pipeline."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.record_experience = AsyncMock()
        return memory

    async def test_full_pipeline_timestamped(self, mock_memory):
        """Test full ingestion of a timestamped transcript."""
        transcript = (
            "[00:00:00] Alice: Let's start the standup.\n"
            "[00:00:10] Bob: I finished the API endpoint yesterday.\n"
            "[00:00:20] Alice: Great. Any blockers?\n"
            "[00:00:30] Bob: None. TODO: write integration tests.\n"
            "[00:05:30] Alice: Moving on to the next topic.\n"
            "[00:05:45] Charlie: I'm working on the frontend.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(
                resolve_entities=False,
                time_block_minutes=5,
            )
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-1", "title": "Standup"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "meeting_transcript"
        assert result.action_items_detected >= 1
        assert not result.has_errors or result.success

    async def test_full_pipeline_speaker_format(self, mock_memory):
        """Test ingestion of speaker-labeled transcript."""
        transcript = (
            "Alice: Welcome to the meeting.\n"
            "Bob: Thanks. I have an update on the migration.\n"
            "Alice: Go ahead.\n"
            "Bob: We should switch to PostgreSQL. ACTION: Create migration plan.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-2"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1

    async def test_full_pipeline_raw_text(self, mock_memory):
        """Test ingestion of raw text transcript."""
        transcript = (
            "The team discussed the upcoming release.\n"
            "Several concerns were raised about performance.\n"
            "It was decided to add caching before launch.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-3"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1

    async def test_empty_transcript(self, mock_memory):
        """Test handling of empty transcript."""
        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content="",
            metadata={"meeting_id": "mtg-4"},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors

    async def test_bytes_input(self, mock_memory):
        """Test that bytes input is decoded."""
        transcript = b"Alice: Hello\nBob: Hi\n"

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-5"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1

    async def test_experience_context_metadata(self, mock_memory):
        """Test that experiences include correct context metadata."""
        transcript = "Alice: Let's discuss the API design.\n"

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-6", "title": "API Review"},
            memory=mock_memory,
        )

        # Check that record_experience was called with proper context
        call_args = mock_memory.record_experience.call_args_list
        assert len(call_args) >= 1
        exp = call_args[0][0][0]  # First positional arg of first call
        assert exp.context["meeting_id"] == "mtg-6"
        assert exp.context["source_type"] == "meeting_transcript"
        assert exp.session_id == "mtg-6"
        assert exp.user_id == "user-1"
        assert exp.tenant_id == "acme"

    async def test_llm_segmentation_mock(self, mock_memory):
        """Test that LLM segmentation is used when provider is available."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value='[{"topic": "Standup", "start_line": 0, "end_line": 1, "speakers": ["Alice"]}]')

        transcript = "Alice: Hello\nBob: Hi\n"

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(
                resolve_entities=False,
                segment_by_topic=True,
            )
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-7"},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        assert result.experiences_created >= 1
        mock_llm.complete.assert_awaited()

    async def test_llm_failure_falls_back(self, mock_memory):
        """Test fallback to heuristic when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM error"))

        transcript = "Alice: Hello\nBob: Hi\n"

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-8"},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        # Should still produce experiences via heuristic fallback
        assert result.experiences_created >= 1


# ============================================================================
# Integration test: Router ingest_from entry point
# ============================================================================


class TestIngestFromRouter:
    """Test the SiliconMemory.ingest_from() entry point."""

    async def test_ingest_from_injects_user_context(self):
        """Test that ingest_from injects user context into metadata."""
        from silicon_memory.memory.silicondb_router import SiliconMemory

        # Create a mock adapter that captures the metadata
        captured_metadata = {}

        class MockAdapter:
            @property
            def source_type(self):
                return "test"

            async def ingest(self, content, metadata, memory, llm_provider=None):
                captured_metadata.update(metadata)
                return IngestionResult(source_type="test", experiences_created=1)

        # We need to mock enough of SiliconMemory to test ingest_from
        # Use a MagicMock with the actual method
        mock_memory = MagicMock()
        mock_memory._user_context = MagicMock(
            user_id="user-1", tenant_id="acme"
        )

        # Bind the actual ingest_from method
        import types
        mock_memory.ingest_from = types.MethodType(
            SiliconMemory.ingest_from, mock_memory
        )

        adapter = MockAdapter()
        result = await mock_memory.ingest_from(
            adapter=adapter,
            content="test content",
            metadata={"meeting_id": "m1"},
        )

        assert result.success
        assert captured_metadata["user_id"] == "user-1"
        assert captured_metadata["tenant_id"] == "acme"
        assert captured_metadata["meeting_id"] == "m1"

    async def test_ingest_from_wraps_errors(self):
        """Test that ingest_from wraps adapter exceptions."""
        from silicon_memory.memory.silicondb_router import SiliconMemory
        import types

        class FailingAdapter:
            @property
            def source_type(self):
                return "failing"

            async def ingest(self, content, metadata, memory, llm_provider=None):
                raise RuntimeError("Adapter crashed")

        mock_memory = MagicMock()
        mock_memory._user_context = MagicMock(user_id="u1", tenant_id="t1")
        mock_memory.ingest_from = types.MethodType(
            SiliconMemory.ingest_from, mock_memory
        )

        adapter = FailingAdapter()
        result = await mock_memory.ingest_from(
            adapter=adapter, content="test", metadata={}
        )

        assert result.has_errors
        assert "Adapter crashed" in result.errors[0]


# ============================================================================
# Unit tests: Entity resolution (optional)
# ============================================================================


class TestEntityResolution:
    """Test optional entity resolution integration."""

    async def test_resolve_entities_with_resolver(self):
        """Test that entity resolution runs when resolver is provided."""
        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(return_value=MagicMock(resolved=[MagicMock()]))

        adapter = MeetingTranscriptAdapter(entity_resolver=mock_resolver)

        segments = [
            TranscriptSegment(
                content="Alice discussed the plan.",
                speakers=["Alice", "Bob"],
            ),
        ]

        count = await adapter._resolve_entities(segments, MagicMock(), {})
        assert count >= 1
        assert mock_resolver.resolve.await_count >= 1

    async def test_no_resolver_returns_zero(self):
        """Test that no resolver means zero entities resolved."""
        adapter = MeetingTranscriptAdapter()  # No resolver
        count = await adapter._resolve_entities([], MagicMock(), {})
        assert count == 0


# ============================================================================
# Integration test: Action items persisted as Procedures
# ============================================================================


class TestActionItemPersistence:
    """Test that extracted action items are stored as Procedure memories."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.record_experience = AsyncMock()
        memory.commit_procedure = AsyncMock()
        memory._backend = MagicMock()
        memory._backend._db = MagicMock()
        memory._backend._db.add_edge = MagicMock()
        return memory

    async def test_action_items_stored_as_procedures(self, mock_memory):
        """Test that detected action items are committed as Procedure memories."""
        transcript = (
            "Alice: We need to update the docs by Friday.\n"
            "Bob: Sure. TODO: Update API documentation.\n"
            "Alice: Also ACTION: Deploy staging environment.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-proc"},
            memory=mock_memory,
        )

        assert result.action_items_detected >= 2
        # commit_procedure should be called for each action item
        assert mock_memory.commit_procedure.await_count >= 2

        # Verify the stored procedures have correct properties
        for call in mock_memory.commit_procedure.call_args_list:
            proc = call[0][0]
            assert isinstance(proc, Procedure)
            assert proc.user_id == "user-1"
            assert proc.tenant_id == "acme"
            assert "action_item" in proc.tags
            assert "meeting" in proc.tags
            assert len(proc.steps) >= 1

    async def test_action_item_names_truncated(self, mock_memory):
        """Test that action item names are truncated to 100 chars."""
        long_action = "A" * 200
        transcript = f"Alice: ACTION: {long_action}\n"

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-trunc"},
            memory=mock_memory,
        )

        if mock_memory.commit_procedure.await_count > 0:
            proc = mock_memory.commit_procedure.call_args[0][0]
            assert len(proc.name) <= 100


# ============================================================================
# Integration test: Graph edge creation
# ============================================================================


class TestGraphEdgeCreation:
    """Test that graph edges are created for meetings."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.record_experience = AsyncMock()
        memory.commit_procedure = AsyncMock()
        memory._backend = MagicMock()
        memory._backend._db = MagicMock()
        memory._backend._db.add_edge = MagicMock()
        return memory

    async def test_meeting_speaker_edges_created(self, mock_memory):
        """Test that meeting → speaker edges are created."""
        transcript = (
            "Alice: Hello everyone.\n"
            "Bob: Hi Alice.\n"
            "Charlie: Good morning.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-graph"},
            memory=mock_memory,
        )

        # Should create edges for Alice, Bob, Charlie
        add_edge_calls = mock_memory._backend._db.add_edge.call_args_list
        participated_edges = [
            c for c in add_edge_calls
            if c[1].get("edge_type") == "participated_in"
            or (len(c[0]) >= 3 and c[0][2] == "participated_in")
        ]
        assert len(participated_edges) >= 3

    async def test_meeting_action_edges_created(self, mock_memory):
        """Test that meeting → action item edges are created."""
        transcript = (
            "Alice: Let's discuss the plan.\n"
            "Bob: ACTION: Deploy by Friday.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "mtg-action-edges"},
            memory=mock_memory,
        )

        add_edge_calls = mock_memory._backend._db.add_edge.call_args_list
        action_edges = [
            c for c in add_edge_calls
            if c[1].get("edge_type") == "has_action"
            or (len(c[0]) >= 3 and c[0][2] == "has_action")
        ]
        assert len(action_edges) >= 1

    async def test_graph_edges_graceful_when_no_backend(self):
        """Test that graph edge creation doesn't crash without backend."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme"
        )
        mock_memory.record_experience = AsyncMock()
        mock_memory.commit_procedure = AsyncMock()
        # No _backend attribute
        del mock_memory._backend

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content="Alice: Hello\n",
            metadata={"meeting_id": "mtg-no-backend"},
            memory=mock_memory,
        )

        # Should still succeed for experiences
        assert result.experiences_created >= 1


# ============================================================================
# E2E test: Transcript → ingest → verify output
# ============================================================================


class TestEndToEnd:
    """End-to-end test: transcript ingestion produces experiences, procedures, edges."""

    async def test_full_e2e_pipeline(self):
        """Test full pipeline: transcript → experiences + procedures + edges."""
        mock_memory = AsyncMock()
        mock_memory.user_context = MagicMock(
            user_id="user-e2e", tenant_id="acme", session_id="e2e-session"
        )
        mock_memory.record_experience = AsyncMock()
        mock_memory.commit_procedure = AsyncMock()
        mock_memory._backend = MagicMock()
        mock_memory._backend._db = MagicMock()
        mock_memory._backend._db.add_edge = MagicMock()

        transcript = (
            "[00:00:00] Alice: Welcome to the sprint planning.\n"
            "[00:01:00] Bob: I'll present the backend updates.\n"
            "[00:02:00] Bob: We migrated the database to PostgreSQL.\n"
            "[00:03:00] Alice: Great. Any issues?\n"
            "[00:04:00] Bob: None so far. TODO: Write migration tests.\n"
            "[00:05:00] Alice: Moving to frontend updates.\n"
            "[00:06:00] Charlie: I finished the new dashboard. ACTION: Deploy to staging.\n"
            "[00:07:00] Alice: Thanks. Alice will review the PR by EOD.\n"
            "[00:08:00] Alice: Let's wrap up. Great meeting everyone.\n"
        )

        adapter = MeetingTranscriptAdapter(
            config=MeetingTranscriptConfig(
                resolve_entities=False,
                time_block_minutes=5,
            )
        )
        result = await adapter.ingest(
            content=transcript,
            metadata={"meeting_id": "e2e-sprint-planning", "title": "Sprint Planning"},
            memory=mock_memory,
        )

        # Experiences created (at least 1 segment)
        assert result.experiences_created >= 1
        assert result.source_type == "meeting_transcript"

        # Action items detected and stored as procedures
        assert result.action_items_detected >= 2
        assert mock_memory.commit_procedure.await_count >= 2

        # Graph edges created for speakers
        add_edge_calls = mock_memory._backend._db.add_edge.call_args_list
        assert len(add_edge_calls) >= 3  # At least 3 speakers + action items

        # Verify no fatal errors
        assert result.success
