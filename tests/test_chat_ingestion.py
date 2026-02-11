"""Tests for base chat ingestion adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from silicon_memory.ingestion.types import (
    IngestionAdapter,
    IngestionConfig,
    IngestionResult,
)
from silicon_memory.ingestion.chat import (
    BaseChatAdapter,
    ChatConfig,
    ChatMessage,
    ChatThread,
)


# ============================================================================
# Test adapter (concrete subclass for testing the base pipeline)
# ============================================================================


class _TestChatAdapter(BaseChatAdapter):
    """Concrete subclass of BaseChatAdapter for unit testing."""

    @property
    def source_type(self) -> str:
        return "test_chat"

    def _parse_messages(self, content, metadata):
        data = json.loads(content)
        messages = []
        for msg in data:
            messages.append(ChatMessage(
                author=msg.get("author", ""),
                author_id=msg.get("author_id", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
                message_id=msg.get("id", ""),
                thread_id=msg.get("thread_id"),
                channel=metadata.get("channel", ""),
                channel_id=metadata.get("channel_id", ""),
                is_bot=msg.get("is_bot", False),
            ))
        return messages

    def _group_into_threads(self, messages):
        # Simple grouping: by thread_id, or all in one thread
        groups: dict[str, list[ChatMessage]] = {}
        for msg in messages:
            tid = msg.thread_id or "default"
            groups.setdefault(tid, []).append(msg)

        threads = []
        for tid, msgs in groups.items():
            participants = list({m.author for m in msgs if m.author})
            threads.append(ChatThread(
                thread_id=tid,
                channel=msgs[0].channel if msgs else "",
                channel_id=msgs[0].channel_id if msgs else "",
                messages=msgs,
                participants=participants,
                start_time=msgs[0].timestamp if msgs else None,
                end_time=msgs[-1].timestamp if msgs else None,
            ))
        return threads


# ============================================================================
# Unit tests: ChatMessage dataclass
# ============================================================================


class TestChatMessage:
    """Unit tests for ChatMessage."""

    def test_defaults(self):
        msg = ChatMessage()
        assert msg.author == ""
        assert msg.author_id == ""
        assert msg.content == ""
        assert msg.timestamp is None
        assert msg.message_id == ""
        assert msg.thread_id is None
        assert msg.reply_to is None
        assert msg.channel == ""
        assert msg.channel_id == ""
        assert msg.reactions == []
        assert msg.attachments == []
        assert msg.mentions == []
        assert msg.is_bot is False
        assert msg.platform_metadata == {}

    def test_construction(self):
        msg = ChatMessage(
            author="Alice",
            author_id="U123",
            content="Hello world",
            timestamp="2024-01-01T10:00:00Z",
            message_id="msg-1",
            thread_id="thread-1",
            channel="general",
        )
        assert msg.author == "Alice"
        assert msg.author_id == "U123"
        assert msg.content == "Hello world"
        assert msg.timestamp == "2024-01-01T10:00:00Z"
        assert msg.message_id == "msg-1"
        assert msg.thread_id == "thread-1"
        assert msg.channel == "general"


# ============================================================================
# Unit tests: ChatThread dataclass
# ============================================================================


class TestChatThread:
    """Unit tests for ChatThread."""

    def test_defaults(self):
        thread = ChatThread()
        assert thread.thread_id == ""
        assert thread.messages == []
        assert thread.participants == []
        assert thread.message_count == 0
        assert thread.combined_text == ""

    def test_message_count(self):
        thread = ChatThread(
            messages=[
                ChatMessage(author="Alice", content="Hello"),
                ChatMessage(author="Bob", content="Hi"),
            ]
        )
        assert thread.message_count == 2

    def test_combined_text(self):
        thread = ChatThread(
            messages=[
                ChatMessage(author="Alice", content="Hello"),
                ChatMessage(author="Bob", content="Hi"),
                ChatMessage(content="System message"),
            ]
        )
        text = thread.combined_text
        assert "Alice: Hello" in text
        assert "Bob: Hi" in text
        assert "System message" in text


# ============================================================================
# Unit tests: ChatConfig
# ============================================================================


class TestChatConfig:
    """Unit tests for ChatConfig."""

    def test_defaults(self):
        config = ChatConfig()
        assert config.extract_action_items is True
        assert config.extract_beliefs is True
        assert config.resolve_entities is True
        assert config.create_graph_edges is True
        assert config.group_by_threads is True
        assert config.max_thread_messages == 500
        assert config.min_thread_messages == 2
        assert config.skip_bot_messages is False
        assert config.belief_confidence == 0.5
        assert config.max_beliefs_per_thread == 10

    def test_inherits_from_ingestion_config(self):
        assert issubclass(ChatConfig, IngestionConfig)

    def test_custom_values(self):
        config = ChatConfig(
            skip_bot_messages=True,
            belief_confidence=0.8,
            min_thread_messages=5,
        )
        assert config.skip_bot_messages is True
        assert config.belief_confidence == 0.8
        assert config.min_thread_messages == 5


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestChatProtocol:
    """Test that BaseChatAdapter subclass satisfies IngestionAdapter protocol."""

    def test_protocol_satisfaction(self):
        adapter = _TestChatAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        adapter = _TestChatAdapter()
        assert adapter.source_type == "test_chat"


# ============================================================================
# Integration tests: Full pipeline
# ============================================================================


class TestChatIngestionPipeline:
    """Integration tests for the shared chat ingestion pipeline."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.record_experience = AsyncMock()
        memory.commit_procedure = AsyncMock()
        memory.commit_belief = AsyncMock()
        return memory

    @pytest.fixture
    def sample_messages(self):
        return json.dumps([
            {"author": "Alice", "content": "Let's discuss the API redesign.", "id": "m1", "thread_id": "t1", "timestamp": "1"},
            {"author": "Bob", "content": "I agree. I'll draft the spec by Friday.", "id": "m2", "thread_id": "t1", "timestamp": "2"},
            {"author": "Alice", "content": "We decided to use PostgreSQL.", "id": "m3", "thread_id": "t1", "timestamp": "3"},
        ])

    async def test_full_pipeline(self, mock_memory, sample_messages):
        """Test full ingestion pipeline creates experiences."""
        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={"channel": "engineering"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "test_chat"
        assert not result.has_errors or result.success

    async def test_action_items_detected(self, mock_memory, sample_messages):
        """Test action item extraction from chat messages."""
        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
        )

        # "I'll draft the spec" should trigger the I'll pattern
        assert result.action_items_detected >= 1
        assert "action_items" in result.details

    async def test_beliefs_extracted(self, mock_memory, sample_messages):
        """Test belief/knowledge extraction from chat messages."""
        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
        )

        # "We decided to use PostgreSQL" should trigger belief pattern
        assert result.decisions_detected >= 1
        assert "beliefs" in result.details
        mock_memory.commit_belief.assert_awaited()

    async def test_procedures_stored(self, mock_memory, sample_messages):
        """Test action items are persisted as Procedures."""
        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
        )

        if result.action_items_detected > 0:
            mock_memory.commit_procedure.assert_awaited()

    async def test_experience_context_metadata(self, mock_memory, sample_messages):
        """Test that experiences include correct context metadata."""
        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
                extract_beliefs=False,
            )
        )
        await adapter.ingest(
            content=sample_messages,
            metadata={"channel": "engineering"},
            memory=mock_memory,
        )

        call_args = mock_memory.record_experience.call_args_list
        assert len(call_args) >= 1
        exp = call_args[0][0][0]
        assert exp.context["source_type"] == "test_chat"
        assert exp.context["channel"] == "engineering"
        assert "participants" in exp.context
        assert exp.user_id == "user-1"
        assert exp.tenant_id == "acme"

    async def test_empty_content(self, mock_memory):
        """Test graceful handling of empty input."""
        adapter = _TestChatAdapter(
            config=ChatConfig(resolve_entities=False, create_graph_edges=False)
        )
        result = await adapter.ingest(
            content="",
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors

    async def test_bytes_input(self, mock_memory, sample_messages):
        """Test handling of bytes input."""
        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages.encode("utf-8"),
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1

    async def test_skip_bot_messages(self, mock_memory):
        """Test bot message filtering."""
        messages = json.dumps([
            {"author": "Alice", "content": "Hello", "id": "m1", "thread_id": "t1"},
            {"author": "Bot", "content": "Auto reply", "id": "m2", "thread_id": "t1", "is_bot": True},
            {"author": "Bob", "content": "Hi", "id": "m3", "thread_id": "t1"},
        ])

        adapter = _TestChatAdapter(
            config=ChatConfig(
                skip_bot_messages=True,
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(
            content=messages,
            metadata={},
            memory=mock_memory,
        )

        # Should still create experiences (bot message filtered out but 2 remain)
        assert result.experiences_created >= 1

    async def test_min_thread_messages_filter(self, mock_memory):
        """Test that threads below min_thread_messages are filtered out."""
        messages = json.dumps([
            {"author": "Alice", "content": "Solo message", "id": "m1", "thread_id": "alone"},
        ])

        adapter = _TestChatAdapter(
            config=ChatConfig(
                min_thread_messages=2,
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(
            content=messages,
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors

    async def test_llm_action_items(self, mock_memory, sample_messages):
        """Test action item extraction with LLM mock."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value='[{"action": "Draft spec", "owner": "Bob", "thread_index": 0}]'
        )

        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        assert result.action_items_detected >= 1
        mock_llm.complete.assert_awaited()

    async def test_llm_belief_extraction(self, mock_memory, sample_messages):
        """Test belief extraction with LLM mock."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value='[{"belief": "Team decided to use PostgreSQL", "confidence": 0.9, "thread_id": "t1"}]'
        )

        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
            )
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        assert result.decisions_detected >= 1
        mock_memory.commit_belief.assert_awaited()

    async def test_entity_resolution(self, mock_memory, sample_messages):
        """Test entity resolution with mock resolver."""
        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(
            return_value=MagicMock(resolved=[MagicMock()])
        )

        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=True,
                create_graph_edges=False,
                extract_action_items=False,
                extract_beliefs=False,
            ),
            entity_resolver=mock_resolver,
        )
        result = await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
        )

        assert result.entities_resolved >= 1
        assert mock_resolver.resolve.await_count >= 1

    async def test_graph_edges(self, mock_memory, sample_messages):
        """Test graph edge creation."""
        mock_db = MagicMock()
        mock_db.add_edge = MagicMock()
        mock_backend = MagicMock()
        mock_backend._db = mock_db
        mock_memory._backend = mock_backend

        adapter = _TestChatAdapter(
            config=ChatConfig(
                resolve_entities=False,
                create_graph_edges=True,
                extract_action_items=False,
                extract_beliefs=False,
            )
        )
        await adapter.ingest(
            content=sample_messages,
            metadata={},
            memory=mock_memory,
        )

        # Should create participated_in and communicates_with edges
        assert mock_db.add_edge.call_count >= 1
        edge_types = [call.kwargs.get("edge_type", call.args[2] if len(call.args) > 2 else "")
                      for call in mock_db.add_edge.call_args_list]
        assert "participated_in" in edge_types
        assert "communicates_with" in edge_types
