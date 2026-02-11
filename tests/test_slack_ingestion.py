"""Tests for Slack ingestion adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from silicon_memory.ingestion.types import IngestionAdapter
from silicon_memory.ingestion.chat import ChatConfig
from silicon_memory.ingestion.slack import SlackAdapter, SlackConfig


# ============================================================================
# Unit tests: SlackConfig
# ============================================================================


class TestSlackConfig:
    def test_defaults(self):
        config = SlackConfig()
        assert config.include_reactions is True
        assert config.reaction_weight_boost == 0.1
        assert "channel_join" in config.skip_subtypes
        assert config.user_map == {}

    def test_inherits_from_chat_config(self):
        assert issubclass(SlackConfig, ChatConfig)


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestSlackProtocol:
    def test_protocol_satisfaction(self):
        adapter = SlackAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        adapter = SlackAdapter()
        assert adapter.source_type == "slack"


# ============================================================================
# Unit tests: Message parsing
# ============================================================================


class TestSlackMessageParsing:

    def _adapter(self, **kwargs):
        return SlackAdapter(config=SlackConfig(**kwargs))

    def test_parse_json_array(self):
        content = json.dumps([
            {"user": "U123", "text": "Hello", "ts": "1700000001.000000"},
            {"user": "U456", "text": "Hi there", "ts": "1700000002.000000"},
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[0].author_id == "U123"

    def test_parse_wrapper_object(self):
        content = json.dumps({
            "messages": [
                {"user": "U123", "text": "Hello", "ts": "1700000001.000000"},
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 1

    def test_skip_subtypes(self):
        content = json.dumps([
            {"user": "U123", "text": "Hello", "ts": "1"},
            {"user": "U456", "text": "joined", "ts": "2", "subtype": "channel_join"},
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 1

    def test_user_map(self):
        content = json.dumps([
            {"user": "U123", "text": "Hello", "ts": "1"},
        ])
        messages = self._adapter(user_map={"U123": "Alice"})._parse_messages(content, {})
        assert messages[0].author == "Alice"

    def test_bot_detection(self):
        content = json.dumps([
            {"user": "U123", "text": "Human msg", "ts": "1"},
            {"user": "B001", "text": "Bot msg", "ts": "2", "subtype": "bot_message"},
            {"user": "B002", "text": "Bot msg 2", "ts": "3", "bot_id": "BID"},
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert messages[0].is_bot is False
        assert messages[1].is_bot is True
        assert messages[2].is_bot is True

    def test_reactions_parsed(self):
        content = json.dumps([
            {
                "user": "U123", "text": "Good idea", "ts": "1",
                "reactions": [
                    {"name": "thumbsup", "count": 3},
                    {"name": "fire", "count": 1},
                ],
            },
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages[0].reactions) == 2
        assert messages[0].reactions[0]["name"] == "thumbsup"
        assert messages[0].reactions[0]["count"] == 3

    def test_reactions_excluded(self):
        content = json.dumps([
            {
                "user": "U123", "text": "Hello", "ts": "1",
                "reactions": [{"name": "thumbsup", "count": 1}],
            },
        ])
        messages = self._adapter(include_reactions=False)._parse_messages(content, {})
        assert messages[0].reactions == []

    def test_thread_ts(self):
        content = json.dumps([
            {"user": "U123", "text": "Parent", "ts": "1700000001.000000"},
            {"user": "U456", "text": "Reply", "ts": "1700000002.000000", "thread_ts": "1700000001.000000"},
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert messages[0].thread_id is None
        assert messages[1].thread_id == "1700000001.000000"

    def test_mentions_extracted(self):
        content = json.dumps([
            {"user": "U123", "text": "Hey <@U456> and <@U789>", "ts": "1"},
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert set(messages[0].mentions) == {"U456", "U789"}

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            self._adapter()._parse_messages("not json", {})

    def test_channel_from_metadata(self):
        content = json.dumps([{"user": "U1", "text": "Hi", "ts": "1"}])
        messages = self._adapter()._parse_messages(
            content, {"channel": "general", "channel_id": "C123"}
        )
        assert messages[0].channel == "general"
        assert messages[0].channel_id == "C123"


# ============================================================================
# Unit tests: Thread grouping
# ============================================================================


class TestSlackThreadGrouping:

    def _adapter(self):
        return SlackAdapter()

    def test_group_by_thread_ts(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Thread 1", message_id="1", thread_id="t1", timestamp="1"),
            ChatMessage(author="Bob", content="Reply 1", message_id="2", thread_id="t1", timestamp="2"),
            ChatMessage(author="Carol", content="Thread 2", message_id="3", thread_id="t2", timestamp="3"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        thread_ids = {t.thread_id for t in threads}
        assert "t1" in thread_ids
        assert "t2" in thread_ids

        t1 = next(t for t in threads if t.thread_id == "t1")
        assert t1.message_count == 2
        assert set(t1.participants) == {"Alice", "Bob"}

    def test_standalone_messages_grouped_by_proximity(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="First", message_id="1", timestamp="1000.0"),
            ChatMessage(author="Bob", content="Second", message_id="2", timestamp="1001.0"),
            # 10-minute gap
            ChatMessage(author="Carol", content="Third", message_id="3", timestamp="1601.0"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        assert len(threads) == 2

    def test_mixed_threaded_and_standalone(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Threaded", message_id="1", thread_id="t1", timestamp="1"),
            ChatMessage(author="Bob", content="Reply", message_id="2", thread_id="t1", timestamp="2"),
            ChatMessage(author="Carol", content="Standalone", message_id="3", timestamp="3.0"),
            ChatMessage(author="Dave", content="Also alone", message_id="4", timestamp="4.0"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        # Should have thread t1 plus a standalone group
        assert len(threads) >= 2


# ============================================================================
# Integration tests: Full pipeline
# ============================================================================


class TestSlackPipeline:

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
    def slack_export(self):
        return json.dumps([
            {"user": "U1", "text": "Let's decide on the database.", "ts": "1700000001.000000", "thread_ts": "1700000001.000000"},
            {"user": "U2", "text": "I'll research PostgreSQL options.", "ts": "1700000002.000000", "thread_ts": "1700000001.000000"},
            {"user": "U3", "text": "We decided to go with PostgreSQL.", "ts": "1700000003.000000", "thread_ts": "1700000001.000000"},
        ])

    async def test_full_slack_pipeline(self, mock_memory, slack_export):
        adapter = SlackAdapter(
            config=SlackConfig(
                resolve_entities=False,
                create_graph_edges=False,
                user_map={"U1": "Alice", "U2": "Bob", "U3": "Carol"},
            )
        )
        result = await adapter.ingest(
            content=slack_export,
            metadata={"channel": "engineering"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "slack"
        assert result.action_items_detected >= 1  # "I'll research..."
        assert result.decisions_detected >= 1  # "We decided..."

    async def test_bytes_input(self, mock_memory, slack_export):
        adapter = SlackAdapter(
            config=SlackConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_beliefs=False,
                extract_action_items=False,
            )
        )
        result = await adapter.ingest(
            content=slack_export.encode("utf-8"),
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1

    async def test_empty_content(self, mock_memory):
        adapter = SlackAdapter(
            config=SlackConfig(resolve_entities=False, create_graph_edges=False)
        )
        result = await adapter.ingest(
            content="",
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors
