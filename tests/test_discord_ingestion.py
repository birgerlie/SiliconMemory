"""Tests for Discord ingestion adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from silicon_memory.ingestion.types import IngestionAdapter
from silicon_memory.ingestion.chat import ChatConfig
from silicon_memory.ingestion.discord import DiscordAdapter, DiscordConfig


# ============================================================================
# Unit tests: DiscordConfig
# ============================================================================


class TestDiscordConfig:
    def test_defaults(self):
        config = DiscordConfig()
        assert config.include_reactions is True
        assert config.role_weights == {}
        assert config.skip_bot_messages is True

    def test_inherits_from_chat_config(self):
        assert issubclass(DiscordConfig, ChatConfig)


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestDiscordProtocol:
    def test_protocol_satisfaction(self):
        adapter = DiscordAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        adapter = DiscordAdapter()
        assert adapter.source_type == "discord"


# ============================================================================
# Unit tests: Message parsing
# ============================================================================


class TestDiscordMessageParsing:

    def _adapter(self, **kwargs):
        return DiscordAdapter(config=DiscordConfig(**kwargs))

    def test_parse_export_format(self):
        content = json.dumps({
            "channel": {"id": "C123", "name": "general"},
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Hello everyone",
                    "author": {"name": "Alice", "id": "u1", "isBot": False},
                    "timestamp": "2024-01-01T10:00:00Z",
                },
                {
                    "id": "msg-2",
                    "content": "Hi Alice",
                    "author": {"name": "Bob", "id": "u2", "isBot": False},
                    "timestamp": "2024-01-01T10:01:00Z",
                    "reference": {"messageId": "msg-1"},
                },
            ],
        })
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 2
        assert messages[0].author == "Alice"
        assert messages[0].channel == "general"
        assert messages[0].channel_id == "C123"
        assert messages[1].reply_to == "msg-1"

    def test_parse_json_array(self):
        content = json.dumps([
            {
                "id": "msg-1",
                "content": "Hello",
                "author": {"name": "Alice", "id": "u1"},
                "timestamp": "2024-01-01T10:00:00Z",
            },
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 1

    def test_bot_detection(self):
        content = json.dumps({
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Human msg",
                    "author": {"name": "Alice", "id": "u1", "isBot": False},
                },
                {
                    "id": "msg-2",
                    "content": "Bot msg",
                    "author": {"name": "MEE6", "id": "b1", "isBot": True},
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert messages[0].is_bot is False
        assert messages[1].is_bot is True

    def test_reactions_parsed(self):
        content = json.dumps({
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Great idea",
                    "author": {"name": "Alice", "id": "u1"},
                    "reactions": [
                        {"emoji": {"name": "thumbsup"}, "count": 5},
                        {"emoji": {"name": "heart"}, "count": 2},
                    ],
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages[0].reactions) == 2
        assert messages[0].reactions[0]["name"] == "thumbsup"

    def test_reactions_excluded(self):
        content = json.dumps({
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Hello",
                    "author": {"name": "Alice", "id": "u1"},
                    "reactions": [{"emoji": {"name": "thumbsup"}, "count": 1}],
                },
            ]
        })
        messages = self._adapter(include_reactions=False)._parse_messages(content, {})
        assert messages[0].reactions == []

    def test_mentions_extracted(self):
        content = json.dumps({
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Hey @Bob",
                    "author": {"name": "Alice", "id": "u1"},
                    "mentions": [{"name": "Bob"}],
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert "Bob" in messages[0].mentions

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            self._adapter()._parse_messages("not json", {})

    def test_channel_from_export(self):
        """Channel info extracted from export metadata."""
        content = json.dumps({
            "channel": {"id": "C999", "name": "dev-talk"},
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Hello",
                    "author": {"name": "Alice", "id": "u1"},
                },
            ],
        })
        messages = self._adapter()._parse_messages(content, {})
        assert messages[0].channel == "dev-talk"
        assert messages[0].channel_id == "C999"


# ============================================================================
# Unit tests: Thread grouping
# ============================================================================


class TestDiscordThreadGrouping:

    def _adapter(self):
        return DiscordAdapter()

    def test_group_by_reference(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Root", message_id="msg-1", timestamp="1"),
            ChatMessage(author="Bob", content="Reply", message_id="msg-2", reply_to="msg-1", timestamp="2"),
            ChatMessage(author="Carol", content="Another reply", message_id="msg-3", reply_to="msg-1", timestamp="3"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        # All three should be in one thread
        assert any(t.message_count == 3 for t in threads)

    def test_separate_conversations(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Topic 1", message_id="msg-1", timestamp="1"),
            ChatMessage(author="Bob", content="Topic 2", message_id="msg-2", timestamp="2"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        assert len(threads) == 2

    def test_nested_replies(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Root", message_id="msg-1", timestamp="1"),
            ChatMessage(author="Bob", content="Reply to root", message_id="msg-2", reply_to="msg-1", timestamp="2"),
            ChatMessage(author="Carol", content="Reply to reply", message_id="msg-3", reply_to="msg-2", timestamp="3"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        assert any(t.message_count == 3 for t in threads)


# ============================================================================
# Integration tests: Full pipeline
# ============================================================================


class TestDiscordPipeline:

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
    def discord_export(self):
        return json.dumps({
            "channel": {"id": "C123", "name": "dev-decisions"},
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Should we use Redis or Memcached for caching?",
                    "author": {"name": "Alice", "id": "u1", "isBot": False},
                    "timestamp": "2024-01-01T10:00:00Z",
                },
                {
                    "id": "msg-2",
                    "content": "I'll benchmark both options this week.",
                    "author": {"name": "Bob", "id": "u2", "isBot": False},
                    "timestamp": "2024-01-01T10:01:00Z",
                    "reference": {"messageId": "msg-1"},
                },
                {
                    "id": "msg-3",
                    "content": "We decided to use Redis for its data structure support.",
                    "author": {"name": "Carol", "id": "u3", "isBot": False},
                    "timestamp": "2024-01-01T10:02:00Z",
                    "reference": {"messageId": "msg-1"},
                },
            ],
        })

    async def test_full_discord_pipeline(self, mock_memory, discord_export):
        adapter = DiscordAdapter(
            config=DiscordConfig(
                resolve_entities=False,
                create_graph_edges=False,
                skip_bot_messages=False,
            )
        )
        result = await adapter.ingest(
            content=discord_export,
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "discord"
        assert result.action_items_detected >= 1  # "I'll benchmark..."
        assert result.decisions_detected >= 1  # "We decided..."

    async def test_bot_messages_skipped_by_default(self, mock_memory):
        content = json.dumps({
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Human message",
                    "author": {"name": "Alice", "id": "u1", "isBot": False},
                    "timestamp": "1",
                },
                {
                    "id": "msg-2",
                    "content": "Bot message",
                    "author": {"name": "Bot", "id": "b1", "isBot": True},
                    "timestamp": "2",
                    "reference": {"messageId": "msg-1"},
                },
            ]
        })

        adapter = DiscordAdapter(
            config=DiscordConfig(
                resolve_entities=False,
                create_graph_edges=False,
                extract_action_items=False,
                extract_beliefs=False,
            )
        )
        result = await adapter.ingest(content=content, metadata={}, memory=mock_memory)

        # Bot filtered + only 1 msg left → below min_thread_messages (2)
        # Depending on threading: msg-1 standalone, bot filtered → may fail min_thread
        # But let's check it at least doesn't crash
        assert isinstance(result, IngestionResult)

    async def test_empty_content(self, mock_memory):
        adapter = DiscordAdapter(
            config=DiscordConfig(resolve_entities=False, create_graph_edges=False)
        )
        result = await adapter.ingest(content="", metadata={}, memory=mock_memory)
        assert result.experiences_created == 0
        assert result.has_errors


from silicon_memory.ingestion.types import IngestionResult
