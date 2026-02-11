"""Tests for Microsoft Teams ingestion adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from silicon_memory.ingestion.types import IngestionAdapter
from silicon_memory.ingestion.chat import ChatConfig
from silicon_memory.ingestion.teams import TeamsAdapter, TeamsConfig


# ============================================================================
# Unit tests: TeamsConfig
# ============================================================================


class TestTeamsConfig:
    def test_defaults(self):
        config = TeamsConfig()
        assert config.importance_boost == {"high": 0.2, "urgent": 0.3}
        assert config.include_html_body is False

    def test_inherits_from_chat_config(self):
        assert issubclass(TeamsConfig, ChatConfig)


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestTeamsProtocol:
    def test_protocol_satisfaction(self):
        adapter = TeamsAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        adapter = TeamsAdapter()
        assert adapter.source_type == "teams"


# ============================================================================
# Unit tests: Message parsing
# ============================================================================


class TestTeamsMessageParsing:

    def _adapter(self, **kwargs):
        return TeamsAdapter(config=TeamsConfig(**kwargs))

    def test_parse_graph_api_format(self):
        content = json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {"contentType": "text", "content": "Hello team"},
                    "from": {"user": {"displayName": "Alice", "id": "u1"}},
                    "createdDateTime": "2024-01-01T10:00:00Z",
                },
                {
                    "id": "msg-2",
                    "body": {"contentType": "text", "content": "Hi Alice"},
                    "from": {"user": {"displayName": "Bob", "id": "u2"}},
                    "createdDateTime": "2024-01-01T10:01:00Z",
                    "replyToId": "msg-1",
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 2
        assert messages[0].author == "Alice"
        assert messages[0].content == "Hello team"
        assert messages[1].reply_to == "msg-1"

    def test_parse_json_array(self):
        content = json.dumps([
            {
                "id": "msg-1",
                "body": {"contentType": "text", "content": "Hello"},
                "from": {"user": {"displayName": "Alice", "id": "u1"}},
            },
        ])
        messages = self._adapter()._parse_messages(content, {})
        assert len(messages) == 1

    def test_html_body_stripped(self):
        content = json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {
                        "contentType": "html",
                        "content": "<p>Hello <b>team</b></p>",
                    },
                    "from": {"user": {"displayName": "Alice", "id": "u1"}},
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert "<" not in messages[0].content
        assert "Hello" in messages[0].content
        assert "team" in messages[0].content

    def test_html_body_preserved_when_configured(self):
        content = json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {
                        "contentType": "html",
                        "content": "<p>Hello <b>team</b></p>",
                    },
                    "from": {"user": {"displayName": "Alice", "id": "u1"}},
                },
            ]
        })
        messages = self._adapter(include_html_body=True)._parse_messages(content, {})
        assert "<p>" in messages[0].content

    def test_importance_in_metadata(self):
        content = json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {"contentType": "text", "content": "Urgent!"},
                    "from": {"user": {"displayName": "Alice", "id": "u1"}},
                    "importance": "urgent",
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert messages[0].platform_metadata["importance"] == "urgent"

    def test_mentions_extracted(self):
        content = json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {"contentType": "text", "content": "Hey @Bob"},
                    "from": {"user": {"displayName": "Alice", "id": "u1"}},
                    "mentions": [
                        {"mentioned": {"user": {"displayName": "Bob"}}},
                    ],
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert "Bob" in messages[0].mentions

    def test_bot_detection(self):
        content = json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {"contentType": "text", "content": "Bot message"},
                    "from": {"application": {"displayName": "BotApp"}},
                },
            ]
        })
        messages = self._adapter()._parse_messages(content, {})
        assert messages[0].is_bot is True

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            self._adapter()._parse_messages("not json", {})


# ============================================================================
# Unit tests: Thread grouping
# ============================================================================


class TestTeamsThreadGrouping:

    def _adapter(self):
        return TeamsAdapter()

    def test_group_by_reply_to_id(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Root msg", message_id="msg-1", timestamp="1"),
            ChatMessage(author="Bob", content="Reply", message_id="msg-2", reply_to="msg-1", timestamp="2"),
            ChatMessage(author="Carol", content="Another reply", message_id="msg-3", reply_to="msg-1", timestamp="3"),
        ]
        threads = self._adapter()._group_into_threads(messages)

        # All three should be in one thread
        assert any(t.message_count == 3 for t in threads)

    def test_separate_threads(self):
        from silicon_memory.ingestion.chat import ChatMessage

        messages = [
            ChatMessage(author="Alice", content="Thread 1", message_id="msg-1", timestamp="1"),
            ChatMessage(author="Bob", content="Thread 2", message_id="msg-2", timestamp="2"),
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

        # All should be in one thread
        assert any(t.message_count == 3 for t in threads)


# ============================================================================
# Integration tests: Full pipeline
# ============================================================================


class TestTeamsPipeline:

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
    def teams_export(self):
        return json.dumps({
            "value": [
                {
                    "id": "msg-1",
                    "body": {"contentType": "text", "content": "We need to decide on the deployment strategy."},
                    "from": {"user": {"displayName": "Alice", "id": "u1"}},
                    "createdDateTime": "2024-01-01T10:00:00Z",
                },
                {
                    "id": "msg-2",
                    "body": {"contentType": "text", "content": "I'll prepare the Kubernetes manifests by Monday."},
                    "from": {"user": {"displayName": "Bob", "id": "u2"}},
                    "createdDateTime": "2024-01-01T10:01:00Z",
                    "replyToId": "msg-1",
                },
                {
                    "id": "msg-3",
                    "body": {"contentType": "text", "content": "Agreed, let's go with Kubernetes for prod."},
                    "from": {"user": {"displayName": "Carol", "id": "u3"}},
                    "createdDateTime": "2024-01-01T10:02:00Z",
                    "replyToId": "msg-1",
                },
            ]
        })

    async def test_full_teams_pipeline(self, mock_memory, teams_export):
        adapter = TeamsAdapter(
            config=TeamsConfig(
                resolve_entities=False,
                create_graph_edges=False,
            )
        )
        result = await adapter.ingest(
            content=teams_export,
            metadata={"channel": "deployment"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "teams"
        assert result.action_items_detected >= 1  # "I'll prepare..."
        assert result.decisions_detected >= 1  # "Agreed, let's go with..."

    async def test_empty_content(self, mock_memory):
        adapter = TeamsAdapter(
            config=TeamsConfig(resolve_entities=False, create_graph_edges=False)
        )
        result = await adapter.ingest(content="", metadata={}, memory=mock_memory)
        assert result.experiences_created == 0
        assert result.has_errors
