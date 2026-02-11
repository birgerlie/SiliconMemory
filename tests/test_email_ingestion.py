"""Tests for email ingestion adapter."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from silicon_memory.ingestion.types import (
    IngestionAdapter,
    IngestionConfig,
    IngestionResult,
)
from silicon_memory.ingestion.email import (
    EmailAdapter,
    EmailConfig,
    EmailMessage,
)
from silicon_memory.core.types import Experience


# ============================================================================
# Unit tests: EmailMessage dataclass
# ============================================================================


class TestEmailMessage:
    """Unit tests for EmailMessage."""

    def test_defaults(self):
        msg = EmailMessage()
        assert msg.from_addr == ""
        assert msg.to_addrs == []
        assert msg.cc_addrs == []
        assert msg.subject == ""
        assert msg.body == ""
        assert msg.date is None
        assert msg.message_id == ""
        assert msg.in_reply_to is None
        assert msg.thread_id is None
        assert msg.attachments == []

    def test_construction(self):
        msg = EmailMessage(
            from_addr="alice@example.com",
            to_addrs=["bob@example.com"],
            subject="Test",
            body="Hello",
            message_id="<123@example.com>",
        )
        assert msg.from_addr == "alice@example.com"
        assert msg.to_addrs == ["bob@example.com"]
        assert msg.subject == "Test"
        assert msg.body == "Hello"
        assert msg.message_id == "<123@example.com>"


# ============================================================================
# Unit tests: EmailConfig
# ============================================================================


class TestEmailConfig:
    """Unit tests for EmailConfig."""

    def test_defaults(self):
        config = EmailConfig()
        assert config.extract_action_items is True
        assert config.resolve_entities is True
        assert config.parse_threads is True
        assert config.max_thread_depth == 20
        # Inherited from IngestionConfig
        assert config.max_segment_length == 2000
        assert config.min_segment_length == 50

    def test_inherits_from_ingestion_config(self):
        assert issubclass(EmailConfig, IngestionConfig)


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestEmailProtocol:
    """Test that EmailAdapter satisfies IngestionAdapter protocol."""

    def test_protocol_satisfaction(self):
        adapter = EmailAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        adapter = EmailAdapter()
        assert adapter.source_type == "email"


# ============================================================================
# Unit tests: Raw email parsing
# ============================================================================


class TestRawEmailParsing:
    """Test parsing of RFC 2822 raw email text."""

    def _adapter(self):
        return EmailAdapter()

    def test_parse_basic_headers(self):
        """Test parsing From/To/Subject/Date/Body from RFC 2822 text."""
        raw = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Project Update\n"
            "Date: Mon, 1 Jan 2024 10:00:00 +0000\n"
            "Message-ID: <msg-1@example.com>\n"
            "\n"
            "Hi Bob,\n"
            "\n"
            "Here is the update on the project.\n"
        )
        msg = self._adapter()._parse_raw_email(raw)
        assert msg.from_addr == "alice@example.com"
        assert "bob@example.com" in msg.to_addrs
        assert msg.subject == "Project Update"
        assert msg.date == "Mon, 1 Jan 2024 10:00:00 +0000"
        assert msg.message_id == "<msg-1@example.com>"
        assert "update on the project" in msg.body

    def test_parse_cc_header(self):
        """Test parsing CC addresses."""
        raw = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Cc: charlie@example.com, dave@example.com\n"
            "Subject: Team Update\n"
            "\n"
            "Body text.\n"
        )
        msg = self._adapter()._parse_raw_email(raw)
        assert len(msg.cc_addrs) == 2
        assert "charlie@example.com" in msg.cc_addrs

    def test_parse_multiple_to(self):
        """Test parsing multiple To addresses."""
        raw = (
            "From: alice@example.com\n"
            "To: bob@example.com, charlie@example.com\n"
            "Subject: Group\n"
            "\n"
            "Hello all.\n"
        )
        msg = self._adapter()._parse_raw_email(raw)
        assert len(msg.to_addrs) == 2

    def test_parse_in_reply_to(self):
        """Test parsing In-Reply-To header."""
        raw = (
            "From: bob@example.com\n"
            "To: alice@example.com\n"
            "Subject: Re: Project Update\n"
            "In-Reply-To: <msg-1@example.com>\n"
            "\n"
            "Thanks Alice.\n"
        )
        msg = self._adapter()._parse_raw_email(raw)
        assert msg.in_reply_to == "<msg-1@example.com>"

    def test_strip_html_body(self):
        """Test that HTML tags are stripped from body."""
        raw = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: HTML Email\n"
            "\n"
            "<html><body><p>Hello <b>Bob</b></p></body></html>\n"
        )
        msg = self._adapter()._parse_raw_email(raw)
        assert "<" not in msg.body
        assert "Hello" in msg.body
        assert "Bob" in msg.body


# ============================================================================
# Unit tests: Dict email parsing
# ============================================================================


class TestDictEmailParsing:
    """Test parsing from pre-parsed dict."""

    def _adapter(self):
        return EmailAdapter()

    def test_parse_basic_dict(self):
        data = {
            "from": "alice@example.com",
            "to": ["bob@example.com"],
            "subject": "Update",
            "body": "Hello Bob.",
            "message_id": "msg-1",
        }
        msg = self._adapter()._parse_dict_email(data)
        assert msg.from_addr == "alice@example.com"
        assert msg.to_addrs == ["bob@example.com"]
        assert msg.subject == "Update"
        assert msg.body == "Hello Bob."

    def test_parse_string_to_field(self):
        """Test that comma-separated string 'to' is split."""
        data = {
            "from": "alice@example.com",
            "to": "bob@example.com, charlie@example.com",
            "subject": "Group",
            "body": "Hi all.",
        }
        msg = self._adapter()._parse_dict_email(data)
        assert len(msg.to_addrs) == 2

    def test_parse_from_addr_key(self):
        """Test that both 'from' and 'from_addr' keys work."""
        data1 = {"from": "alice@example.com", "body": "test"}
        data2 = {"from_addr": "alice@example.com", "body": "test"}
        assert self._adapter()._parse_dict_email(data1).from_addr == "alice@example.com"
        assert self._adapter()._parse_dict_email(data2).from_addr == "alice@example.com"


# ============================================================================
# Unit tests: Auto-detect format
# ============================================================================


class TestAutoDetect:
    """Test auto-detection of email format."""

    def _adapter(self):
        return EmailAdapter()

    def test_detect_raw_string(self):
        raw = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Test\n"
            "\n"
            "Body text.\n"
        )
        messages = self._adapter()._detect_and_parse(raw)
        assert len(messages) >= 1
        assert messages[0].from_addr == "alice@example.com"

    def test_detect_dict_input(self):
        data = {"from": "alice@example.com", "body": "Hello", "subject": "Test"}
        messages = self._adapter()._detect_and_parse(data)
        assert len(messages) == 1
        assert messages[0].from_addr == "alice@example.com"

    def test_detect_json_string(self):
        import json
        data = {"from": "alice@example.com", "body": "Hello", "subject": "Test"}
        messages = self._adapter()._detect_and_parse(json.dumps(data))
        assert len(messages) == 1
        assert messages[0].from_addr == "alice@example.com"

    def test_detect_json_array(self):
        import json
        data = [
            {"from": "alice@example.com", "body": "First", "subject": "Test"},
            {"from": "bob@example.com", "body": "Second", "subject": "Re: Test"},
        ]
        messages = self._adapter()._detect_and_parse(json.dumps(data))
        assert len(messages) == 2

    def test_detect_plain_body_fallback(self):
        """Plain text without headers falls back to body-only message."""
        messages = self._adapter()._detect_and_parse("Just a plain text message.")
        assert len(messages) == 1
        assert messages[0].body == "Just a plain text message."


# ============================================================================
# Unit tests: Thread splitting
# ============================================================================


class TestThreadSplitting:
    """Test reply chain and thread splitting."""

    def _adapter(self):
        return EmailAdapter()

    def test_on_wrote_chain(self):
        """Test splitting on 'On ... wrote:' patterns."""
        body = (
            "Thanks for the update.\n"
            "\n"
            "On Mon, Jan 1, Alice wrote:\n"
            "> Original message here.\n"
            "> More original text.\n"
        )
        parts = self._adapter()._split_thread(body)
        assert len(parts) == 2
        assert "Thanks for the update" in parts[0]["body"]
        assert "Original message here" in parts[1]["body"]
        assert "Alice" in parts[1]["author"]

    def test_quoted_reply(self):
        """Test that '>' quoting is stripped from replies."""
        body = (
            "My reply.\n"
            "\n"
            "On Jan 1, Bob wrote:\n"
            "> This is quoted.\n"
            "> More quoted text.\n"
        )
        parts = self._adapter()._split_thread(body)
        assert len(parts) == 2
        # Quoting should be stripped
        assert ">" not in parts[1]["body"]
        assert "This is quoted" in parts[1]["body"]

    def test_forwarded_message(self):
        """Test splitting on forwarded message markers."""
        body = (
            "FYI, see below.\n"
            "\n"
            "---------- Forwarded message ----------\n"
            "Original forwarded content here.\n"
        )
        parts = self._adapter()._split_thread(body)
        assert len(parts) == 2
        assert "FYI" in parts[0]["body"]
        assert "Original forwarded content" in parts[1]["body"]

    def test_single_email_no_thread(self):
        """Test that a single email without thread markers returns one part."""
        body = "Just a simple email with no replies."
        parts = self._adapter()._split_thread(body)
        assert len(parts) == 1
        assert parts[0]["body"] == body


# ============================================================================
# Unit tests: Action item extraction
# ============================================================================


class TestEmailActionItemExtraction:
    """Test heuristic action item extraction from emails."""

    def _adapter(self):
        return EmailAdapter()

    def test_please_pattern(self):
        """Test 'Please [verb]' extraction."""
        messages = [
            EmailMessage(body="Please review the attached document.", message_id="m1"),
        ]
        items = self._adapter()._extract_action_items_heuristic(messages)
        assert len(items) >= 1
        assert any("review" in item["action"].lower() for item in items)

    def test_can_you_pattern(self):
        """Test 'Can you [verb]' extraction."""
        messages = [
            EmailMessage(body="Can you send me the report by Friday?", message_id="m1"),
        ]
        items = self._adapter()._extract_action_items_heuristic(messages)
        assert len(items) >= 1
        assert any("send" in item["action"].lower() for item in items)

    def test_todo_pattern(self):
        """Test 'TODO:' extraction."""
        messages = [
            EmailMessage(body="TODO: Update the deployment scripts.", message_id="m1"),
        ]
        items = self._adapter()._extract_action_items_heuristic(messages)
        assert len(items) >= 1
        assert any("Update" in item["action"] for item in items)

    def test_name_will_pattern(self):
        """Test '[Name] will' extraction."""
        messages = [
            EmailMessage(body="Alice will prepare the presentation.", message_id="m1"),
        ]
        items = self._adapter()._extract_action_items_heuristic(messages)
        assert len(items) >= 1
        assert any(item.get("owner") == "Alice" for item in items)

    def test_no_action_items(self):
        """Test no extraction when no patterns match."""
        messages = [
            EmailMessage(body="The meeting went well.", message_id="m1"),
        ]
        items = self._adapter()._extract_action_items_heuristic(messages)
        assert items == []


# ============================================================================
# Integration tests: Full pipeline
# ============================================================================


class TestEmailIngestionPipeline:
    """Integration tests for the full email ingestion pipeline."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.record_experience = AsyncMock()
        return memory

    async def test_full_pipeline_raw_email(self, mock_memory):
        """Test full ingestion of a raw RFC 2822 email."""
        raw_email = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Project Update\n"
            "Date: Mon, 1 Jan 2024 10:00:00 +0000\n"
            "Message-ID: <msg-1@example.com>\n"
            "\n"
            "Hi Bob,\n"
            "\n"
            "The project is on track. TODO: Review the timeline.\n"
        )

        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=raw_email,
            metadata={"thread_id": "t-1"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "email"
        assert result.action_items_detected >= 1
        assert not result.has_errors or result.success

    async def test_thread_pipeline(self, mock_memory):
        """Test ingestion of email thread produces one experience per message."""
        raw_email = (
            "From: bob@example.com\n"
            "To: alice@example.com\n"
            "Subject: Re: Project Update\n"
            "In-Reply-To: <msg-1@example.com>\n"
            "\n"
            "Thanks Alice, looks good.\n"
            "\n"
            "On Mon, Jan 1, Alice wrote:\n"
            "> Hi Bob,\n"
            "> The project is on track.\n"
        )

        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=raw_email,
            metadata={"thread_id": "t-1"},
            memory=mock_memory,
        )

        assert result.experiences_created == 2
        assert mock_memory.record_experience.await_count == 2

    async def test_dict_input_pipeline(self, mock_memory):
        """Test ingestion from pre-parsed dict."""
        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        import json
        data = {
            "from": "alice@example.com",
            "to": ["bob@example.com"],
            "subject": "Quick Note",
            "body": "Please check the logs.",
            "message_id": "msg-2",
        }
        result = await adapter.ingest(
            content=json.dumps(data),
            metadata={"thread_id": "t-2"},
            memory=mock_memory,
        )

        assert result.experiences_created == 1

    async def test_experience_context_metadata(self, mock_memory):
        """Test that experiences include correct email context metadata."""
        raw_email = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Meeting Notes\n"
            "\n"
            "Here are the notes from today's meeting.\n"
        )

        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        await adapter.ingest(
            content=raw_email,
            metadata={"thread_id": "t-3"},
            memory=mock_memory,
        )

        call_args = mock_memory.record_experience.call_args_list
        assert len(call_args) >= 1
        exp = call_args[0][0][0]
        assert exp.context["source_type"] == "email"
        assert exp.context["subject"] == "Meeting Notes"
        assert exp.context["from"] == "alice@example.com"
        assert exp.session_id == "t-3"
        assert exp.user_id == "user-1"
        assert exp.tenant_id == "acme"

    async def test_llm_action_items(self, mock_memory):
        """Test action item extraction with LLM mock."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value='[{"action": "Review docs", "owner": "Bob", "message_index": 0}]'
        )

        raw_email = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Review Request\n"
            "\n"
            "Bob, please review the docs by Friday.\n"
        )

        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=raw_email,
            metadata={"thread_id": "t-4"},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        assert result.action_items_detected >= 1
        mock_llm.complete.assert_awaited()

    async def test_entity_resolution(self, mock_memory):
        """Test sender/recipient entity resolution."""
        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(
            return_value=MagicMock(resolved=[MagicMock()])
        )

        raw_email = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Test\n"
            "\n"
            "Hello.\n"
        )

        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=True),
            entity_resolver=mock_resolver,
        )
        result = await adapter.ingest(
            content=raw_email,
            metadata={"thread_id": "t-5"},
            memory=mock_memory,
        )

        assert result.entities_resolved >= 1
        assert mock_resolver.resolve.await_count >= 1

    async def test_empty_content(self, mock_memory):
        """Test graceful handling of empty input."""
        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content="",
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors

    async def test_malformed_input(self, mock_memory):
        """Test graceful handling of bytes input."""
        adapter = EmailAdapter(
            config=EmailConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=b"From: test@example.com\nTo: bob@example.com\nSubject: Bytes\n\nHello from bytes.",
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
