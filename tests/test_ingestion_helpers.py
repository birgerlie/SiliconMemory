"""Tests for shared ingestion helper functions."""

from __future__ import annotations

import json
import re

import pytest

from silicon_memory.ingestion._helpers import (
    ACTION_ITEM_PATTERNS,
    extract_action_items_from_text,
    parse_llm_json_array,
)


# ============================================================================
# Unit tests: parse_llm_json_array
# ============================================================================


class TestParseLlmJsonArray:
    """Test LLM response JSON array extraction."""

    def test_plain_json_array(self):
        response = '[{"action": "Deploy", "owner": "Alice"}]'
        result = parse_llm_json_array(response)
        assert len(result) == 1
        assert result[0]["action"] == "Deploy"

    def test_json_embedded_in_text(self):
        response = (
            "Here are the action items:\n"
            '[{"action": "Review PR", "owner": "Bob"}]\n'
            "Let me know if you need more."
        )
        result = parse_llm_json_array(response)
        assert len(result) == 1
        assert result[0]["action"] == "Review PR"

    def test_empty_array(self):
        result = parse_llm_json_array("[]")
        assert result == []

    def test_multi_element_array(self):
        response = '[{"a": 1}, {"a": 2}, {"a": 3}]'
        result = parse_llm_json_array(response)
        assert len(result) == 3

    def test_no_json_array_raises(self):
        with pytest.raises(ValueError, match="No JSON array"):
            parse_llm_json_array("No JSON here, just text.")

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json_array("[{invalid json}]")

    def test_multiline_json(self):
        response = '[\n  {"action": "test"},\n  {"action": "deploy"}\n]'
        result = parse_llm_json_array(response)
        assert len(result) == 2


# ============================================================================
# Unit tests: ACTION_ITEM_PATTERNS
# ============================================================================


class TestActionItemPatterns:
    """Test that the shared patterns list is well-formed."""

    def test_patterns_exist(self):
        assert len(ACTION_ITEM_PATTERNS) >= 2

    def test_patterns_are_compiled_regex(self):
        for p in ACTION_ITEM_PATTERNS:
            assert isinstance(p, re.Pattern)

    def test_keyword_pattern_matches(self):
        """ACTION:/TODO:/FIXME: should match."""
        keyword_pattern = ACTION_ITEM_PATTERNS[0]
        assert keyword_pattern.search("ACTION: Deploy by Friday")
        assert keyword_pattern.search("TODO: Update docs")
        assert keyword_pattern.search("FIXME: Fix the bug")
        assert keyword_pattern.search("action: lowercase too")

    def test_name_will_pattern_matches(self):
        """[Name] will/should/needs to should match."""
        name_pattern = ACTION_ITEM_PATTERNS[1]
        assert name_pattern.search("Alice will review the PR")
        assert name_pattern.search("Bob should update the tests")
        assert name_pattern.search("Charlie needs to deploy")
        assert name_pattern.search("Dave is going to fix it")
        assert name_pattern.search("Eve has to finish the report")

    def test_name_will_pattern_captures_owner(self):
        name_pattern = ACTION_ITEM_PATTERNS[1]
        match = name_pattern.search("Alice will review the PR by EOD")
        assert match.group(1) == "Alice"
        assert match.group(2) == "review the PR by EOD"


# ============================================================================
# Unit tests: extract_action_items_from_text
# ============================================================================


class TestExtractActionItemsFromText:
    """Test the shared action item extraction function."""

    def test_action_keyword(self):
        text = "We discussed the plan.\nACTION: Deploy by Friday."
        items = extract_action_items_from_text(text, "seg", 0)
        assert len(items) == 1
        assert items[0]["action"] == "Deploy by Friday."
        assert items[0]["owner"] is None
        assert items[0]["seg"] == 0

    def test_todo_keyword(self):
        text = "TODO: Update the docs."
        items = extract_action_items_from_text(text, "seg", 2)
        assert len(items) == 1
        assert "Update" in items[0]["action"]
        assert items[0]["seg"] == 2

    def test_name_will_pattern(self):
        text = "Alice will review the PR.\nBob should update tests."
        items = extract_action_items_from_text(text, "idx", 0)
        assert len(items) == 2
        assert items[0]["owner"] == "Alice"
        assert items[1]["owner"] == "Bob"

    def test_no_matches(self):
        text = "The meeting went well. Everyone agreed."
        items = extract_action_items_from_text(text, "seg", 0)
        assert items == []

    def test_empty_text(self):
        items = extract_action_items_from_text("", "seg", 0)
        assert items == []

    def test_blank_lines_skipped(self):
        text = "\n\n\nACTION: Do something.\n\n\n"
        items = extract_action_items_from_text(text, "seg", 0)
        assert len(items) == 1

    def test_first_pattern_wins_per_line(self):
        """Only the first matching pattern should fire per line."""
        text = "TODO: Alice will review the PR"
        items = extract_action_items_from_text(text, "seg", 0)
        assert len(items) == 1
        # TODO: pattern should match first (it's first in the list)
        assert items[0]["owner"] is None
        assert "Alice will review the PR" in items[0]["action"]

    def test_custom_index_key(self):
        text = "ACTION: Deploy."
        items = extract_action_items_from_text(text, "message_index", 5)
        assert items[0]["message_index"] == 5

    def test_extra_patterns_prepended(self):
        """Extra patterns should be checked before the defaults."""
        please_pattern = re.compile(
            r"(?:please|can you)\s+(.+?)(?:\.|$)", re.IGNORECASE
        )
        text = "Please review the attached document."
        items = extract_action_items_from_text(
            text, "idx", 0, extra_patterns=[please_pattern]
        )
        assert len(items) == 1
        assert "review" in items[0]["action"].lower()

    def test_extra_patterns_dont_suppress_defaults(self):
        """Default patterns should still work alongside extra ones."""
        extra = re.compile(r"(?:please)\s+(.+?)(?:\.|$)", re.IGNORECASE)
        text = "Please check.\nTODO: Fix bug.\nAlice will deploy."
        items = extract_action_items_from_text(
            text, "idx", 0, extra_patterns=[extra]
        )
        assert len(items) == 3
