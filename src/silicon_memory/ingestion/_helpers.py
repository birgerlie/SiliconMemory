"""Shared helpers for ingestion adapters."""

from __future__ import annotations

import json
import re
from typing import Any


# Shared action item regex patterns used by all adapters.
ACTION_ITEM_PATTERNS: list[re.Pattern[str]] = [
    # "ACTION: do something" or "TODO: do something" or "FIXME: ..."
    re.compile(r"(?:ACTION|TODO|FIXME)\s*:\s*(.+)", re.IGNORECASE),
    # "[Name] will/should/needs to [verb] ..."
    re.compile(
        r"([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)?)\s+"
        r"(?:will|should|needs to|is going to|has to)\s+(.+)",
    ),
]


def parse_llm_json_array(response: str) -> list[Any]:
    """Extract and parse a JSON array from LLM response text.

    Searches for the first ``[...]`` block in the response and
    parses it as JSON.

    Raises:
        ValueError: If no JSON array is found in the response.
        json.JSONDecodeError: If the matched text is not valid JSON.
    """
    json_match = re.search(r"\[[\s\S]*\]", response)
    if not json_match:
        raise ValueError("No JSON array found in response")
    return json.loads(json_match.group())


def extract_action_items_from_text(
    text: str,
    index_key: str,
    index_value: int,
    extra_patterns: list[re.Pattern[str]] | None = None,
) -> list[dict[str, Any]]:
    """Extract action items from a block of text using regex patterns.

    Args:
        text: The text to scan (newline-delimited lines).
        index_key: Key name for the source index (e.g. "segment_index").
        index_value: Value for the source index.
        extra_patterns: Additional patterns prepended before the defaults.
            Single-group patterns produce ``{action, owner=None}``.
            Two-group patterns produce ``{action=group2, owner=group1}``.

    Returns:
        List of ``{action, owner, <index_key>}`` dicts.
    """
    patterns = (extra_patterns or []) + ACTION_ITEM_PATTERNS
    action_items: list[dict[str, Any]] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    action_items.append({
                        "action": groups[0].strip(),
                        "owner": None,
                        index_key: index_value,
                    })
                elif len(groups) == 2:
                    action_items.append({
                        "action": groups[1].strip(),
                        "owner": groups[0].strip(),
                        index_key: index_value,
                    })
                break  # Only match first pattern per line

    return action_items
