"""Shared helpers for ingestion adapters."""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from silicon_memory.core.types import Experience
    from silicon_memory.ingestion.types import IngestionConfig, IngestionResult
    from silicon_memory.memory.silicondb_router import SiliconMemory


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


async def persist_experiences(
    *,
    memory: "SiliconMemory",
    experiences: list["Experience"],
    result: "IngestionResult",
    config: "IngestionConfig",
    item_label: str,
) -> list[str]:
    """Persist experiences with optional batch+visibility barrier.

    Falls back to per-item writes when the memory implementation does not
    expose a concrete class-level batch API (e.g., mocked memory objects).
    """
    if not experiences:
        return []

    use_batch = bool(getattr(config, "ingest_use_batch_api", True))
    batch_method = getattr(type(memory), "ingest_experiences_batch", None)
    if use_batch and callable(batch_method):
        batch_result = await memory.ingest_experiences_batch(
            experiences,
            wait_for_visibility=bool(getattr(config, "ingest_wait_for_visibility", True)),
            timeout_s=float(getattr(config, "ingest_visibility_timeout_s", 5.0)),
            poll_interval_s=float(getattr(config, "ingest_visibility_poll_s", 0.1)),
        )
        created_ids = list(batch_result.receipt.accepted_experience_ids)
        result.experiences_created += len(created_ids)
        result.details["ingest_batch_state"] = batch_result.state
        result.details["ingest_batch_failed_count"] = int(batch_result.receipt.failed_count)
        if batch_result.visibility_ok is not None:
            result.details["ingest_visibility_ok"] = bool(batch_result.visibility_ok)
        for err in batch_result.receipt.errors:
            result.errors.append(f"Failed to store {item_label}: {err}")
        if created_ids:
            result.details["experience_ids"] = created_ids
        return created_ids

    created_ids: list[str] = []
    for i, exp in enumerate(experiences):
        try:
            await memory.record_experience(exp)
            result.experiences_created += 1
            created_ids.append(str(exp.id))
        except Exception as e:  # noqa: PERF203
            result.errors.append(f"Failed to store {item_label} {i}: {e}")
    if created_ids:
        result.details["experience_ids"] = created_ids
    return created_ids
