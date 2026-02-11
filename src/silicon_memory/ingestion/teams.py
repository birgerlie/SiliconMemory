"""Microsoft Teams chat ingestion adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from silicon_memory.ingestion.chat import (
    BaseChatAdapter,
    ChatConfig,
    ChatMessage,
    ChatThread,
)

if TYPE_CHECKING:
    from silicon_memory.entities.resolver import EntityResolver


@dataclass
class TeamsConfig(ChatConfig):
    """Configuration for Microsoft Teams ingestion."""

    importance_boost: dict[str, float] = field(
        default_factory=lambda: {"high": 0.2, "urgent": 0.3}
    )
    include_html_body: bool = False


class TeamsAdapter(BaseChatAdapter):
    """Ingests Microsoft Teams messages as experiences.

    Input format: Microsoft Graph API JSON ``{"value": [...]}``.

    Thread grouping uses ``replyToId`` to build reply trees.
    """

    def __init__(
        self,
        config: TeamsConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        super().__init__(config=config or TeamsConfig(), entity_resolver=entity_resolver)

    @property
    def _teams_config(self) -> TeamsConfig:
        return self._config  # type: ignore[return-value]

    @property
    def source_type(self) -> str:
        return "teams"

    def _parse_messages(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> list[ChatMessage]:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if isinstance(data, dict):
            raw_messages = data.get("value", data.get("messages", []))
        elif isinstance(data, list):
            raw_messages = data
        else:
            raise ValueError("Expected JSON object with 'value' key or array")

        messages: list[ChatMessage] = []

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            # Extract body text
            body = msg.get("body", {})
            body_content = ""
            if isinstance(body, dict):
                content_type = body.get("contentType", "text")
                body_content = body.get("content", "")
                if content_type == "html" and not self._teams_config.include_html_body:
                    body_content = _strip_html(body_content)
            elif isinstance(body, str):
                body_content = body

            # Extract author
            from_field = msg.get("from", {})
            user_field = from_field.get("user", {}) if isinstance(from_field, dict) else {}
            author = user_field.get("displayName", "") if isinstance(user_field, dict) else ""
            author_id = user_field.get("id", "") if isinstance(user_field, dict) else ""

            # Extract mentions
            mentions = []
            for m in msg.get("mentions", []):
                mentioned = m.get("mentioned", {})
                if isinstance(mentioned, dict):
                    name = mentioned.get("user", {}).get("displayName", "")
                    if name:
                        mentions.append(name)

            # Importance boost stored in platform_metadata
            importance = msg.get("importance", "normal")

            messages.append(ChatMessage(
                author=author,
                author_id=author_id,
                content=body_content,
                timestamp=msg.get("createdDateTime", None),
                message_id=msg.get("id", ""),
                thread_id=None,  # Set during grouping
                reply_to=msg.get("replyToId", None),
                channel=metadata.get("channel", ""),
                channel_id=metadata.get("channel_id", msg.get("channelIdentity", {}).get("channelId", "")),
                mentions=mentions,
                is_bot=msg.get("from", {}).get("application") is not None,
                platform_metadata={
                    "importance": importance,
                    "message_type": msg.get("messageType", ""),
                },
            ))

        return messages

    def _group_into_threads(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatThread]:
        # Build reply tree: find root messages and collect descendants
        by_id: dict[str, ChatMessage] = {}
        children: dict[str, list[str]] = {}
        roots: list[str] = []

        for msg in messages:
            by_id[msg.message_id] = msg
            reply_to = msg.reply_to
            if reply_to and reply_to in by_id:
                children.setdefault(reply_to, []).append(msg.message_id)
            elif reply_to:
                # replyToId references a message not in the set — treat as root
                children.setdefault(reply_to, []).append(msg.message_id)
                if reply_to not in [m.message_id for m in messages]:
                    roots.append(msg.message_id)
            else:
                roots.append(msg.message_id)

        # Collect threads by walking from each root
        visited: set[str] = set()
        threads: list[ChatThread] = []

        for root_id in roots:
            if root_id in visited:
                continue
            thread_msgs = self._collect_thread(root_id, by_id, children, visited)
            if thread_msgs:
                thread_msgs.sort(key=lambda m: m.timestamp or "")
                participants = list({m.author for m in thread_msgs if m.author})
                threads.append(ChatThread(
                    thread_id=root_id,
                    channel=thread_msgs[0].channel if thread_msgs else "",
                    channel_id=thread_msgs[0].channel_id if thread_msgs else "",
                    messages=thread_msgs,
                    participants=participants,
                    start_time=thread_msgs[0].timestamp if thread_msgs else None,
                    end_time=thread_msgs[-1].timestamp if thread_msgs else None,
                ))

        # Pick up orphaned messages not in any thread
        orphans = [m for m in messages if m.message_id not in visited]
        if orphans:
            from uuid import uuid4
            participants = list({m.author for m in orphans if m.author})
            threads.append(ChatThread(
                thread_id=str(uuid4()),
                channel=orphans[0].channel if orphans else "",
                channel_id=orphans[0].channel_id if orphans else "",
                messages=orphans,
                participants=participants,
                start_time=orphans[0].timestamp if orphans else None,
                end_time=orphans[-1].timestamp if orphans else None,
            ))

        return threads

    def _collect_thread(
        self,
        msg_id: str,
        by_id: dict[str, ChatMessage],
        children: dict[str, list[str]],
        visited: set[str],
    ) -> list[ChatMessage]:
        """BFS to collect all messages in a reply tree."""
        result: list[ChatMessage] = []
        queue = [msg_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in by_id:
                result.append(by_id[current])
            for child_id in children.get(current, []):
                queue.append(child_id)

        return result


def _strip_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
