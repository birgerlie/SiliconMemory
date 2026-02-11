"""Discord chat ingestion adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.ingestion.chat import (
    BaseChatAdapter,
    ChatConfig,
    ChatMessage,
    ChatThread,
)

if TYPE_CHECKING:
    from silicon_memory.entities.resolver import EntityResolver


@dataclass
class DiscordConfig(ChatConfig):
    """Configuration for Discord ingestion."""

    include_reactions: bool = True
    role_weights: dict[str, float] = field(default_factory=dict)
    skip_bot_messages: bool = True


class DiscordAdapter(BaseChatAdapter):
    """Ingests Discord channel exports as experiences.

    Input format: Discord export JSON
    ``{"messages": [...], "channel": {...}}``.

    Thread grouping uses ``reference.messageId`` for replies
    and forum channel thread grouping.
    """

    def __init__(
        self,
        config: DiscordConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        super().__init__(
            config=config or DiscordConfig(),
            entity_resolver=entity_resolver,
        )

    @property
    def _discord_config(self) -> DiscordConfig:
        return self._config  # type: ignore[return-value]

    @property
    def source_type(self) -> str:
        return "discord"

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
            raw_messages = data.get("messages", [])
            channel_info = data.get("channel", {})
            if isinstance(channel_info, dict):
                metadata.setdefault("channel", channel_info.get("name", ""))
                metadata.setdefault("channel_id", channel_info.get("id", ""))
        elif isinstance(data, list):
            raw_messages = data
            channel_info = {}
        else:
            raise ValueError("Expected JSON object with 'messages' key or array")

        messages: list[ChatMessage] = []

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            # Extract author
            author_data = msg.get("author", {})
            if isinstance(author_data, dict):
                author = author_data.get("name", author_data.get("username", ""))
                author_id = author_data.get("id", "")
                is_bot = author_data.get("isBot", False)
            else:
                author = str(author_data) if author_data else ""
                author_id = ""
                is_bot = False

            # Extract reactions
            reactions: list[dict] = []
            if self._discord_config.include_reactions:
                for r in msg.get("reactions", []):
                    emoji = r.get("emoji", {})
                    reactions.append({
                        "name": emoji.get("name", "") if isinstance(emoji, dict) else str(emoji),
                        "count": r.get("count", 0),
                    })

            # Extract reply reference
            reference = msg.get("reference", {})
            reply_to = None
            if isinstance(reference, dict):
                reply_to = reference.get("messageId", None)

            # Extract mentions
            mentions = []
            for m in msg.get("mentions", []):
                if isinstance(m, dict):
                    mentions.append(m.get("name", m.get("username", "")))
                elif isinstance(m, str):
                    mentions.append(m)

            # Attachments
            attachments = []
            for a in msg.get("attachments", []):
                if isinstance(a, dict):
                    attachments.append(a.get("fileName", a.get("url", "")))
                elif isinstance(a, str):
                    attachments.append(a)

            messages.append(ChatMessage(
                author=author,
                author_id=author_id,
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", None),
                message_id=msg.get("id", ""),
                thread_id=None,  # Set during grouping
                reply_to=reply_to,
                channel=metadata.get("channel", ""),
                channel_id=metadata.get("channel_id", ""),
                reactions=reactions,
                attachments=attachments,
                mentions=mentions,
                is_bot=is_bot,
                platform_metadata={
                    k: v for k, v in msg.items()
                    if k not in ("content", "author", "id", "timestamp",
                                 "reference", "reactions", "mentions", "attachments")
                },
            ))

        return messages

    def _group_into_threads(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatThread]:
        # Build reply graph: find root messages and collect descendants
        by_id: dict[str, ChatMessage] = {}
        children: dict[str, list[str]] = {}
        roots: list[str] = []

        for msg in messages:
            by_id[msg.message_id] = msg

        for msg in messages:
            if msg.reply_to and msg.reply_to in by_id:
                children.setdefault(msg.reply_to, []).append(msg.message_id)
            else:
                roots.append(msg.message_id)

        # Walk from each root to collect threads
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
