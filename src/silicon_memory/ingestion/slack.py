"""Slack chat ingestion adapter."""

from __future__ import annotations

import json
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
class SlackConfig(ChatConfig):
    """Configuration for Slack ingestion."""

    include_reactions: bool = True
    reaction_weight_boost: float = 0.1
    skip_subtypes: list[str] = field(
        default_factory=lambda: ["channel_join", "channel_leave", "channel_topic"]
    )
    user_map: dict[str, str] = field(default_factory=dict)


class SlackAdapter(BaseChatAdapter):
    """Ingests Slack channel exports as experiences.

    Input formats:
    - JSON array of Slack messages: ``[{...}, ...]``
    - Wrapper object: ``{"messages": [...]}``

    Thread grouping uses Slack's ``thread_ts`` field. Messages
    without ``thread_ts`` are grouped by time-proximity windows.
    """

    def __init__(
        self,
        config: SlackConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        super().__init__(config=config or SlackConfig(), entity_resolver=entity_resolver)

    @property
    def _slack_config(self) -> SlackConfig:
        return self._config  # type: ignore[return-value]

    @property
    def source_type(self) -> str:
        return "slack"

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
        elif isinstance(data, list):
            raw_messages = data
        else:
            raise ValueError("Expected JSON array or object with 'messages' key")

        skip_subtypes = set(self._slack_config.skip_subtypes)
        user_map = self._slack_config.user_map
        messages: list[ChatMessage] = []

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            # Skip filtered subtypes
            subtype = msg.get("subtype", "")
            if subtype in skip_subtypes:
                continue

            user_id = msg.get("user", "")
            author = user_map.get(user_id, msg.get("username", msg.get("user", "")))

            reactions: list[dict] = []
            if self._slack_config.include_reactions:
                for r in msg.get("reactions", []):
                    reactions.append({
                        "name": r.get("name", ""),
                        "count": r.get("count", 0),
                    })

            messages.append(ChatMessage(
                author=author,
                author_id=user_id,
                content=msg.get("text", ""),
                timestamp=msg.get("ts", None),
                message_id=msg.get("ts", ""),
                thread_id=msg.get("thread_ts", None),
                reply_to=msg.get("thread_ts", None) if msg.get("thread_ts") != msg.get("ts") else None,
                channel=metadata.get("channel", ""),
                channel_id=metadata.get("channel_id", ""),
                reactions=reactions,
                attachments=[a.get("name", "") for a in msg.get("files", [])],
                mentions=_extract_slack_mentions(msg.get("text", "")),
                is_bot=msg.get("subtype") == "bot_message" or bool(msg.get("bot_id")),
                platform_metadata={
                    k: v for k, v in msg.items()
                    if k not in ("text", "user", "ts", "thread_ts", "reactions", "files")
                },
            ))

        return messages

    def _group_into_threads(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatThread]:
        # Group by thread_ts (Slack's native threading)
        threaded: dict[str, list[ChatMessage]] = {}
        standalone: list[ChatMessage] = []

        for msg in messages:
            tid = msg.thread_id
            if tid:
                threaded.setdefault(tid, []).append(msg)
            else:
                standalone.append(msg)

        threads: list[ChatThread] = []

        # Create threads from Slack thread_ts groups
        for tid, msgs in threaded.items():
            msgs.sort(key=lambda m: m.timestamp or "")
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

        # Group standalone messages by time-proximity (5-minute windows)
        if standalone:
            standalone.sort(key=lambda m: m.timestamp or "")
            threads.extend(_group_by_time_proximity(standalone))

        return threads


def _extract_slack_mentions(text: str) -> list[str]:
    """Extract <@U...> mentions from Slack message text."""
    import re
    return re.findall(r"<@(\w+)>", text)


def _group_by_time_proximity(
    messages: list[ChatMessage],
    window_seconds: float = 300.0,
) -> list[ChatThread]:
    """Group messages into time-proximity windows (default 5 min)."""
    if not messages:
        return []

    threads: list[ChatThread] = []
    current: list[ChatMessage] = [messages[0]]
    current_ts = _ts_to_float(messages[0].timestamp)

    for msg in messages[1:]:
        msg_ts = _ts_to_float(msg.timestamp)
        if msg_ts - current_ts > window_seconds:
            threads.append(_messages_to_thread(current))
            current = [msg]
            current_ts = msg_ts
        else:
            current.append(msg)

    if current:
        threads.append(_messages_to_thread(current))

    return threads


def _ts_to_float(ts: str | None) -> float:
    """Convert Slack-style timestamp to float."""
    if not ts:
        return 0.0
    try:
        return float(ts)
    except ValueError:
        return 0.0


def _messages_to_thread(messages: list[ChatMessage]) -> ChatThread:
    """Build a ChatThread from a list of messages."""
    from uuid import uuid4

    participants = list({m.author for m in messages if m.author})
    return ChatThread(
        thread_id=str(uuid4()),
        channel=messages[0].channel if messages else "",
        channel_id=messages[0].channel_id if messages else "",
        messages=messages,
        participants=participants,
        start_time=messages[0].timestamp if messages else None,
        end_time=messages[-1].timestamp if messages else None,
    )
