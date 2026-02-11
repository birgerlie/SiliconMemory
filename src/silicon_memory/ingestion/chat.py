"""Chat platform ingestion adapter (base class).

Provides shared data types and pipeline for chat platform adapters
(Slack, Teams, Discord). Platform-specific subclasses override only
``_parse_messages()`` and ``_group_into_threads()``.

Knowledge discovery from chat: extracts decisions, deadlines, and
tribal knowledge as Beliefs — similar to how NewsArticleAdapter
extracts claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Belief, Experience, Procedure, Source, SourceType
from silicon_memory.ingestion.types import IngestionConfig, IngestionResult
from silicon_memory.ingestion._helpers import (
    extract_action_items_from_text,
    parse_llm_json_array,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.reflection.llm import LLMProvider
    from silicon_memory.entities.resolver import EntityResolver


# ============================================================================
# Data types
# ============================================================================


@dataclass
class ChatMessage:
    """A single chat message from any platform."""

    author: str = ""
    author_id: str = ""
    content: str = ""
    timestamp: str | None = None
    message_id: str = ""
    thread_id: str | None = None
    reply_to: str | None = None
    channel: str = ""
    channel_id: str = ""
    reactions: list[dict] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    is_bot: bool = False
    platform_metadata: dict = field(default_factory=dict)


@dataclass
class ChatThread:
    """A group of related chat messages (thread or time-proximity window)."""

    thread_id: str = ""
    channel: str = ""
    channel_id: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    topic: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def combined_text(self) -> str:
        return "\n".join(
            f"{m.author}: {m.content}" if m.author else m.content
            for m in self.messages
        )


@dataclass
class ChatConfig(IngestionConfig):
    """Configuration for chat platform ingestion."""

    extract_action_items: bool = True
    extract_beliefs: bool = True
    resolve_entities: bool = True
    create_graph_edges: bool = True
    group_by_threads: bool = True
    max_thread_messages: int = 500
    min_thread_messages: int = 2
    skip_bot_messages: bool = False
    belief_confidence: float = 0.5
    max_beliefs_per_thread: int = 10


# Chat-specific action item patterns.
_CHAT_ACTION_PATTERNS: list[re.Pattern[str]] = [
    # "I'll [verb]", "I will [verb]"
    re.compile(r"I'?ll\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "@mention please [verb]"
    re.compile(r"@\w+\s+(?:please|pls)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "can someone [verb]"
    re.compile(r"can someone\s+(.+?)(?:\.|$)", re.IGNORECASE),
]

# Heuristic patterns for belief/knowledge extraction.
_BELIEF_PATTERNS: list[re.Pattern[str]] = [
    # "we decided to ..."
    re.compile(r"we decided (?:to\s+)?(.+?)(?:\.|$)", re.IGNORECASE),
    # "the decision is ..."
    re.compile(r"the decision is\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "deadline is ..."
    re.compile(r"deadline (?:is|:)\s*(.+?)(?:\.|$)", re.IGNORECASE),
    # "FYI: ..."
    re.compile(r"FYI[:\s]+(.+?)(?:\.|$)", re.IGNORECASE),
    # "let's go with ..."
    re.compile(r"let'?s go with\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "agreed: ..." / "agreed, ..."
    re.compile(r"agreed[,:]\s*(.+?)(?:\.|$)", re.IGNORECASE),
    # "confirmed: ..."
    re.compile(r"confirmed[,:]\s*(.+?)(?:\.|$)", re.IGNORECASE),
]


# ============================================================================
# Base chat adapter
# ============================================================================


class BaseChatAdapter:
    """Base adapter for chat platform ingestion.

    Subclasses must implement:
    - ``source_type`` property
    - ``_parse_messages(content, metadata)``
    - ``_group_into_threads(messages)``

    The shared pipeline handles:
    1. Parse → platform-specific ``_parse_messages``
    2. Group → platform-specific ``_group_into_threads``
    3. Create experiences (one per thread)
    4. Extract action items (heuristic or LLM)
    5. Extract beliefs/knowledge (heuristic or LLM)
    6. Persist action items as Procedures
    7. Resolve entities
    8. Create graph edges
    """

    def __init__(
        self,
        config: ChatConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        self._config = config or ChatConfig()
        self._entity_resolver = entity_resolver

    @property
    def source_type(self) -> str:
        raise NotImplementedError

    def _parse_messages(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> list[ChatMessage]:
        """Parse raw content into ChatMessage list. Override in subclass."""
        raise NotImplementedError

    def _group_into_threads(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatThread]:
        """Group messages into threads. Override in subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> IngestionResult:
        result = IngestionResult(source_type=self.source_type)

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        content = content.strip()
        if not content:
            result.errors.append("Empty chat content")
            return result

        # Step 1: Parse
        try:
            messages = self._parse_messages(content, metadata)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        if not messages:
            result.errors.append("No parseable messages found")
            return result

        # Optionally filter bot messages
        if self._config.skip_bot_messages:
            messages = [m for m in messages if not m.is_bot]

        if not messages:
            result.errors.append("No messages after filtering")
            return result

        # Step 2: Group into threads
        if self._config.group_by_threads:
            threads = self._group_into_threads(messages)
        else:
            # Single thread with all messages
            threads = [ChatThread(
                thread_id=str(uuid4()),
                messages=messages,
                participants=list({m.author for m in messages if m.author}),
            )]

        # Filter threads by min message count
        threads = [
            t for t in threads
            if t.message_count >= self._config.min_thread_messages
        ]

        if not threads:
            result.errors.append("No threads meet minimum message count")
            return result

        # Truncate oversized threads
        for thread in threads:
            thread.messages = thread.messages[:self._config.max_thread_messages]

        user_ctx = memory.user_context
        conversation_id = metadata.get("conversation_id", str(uuid4()))

        # Step 3: Create experiences (one per thread)
        for i, thread in enumerate(threads):
            try:
                exp = Experience(
                    id=uuid4(),
                    content=thread.combined_text,
                    context={
                        "source_type": self.source_type,
                        "thread_id": thread.thread_id,
                        "channel": thread.channel,
                        "channel_id": thread.channel_id,
                        "participants": thread.participants,
                        "message_count": thread.message_count,
                        "topic": thread.topic,
                        "start_time": thread.start_time,
                        "end_time": thread.end_time,
                        "thread_index": i,
                        **{k: v for k, v in metadata.items()
                           if k not in ("conversation_id",)},
                    },
                    session_id=conversation_id,
                    user_id=user_ctx.user_id,
                    tenant_id=user_ctx.tenant_id,
                )
                await memory.record_experience(exp)
                result.experiences_created += 1
            except Exception as e:
                result.errors.append(f"Failed to store thread {i}: {e}")

        # Step 4: Extract action items
        if self._config.extract_action_items:
            try:
                if llm_provider:
                    action_items = await self._extract_action_items_llm(
                        threads, llm_provider
                    )
                else:
                    action_items = self._extract_action_items_heuristic(threads)
                result.action_items_detected = len(action_items)
                result.details["action_items"] = action_items

                # Persist as Procedures
                for item in action_items:
                    try:
                        procedure = Procedure(
                            id=uuid4(),
                            name=item.get("action", "")[:100],
                            description=item.get("action", ""),
                            trigger=f"From {self.source_type} conversation {conversation_id}",
                            steps=[item.get("action", "")],
                            confidence=0.6,
                            tags={"action_item", "chat", self.source_type},
                            user_id=user_ctx.user_id,
                            tenant_id=user_ctx.tenant_id,
                        )
                        await memory.commit_procedure(procedure)
                    except Exception as e:
                        result.errors.append(f"Failed to store action item: {e}")
            except Exception as e:
                result.errors.append(f"Action item extraction error: {e}")

        # Step 5: Extract beliefs/knowledge
        if self._config.extract_beliefs:
            try:
                if llm_provider:
                    beliefs = await self._extract_beliefs_llm(threads, llm_provider)
                else:
                    beliefs = self._extract_beliefs_heuristic(threads)

                beliefs = beliefs[:self._config.max_beliefs_per_thread * len(threads)]
                result.details["beliefs"] = beliefs
                result.decisions_detected = len(beliefs)

                source = Source(
                    id=f"{self.source_type}:{conversation_id}",
                    type=SourceType.OBSERVATION,
                    reliability=self._config.belief_confidence,
                    metadata={"source_type": self.source_type},
                )

                for claim in beliefs:
                    try:
                        belief = Belief(
                            id=uuid4(),
                            content=claim.get("belief", ""),
                            confidence=min(
                                1.0,
                                self._config.belief_confidence
                                * claim.get("confidence", 0.7),
                            ),
                            source=source,
                            tags={"chat", self.source_type, "knowledge_discovery"},
                            user_id=user_ctx.user_id,
                            tenant_id=user_ctx.tenant_id,
                        )
                        await memory.commit_belief(belief)
                    except Exception as e:
                        result.errors.append(f"Failed to store belief: {e}")
            except Exception as e:
                result.errors.append(f"Belief extraction error: {e}")

        # Step 6: Resolve entities
        if self._config.resolve_entities and self._entity_resolver:
            try:
                resolved_count = await self._resolve_entities(threads, memory)
                result.entities_resolved = resolved_count
            except Exception as e:
                result.errors.append(f"Entity resolution error: {e}")

        # Step 7: Create graph edges
        if self._config.create_graph_edges:
            try:
                await self._create_graph_edges(
                    memory, conversation_id, threads, result
                )
            except Exception as e:
                result.errors.append(f"Graph edge creation error: {e}")

        return result

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------

    def _extract_action_items_heuristic(
        self,
        threads: list[ChatThread],
    ) -> list[dict[str, Any]]:
        action_items: list[dict[str, Any]] = []
        for i, thread in enumerate(threads):
            action_items.extend(
                extract_action_items_from_text(
                    thread.combined_text,
                    "thread_index",
                    i,
                    extra_patterns=_CHAT_ACTION_PATTERNS,
                )
            )
        return action_items

    async def _extract_action_items_llm(
        self,
        threads: list[ChatThread],
        provider: "LLMProvider",
    ) -> list[dict[str, Any]]:
        text = "\n\n".join(
            f"[Thread {i}] {t.combined_text}" for i, t in enumerate(threads)
        )

        prompt = (
            "Extract action items from this chat conversation. "
            "Return a JSON array where each element has:\n"
            '- "action": what needs to be done\n'
            '- "owner": who is responsible (or null)\n'
            '- "thread_index": which thread it came from\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"{text[:3000]}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a chat action item extractor.",
            temperature=self._config.llm_temperature,
            max_tokens=1500,
        )

        try:
            return parse_llm_json_array(response)
        except (ValueError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------
    # Beliefs / knowledge discovery
    # ------------------------------------------------------------------

    def _extract_beliefs_heuristic(
        self,
        threads: list[ChatThread],
    ) -> list[dict[str, Any]]:
        beliefs: list[dict[str, Any]] = []
        for thread in threads:
            for line in thread.combined_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                for pattern in _BELIEF_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        claim = match.group(1).strip()
                        if len(claim) >= 10:
                            beliefs.append({
                                "belief": claim,
                                "confidence": 0.7,
                                "thread_id": thread.thread_id,
                            })
                        break
        return beliefs

    async def _extract_beliefs_llm(
        self,
        threads: list[ChatThread],
        provider: "LLMProvider",
    ) -> list[dict[str, Any]]:
        text = "\n\n".join(
            f"[Thread {i}] {t.combined_text}" for i, t in enumerate(threads)
        )

        prompt = (
            "Extract key decisions, facts, and knowledge from this chat conversation. "
            "Return a JSON array where each element has:\n"
            '- "belief": the factual claim or decision\n'
            '- "confidence": how confident this claim appears (0-1)\n'
            '- "thread_id": which thread it came from\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"{text[:3000]}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a knowledge extraction engine for chat conversations.",
            temperature=0.2,
            max_tokens=2000,
        )

        try:
            return parse_llm_json_array(response)
        except (ValueError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    async def _resolve_entities(
        self,
        threads: list[ChatThread],
        memory: "SiliconMemory",
    ) -> int:
        if not self._entity_resolver:
            return 0

        resolved_count = 0
        all_participants: set[str] = set()

        for thread in threads:
            all_participants.update(thread.participants)

        for name in all_participants:
            try:
                result = await self._entity_resolver.resolve(name)
                if result.resolved:
                    resolved_count += len(result.resolved)
            except Exception:
                pass

        return resolved_count

    # ------------------------------------------------------------------
    # Graph edges
    # ------------------------------------------------------------------

    async def _create_graph_edges(
        self,
        memory: "SiliconMemory",
        conversation_id: str,
        threads: list[ChatThread],
        result: IngestionResult,
    ) -> None:
        backend = getattr(memory, "_backend", None)
        if backend is None:
            return

        db = getattr(backend, "_db", None)
        if db is None:
            return

        all_participants: set[str] = set()
        all_channels: set[str] = set()
        for thread in threads:
            all_participants.update(thread.participants)
            if thread.channel:
                all_channels.add(thread.channel)

        # participant → conversation (participated_in)
        for participant in all_participants:
            try:
                db.add_edge(
                    participant,
                    conversation_id,
                    edge_type="participated_in",
                    metadata={"source": self.source_type},
                )
            except Exception:
                pass

        # participant ↔ participant (communicates_with)
        participants_list = sorted(all_participants)
        for i, p1 in enumerate(participants_list):
            for p2 in participants_list[i + 1:]:
                try:
                    db.add_edge(
                        p1,
                        p2,
                        edge_type="communicates_with",
                        metadata={"source": self.source_type},
                    )
                except Exception:
                    pass

        # participant → channel (member_of)
        for participant in all_participants:
            for channel in all_channels:
                try:
                    db.add_edge(
                        participant,
                        channel,
                        edge_type="member_of",
                        metadata={"source": self.source_type},
                    )
                except Exception:
                    pass

        # conversation → action_item (has_action)
        action_items = result.details.get("action_items", [])
        for item in action_items:
            action_text = item.get("action", "")
            if action_text:
                try:
                    db.add_edge(
                        conversation_id,
                        action_text[:80],
                        edge_type="has_action",
                        metadata={
                            "owner": item.get("owner"),
                            "source": self.source_type,
                        },
                    )
                except Exception:
                    pass
