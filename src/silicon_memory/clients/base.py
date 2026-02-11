"""Base class for memory-augmented LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Experience, Source, SourceType
from silicon_memory.memory.silicondb_router import RecallContext
from silicon_memory.tools.memory_tool import MemoryTool
from silicon_memory.tools.query_tool import QueryTool
from silicon_memory.security.types import UserContext

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class MessageRole(str, Enum):
    """Role in a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """A message in a chat conversation."""

    role: MessageRole | str
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            name=d.get("name"),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
        )


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    content: str
    role: str = "assistant"
    model: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    tool_calls: list[dict] | None = None

    # Memory tracking
    memory_context_used: bool = False
    experience_recorded: bool = False
    experience_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "role": self.role,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "memory_context_used": self.memory_context_used,
            "experience_recorded": self.experience_recorded,
        }


@dataclass
class MemoryClientConfig:
    """Configuration for memory-augmented clients."""

    # User context (required)
    user_context: UserContext | None = None

    # Memory retrieval
    auto_recall: bool = True
    max_context_facts: int = 10
    max_context_experiences: int = 5
    max_context_procedures: int = 3
    min_confidence: float = 0.3

    # Experience recording
    auto_record: bool = True
    record_user_messages: bool = True
    record_assistant_responses: bool = True

    # Context injection
    inject_as_system_message: bool = True
    context_prefix: str = "Relevant context from memory:\n"

    # Session tracking
    session_id: str | None = None


class MemoryAugmentedClient(ABC):
    """Base class for memory-augmented LLM clients.

    Features:
    - Automatic context injection from memory
    - Experience recording after each interaction
    - Belief extraction from responses (optional)
    - Working memory management

    Subclasses implement the actual API calls for specific providers.

    Example:
        >>> class MyClient(MemoryAugmentedClient):
        ...     async def _call_api(self, messages, **kwargs):
        ...         # Implement API call
        ...         pass
        >>>
        >>> client = MyClient(memory)
        >>> response = await client.chat(messages)
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        config: MemoryClientConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or MemoryClientConfig()
        self._memory_tool = MemoryTool(memory)
        self._query_tool = QueryTool(memory)

        # Use session from config, or from memory's user context, or generate new
        if self._config.session_id:
            self._session_id = self._config.session_id
        elif hasattr(memory, 'user_context') and memory.user_context:
            self._session_id = memory.user_context.session_id
        else:
            self._session_id = f"session-{uuid4().hex[:8]}"

    @property
    def user_context(self) -> UserContext | None:
        """Get the user context from the memory system."""
        if hasattr(self._memory, 'user_context'):
            return self._memory.user_context
        return self._config.user_context

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    async def chat(
        self,
        messages: list[dict | ChatMessage],
        use_memory: bool = True,
        record_experience: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat message with memory augmentation.

        Args:
            messages: List of messages in the conversation
            use_memory: Whether to inject memory context
            record_experience: Whether to record the interaction
            **kwargs: Additional arguments for the API

        Returns:
            ChatResponse with the assistant's response
        """
        # Normalize messages
        normalized = self._normalize_messages(messages)

        # Inject memory context if enabled
        context_injected = False
        if use_memory and self._config.auto_recall:
            normalized, context_injected = await self._inject_context(normalized)

        # Make the API call
        response = await self._call_api(normalized, **kwargs)
        response.memory_context_used = context_injected

        # Record experience if enabled
        if record_experience and self._config.auto_record:
            exp_id = await self._record_interaction(normalized, response)
            response.experience_recorded = True
            response.experience_id = exp_id

        return response

    async def chat_stream(
        self,
        messages: list[dict | ChatMessage],
        use_memory: bool = True,
        record_experience: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat response with memory augmentation.

        Args:
            messages: List of messages
            use_memory: Whether to inject memory context
            record_experience: Whether to record the interaction
            **kwargs: Additional arguments for the API

        Yields:
            Response content chunks
        """
        normalized = self._normalize_messages(messages)

        if use_memory and self._config.auto_recall:
            normalized, _ = await self._inject_context(normalized)

        # Collect full response while streaming
        full_content = []
        async for chunk in self._stream_api(normalized, **kwargs):
            full_content.append(chunk)
            yield chunk

        # Record experience after streaming completes
        if record_experience and self._config.auto_record:
            response = ChatResponse(content="".join(full_content))
            await self._record_interaction(normalized, response)

    @abstractmethod
    async def _call_api(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Make the actual API call. Implemented by subclasses."""
        ...

    @abstractmethod
    async def _stream_api(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream the API response. Implemented by subclasses."""
        ...

    def _normalize_messages(
        self,
        messages: list[dict | ChatMessage],
    ) -> list[ChatMessage]:
        """Convert messages to ChatMessage objects."""
        normalized = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized.append(msg)
            else:
                normalized.append(ChatMessage.from_dict(msg))
        return normalized

    async def _inject_context(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], bool]:
        """Inject memory context into messages."""
        # Extract the query from the last user message
        query = ""
        for msg in reversed(messages):
            if msg.role in (MessageRole.USER, "user"):
                query = msg.content
                break

        if not query:
            return messages, False

        # Recall relevant memories
        ctx = RecallContext(
            query=query,
            max_facts=self._config.max_context_facts,
            max_experiences=self._config.max_context_experiences,
            max_procedures=self._config.max_context_procedures,
            min_confidence=self._config.min_confidence,
        )
        recall = await self._memory.recall(ctx)

        # Build context string
        context_parts = []

        if recall.facts:
            context_parts.append("Facts:")
            for fact in recall.facts:
                context_parts.append(f"  - {fact.content} ({fact.confidence:.0%})")

        if recall.experiences:
            context_parts.append("Previous interactions:")
            for exp in recall.experiences:
                context_parts.append(f"  - {exp.content}")

        if recall.procedures:
            context_parts.append("Relevant procedures:")
            for proc in recall.procedures:
                context_parts.append(f"  - {proc.content}")

        if recall.working_context:
            context_parts.append("Current context:")
            for key, value in recall.working_context.items():
                context_parts.append(f"  - {key}: {value}")

        if not context_parts:
            return messages, False

        context_str = self._config.context_prefix + "\n".join(context_parts)

        # Inject as system message or prepend to user message
        result = list(messages)
        if self._config.inject_as_system_message:
            # Find or create system message
            system_idx = None
            for i, msg in enumerate(result):
                if msg.role in (MessageRole.SYSTEM, "system"):
                    system_idx = i
                    break

            if system_idx is not None:
                # Append to existing system message
                result[system_idx] = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=result[system_idx].content + "\n\n" + context_str,
                )
            else:
                # Insert new system message at beginning
                result.insert(0, ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=context_str,
                ))
        else:
            # Prepend to user message
            for i in range(len(result) - 1, -1, -1):
                if result[i].role in (MessageRole.USER, "user"):
                    result[i] = ChatMessage(
                        role=result[i].role,
                        content=f"{context_str}\n\n---\n\n{result[i].content}",
                    )
                    break

        return result, True

    async def _record_interaction(
        self,
        messages: list[ChatMessage],
        response: ChatResponse,
    ) -> str | None:
        """Record the interaction as an experience."""
        # Find the user message
        user_content = ""
        for msg in reversed(messages):
            if msg.role in (MessageRole.USER, "user"):
                user_content = msg.content
                break

        if not user_content:
            return None

        # Get user context for security metadata
        user_ctx = self.user_context

        # Create experience with user/tenant info
        experience = Experience(
            id=uuid4(),
            content=f"User: {user_content[:500]}",
            outcome=f"Assistant: {response.content[:500]}",
            session_id=self._session_id,
            user_id=user_ctx.user_id if user_ctx else None,
            tenant_id=user_ctx.tenant_id if user_ctx else None,
        )

        await self._memory.record_experience(experience)
        return str(experience.id)

    async def set_context(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set a value in working memory."""
        await self._memory.set_context(key, value, ttl_seconds)

    async def get_context(self, key: str) -> Any:
        """Get a value from working memory."""
        return await self._memory.get_context(key)

    async def recall(self, query: str) -> dict[str, Any]:
        """Manually recall memories for a query."""
        response = await self._memory_tool.invoke("recall", query=query)
        return response.data

    async def what_do_you_know(self, query: str) -> dict[str, Any]:
        """Query knowledge with proofs."""
        response = await self._memory_tool.invoke("what_do_you_know", query=query)
        return response.data

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tool schemas for function calling."""
        return [
            MemoryTool.get_openai_schema(),
        ]
