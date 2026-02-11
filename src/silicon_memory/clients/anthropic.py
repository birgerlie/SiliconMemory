"""Anthropic client with memory integration."""

from __future__ import annotations

from typing import Any, AsyncIterator, TYPE_CHECKING

from silicon_memory.clients.base import (
    MemoryAugmentedClient,
    MemoryClientConfig,
    ChatMessage,
    ChatResponse,
    MessageRole,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class AnthropicMemoryClient(MemoryAugmentedClient):
    """Anthropic client with memory integration.

    Wraps the Anthropic API with automatic memory augmentation:
    - Injects relevant context from memory into prompts
    - Records interactions as experiences
    - Provides memory tools for function calling

    Example:
        >>> from silicon_memory import SiliconMemory
        >>> from silicon_memory.clients import AnthropicMemoryClient
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     client = AnthropicMemoryClient(
        ...         memory=memory,
        ...         api_key=os.environ["ANTHROPIC_API_KEY"],
        ...         model="claude-sonnet-4-20250514",
        ...     )
        ...
        ...     response = await client.chat(
        ...         messages=[{"role": "user", "content": "What is Python?"}],
        ...         use_memory=True,
        ...     )
        ...     print(response.content)
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        config: MemoryClientConfig | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize the Anthropic memory client.

        Args:
            memory: SiliconMemory instance
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens in response
            config: Memory client configuration
            **client_kwargs: Additional arguments for Anthropic client
        """
        super().__init__(memory, config)

        self._model = model
        self._max_tokens = max_tokens
        self._client_kwargs = client_kwargs

        # Import and create Anthropic client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic is required for AnthropicMemoryClient. "
                "Install with: pip install anthropic"
            ) from e

        self._client = AsyncAnthropic(api_key=api_key, **client_kwargs)

    async def _call_api(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Make the Anthropic API call."""
        # Anthropic uses a different message format
        # System message goes separately
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg.role in (MessageRole.SYSTEM, "system"):
                system_content = msg.content
            else:
                api_messages.append({
                    "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                    "content": msg.content,
                })

        # Set defaults
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        # Build API call kwargs
        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            **kwargs,
        }

        if system_content:
            api_kwargs["system"] = system_content

        # Make the API call
        response = await self._client.messages.create(**api_kwargs)

        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return ChatResponse(
            content=content,
            role="assistant",
            model=response.model,
            finish_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def _stream_api(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream the Anthropic API response."""
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg.role in (MessageRole.SYSTEM, "system"):
                system_content = msg.content
            else:
                api_messages.append({
                    "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                    "content": msg.content,
                })

        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            **kwargs,
        }

        if system_content:
            api_kwargs["system"] = system_content

        async with self._client.messages.stream(**api_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def chat_with_tools(
        self,
        messages: list[dict | ChatMessage],
        tools: list[dict] | None = None,
        include_memory_tools: bool = True,
        use_memory: bool = True,
        record_experience: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat with tool use support.

        Args:
            messages: List of messages
            tools: Additional tools to include
            include_memory_tools: Include memory tool schemas
            use_memory: Inject memory context
            record_experience: Record the interaction
            **kwargs: Additional API arguments

        Returns:
            ChatResponse (may include tool_calls)
        """
        # Build tools list - convert to Anthropic format
        all_tools = []

        if tools:
            for tool in tools:
                anthropic_tool = self._convert_tool_schema(tool)
                all_tools.append(anthropic_tool)

        if include_memory_tools:
            for tool in self.get_tools():
                anthropic_tool = self._convert_tool_schema(tool)
                all_tools.append(anthropic_tool)

        if all_tools:
            kwargs["tools"] = all_tools

        return await self.chat(
            messages=messages,
            use_memory=use_memory,
            record_experience=record_experience,
            **kwargs,
        )

    def _convert_tool_schema(self, openai_schema: dict) -> dict:
        """Convert OpenAI tool schema to Anthropic format."""
        return {
            "name": openai_schema["name"],
            "description": openai_schema.get("description", ""),
            "input_schema": openai_schema.get("parameters", {}),
        }

    async def complete(
        self,
        prompt: str,
        use_memory: bool = True,
        **kwargs: Any,
    ) -> str:
        """Simple completion interface.

        Args:
            prompt: The prompt to complete
            use_memory: Inject memory context
            **kwargs: Additional API arguments

        Returns:
            The completion text
        """
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            use_memory=use_memory,
            **kwargs,
        )
        return response.content

    async def message(
        self,
        messages: list[dict | ChatMessage],
        use_memory: bool = True,
        record_experience: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Alias for chat() to match Anthropic naming convention.

        Args:
            messages: List of messages
            use_memory: Inject memory context
            record_experience: Record the interaction
            **kwargs: Additional API arguments

        Returns:
            ChatResponse with the assistant's response
        """
        return await self.chat(
            messages=messages,
            use_memory=use_memory,
            record_experience=record_experience,
            **kwargs,
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tool schemas in Anthropic format."""
        from silicon_memory.tools.memory_tool import MemoryTool

        openai_schema = MemoryTool.get_openai_schema()
        return [self._convert_tool_schema(openai_schema)]
