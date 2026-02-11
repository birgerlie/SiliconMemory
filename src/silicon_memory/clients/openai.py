"""OpenAI client with memory integration."""

from __future__ import annotations

from typing import Any, AsyncIterator, TYPE_CHECKING

from silicon_memory.clients.base import (
    MemoryAugmentedClient,
    MemoryClientConfig,
    ChatMessage,
    ChatResponse,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class OpenAIMemoryClient(MemoryAugmentedClient):
    """OpenAI client with memory integration.

    Wraps the OpenAI API with automatic memory augmentation:
    - Injects relevant context from memory into prompts
    - Records interactions as experiences
    - Provides memory tools for function calling

    Example:
        >>> from silicon_memory import SiliconMemory
        >>> from silicon_memory.clients import OpenAIMemoryClient
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     client = OpenAIMemoryClient(
        ...         memory=memory,
        ...         api_key=os.environ["OPENAI_API_KEY"],
        ...         model="gpt-4",
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
        model: str = "gpt-4",
        config: MemoryClientConfig | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize the OpenAI memory client.

        Args:
            memory: SiliconMemory instance
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4)
            config: Memory client configuration
            **client_kwargs: Additional arguments for OpenAI client
        """
        super().__init__(memory, config)

        self._model = model
        self._client_kwargs = client_kwargs

        # Import and create OpenAI client
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI is required for OpenAIMemoryClient. "
                "Install with: pip install openai"
            ) from e

        self._client = AsyncOpenAI(api_key=api_key, **client_kwargs)

    async def _call_api(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Make the OpenAI API call."""
        # Convert messages to dict format
        api_messages = [msg.to_dict() for msg in messages]

        # Set defaults
        model = kwargs.pop("model", self._model)

        # Make the API call
        completion = await self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            **kwargs,
        )

        # Extract response
        choice = completion.choices[0]
        message = choice.message

        return ChatResponse(
            content=message.content or "",
            role=message.role,
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            } if completion.usage else None,
            tool_calls=[tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
        )

    async def _stream_api(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream the OpenAI API response."""
        api_messages = [msg.to_dict() for msg in messages]
        model = kwargs.pop("model", self._model)

        stream = await self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat_with_tools(
        self,
        messages: list[dict | ChatMessage],
        tools: list[dict] | None = None,
        include_memory_tools: bool = True,
        use_memory: bool = True,
        record_experience: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat with function calling support.

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
        # Build tools list
        all_tools = list(tools) if tools else []
        if include_memory_tools:
            all_tools.extend(self.get_tools())

        # Make the call with tools
        kwargs["tools"] = all_tools if all_tools else None

        return await self.chat(
            messages=messages,
            use_memory=use_memory,
            record_experience=record_experience,
            **kwargs,
        )

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

    async def embed(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector
        """
        response = await self._client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of embedding vectors
        """
        response = await self._client.embeddings.create(
            model=model,
            input=texts,
        )
        return [d.embedding for d in response.data]
