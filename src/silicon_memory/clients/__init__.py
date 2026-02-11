"""LLM Client Integration - Memory-augmented LLM clients.

Provides ready-to-use clients for OpenAI and Anthropic APIs
with automatic memory integration.

Example:
    >>> from silicon_memory import SiliconMemory
    >>> from silicon_memory.clients import OpenAIMemoryClient
    >>>
    >>> with SiliconMemory("/path/to/db") as memory:
    ...     client = OpenAIMemoryClient(
    ...         memory=memory,
    ...         api_key=os.environ["OPENAI_API_KEY"],
    ...     )
    ...     response = await client.chat(
    ...         messages=[{"role": "user", "content": "What is Python?"}],
    ...         use_memory=True,
    ...     )
    ...     print(response.content)
"""

from silicon_memory.clients.base import (
    MemoryAugmentedClient,
    MemoryClientConfig,
    ChatMessage,
    ChatResponse,
)
from silicon_memory.clients.openai import OpenAIMemoryClient
from silicon_memory.clients.anthropic import AnthropicMemoryClient

__all__ = [
    # Base
    "MemoryAugmentedClient",
    "MemoryClientConfig",
    "ChatMessage",
    "ChatResponse",
    # Implementations
    "OpenAIMemoryClient",
    "AnthropicMemoryClient",
]
