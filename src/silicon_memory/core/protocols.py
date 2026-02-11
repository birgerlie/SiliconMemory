"""Protocols (interfaces) for Silicon Memory components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from silicon_memory.core.types import (
        Belief,
        Experience,
        Procedure,
        RecallResult,
    )


@runtime_checkable
class MemoryStore(Protocol):
    """Abstract storage backend protocol."""

    async def store(self, key: str, value: dict[str, Any]) -> None:
        """Store a value by key."""
        ...

    async def retrieve(self, key: str) -> dict[str, Any] | None:
        """Retrieve a value by key."""
        ...

    async def query(self, filter_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Query values by filter."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value by key. Returns True if deleted."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...


@runtime_checkable
class SemanticMemory(Protocol):
    """Stores facts and beliefs as triplets."""

    async def commit(self, belief: Belief) -> None:
        """Store a belief."""
        ...

    async def get(self, belief_id: str) -> Belief | None:
        """Get a belief by ID."""
        ...

    async def query(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[Belief]:
        """Query beliefs by semantic similarity or keyword."""
        ...

    async def get_by_entity(self, entity: str) -> list[Belief]:
        """Get all beliefs about an entity."""
        ...

    async def find_contradictions(self, belief: Belief) -> list[Belief]:
        """Find beliefs that contradict the given belief."""
        ...

    async def update_confidence(self, belief_id: str, delta: float) -> None:
        """Update a belief's confidence."""
        ...


@runtime_checkable
class EpisodicMemory(Protocol):
    """Stores experiences and events."""

    async def record(self, experience: Experience) -> None:
        """Record an experience."""
        ...

    async def get(self, experience_id: str) -> Experience | None:
        """Get an experience by ID."""
        ...

    async def query(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Experience]:
        """Query experiences by similarity."""
        ...

    async def get_recent(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[Experience]:
        """Get recent experiences."""
        ...

    async def get_unprocessed(self, limit: int = 100) -> list[Experience]:
        """Get experiences not yet processed by reflection."""
        ...

    async def mark_processed(self, experience_ids: list[str]) -> None:
        """Mark experiences as processed."""
        ...


@runtime_checkable
class WorkingMemory(Protocol):
    """Short-term context store with TTL."""

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set a value with TTL."""
        ...

    def get(self, key: str) -> Any | None:
        """Get a value if not expired."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a value."""
        ...

    def clear(self) -> None:
        """Clear all values."""
        ...

    def get_context(self) -> dict[str, Any]:
        """Get all non-expired values as context."""
        ...

    def keys(self) -> list[str]:
        """Get all non-expired keys."""
        ...


@runtime_checkable
class ProceduralMemory(Protocol):
    """Stores how-to knowledge."""

    async def commit(self, procedure: Procedure) -> None:
        """Store a procedure."""
        ...

    async def get(self, procedure_id: str) -> Procedure | None:
        """Get a procedure by ID."""
        ...

    async def find_applicable(
        self,
        context: str,
        limit: int = 5,
    ) -> list[Procedure]:
        """Find procedures applicable to the context."""
        ...

    async def record_outcome(
        self,
        procedure_id: str,
        success: bool,
    ) -> None:
        """Record an outcome for a procedure."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract LLM interface."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
    ) -> Any:
        """Generate structured output matching schema."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


@runtime_checkable
class MemoryTool(Protocol):
    """Tool interface for LLM function calling."""

    @property
    def name(self) -> str:
        """Tool name for function calling."""
        ...

    @property
    def description(self) -> str:
        """Tool description for LLM."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        ...

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with given parameters."""
        ...


@runtime_checkable
class ExternalKnowledgeSource(Protocol):
    """External knowledge source for verification and enrichment."""

    @property
    def name(self) -> str:
        """Source name."""
        ...

    @property
    def reliability(self) -> float:
        """Source reliability (0.0-1.0)."""
        ...

    async def query(self, query: str) -> list[dict[str, Any]]:
        """Query the external source."""
        ...

    async def verify(self, claim: str) -> dict[str, Any]:
        """Verify a claim against the source."""
        ...
