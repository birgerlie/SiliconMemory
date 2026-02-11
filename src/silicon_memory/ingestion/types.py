"""Data types for passive ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

from silicon_memory.core.utils import utc_now

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.reflection.llm import LLMProvider


@runtime_checkable
class IngestionAdapter(Protocol):
    """Protocol for all ingestion adapters.

    Adapters transform external content (transcripts, articles, etc.)
    into silicon-memory experiences and entities.
    """

    @property
    def source_type(self) -> str:
        """The type of source this adapter handles."""
        ...

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> "IngestionResult":
        """Ingest content into memory.

        Args:
            content: The raw content to ingest
            metadata: Source metadata (meeting_id, date, etc.)
            memory: The SiliconMemory instance to store into
            llm_provider: Optional LLM for enhanced extraction

        Returns:
            IngestionResult with statistics
        """
        ...


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    experiences_created: int = 0
    entities_resolved: int = 0
    decisions_detected: int = 0
    action_items_detected: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    source_type: str = ""
    ingested_at: datetime = field(default_factory=utc_now)

    @property
    def success(self) -> bool:
        """Whether the ingestion completed without fatal errors."""
        return self.experiences_created > 0 or not self.errors

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred during ingestion."""
        return len(self.errors) > 0

    def summary(self) -> str:
        """Human-readable summary of the ingestion."""
        parts = [f"Ingested from {self.source_type}:"]
        parts.append(f"  {self.experiences_created} experiences created")
        if self.entities_resolved:
            parts.append(f"  {self.entities_resolved} entities resolved")
        if self.action_items_detected:
            parts.append(f"  {self.action_items_detected} action items detected")
        if self.decisions_detected:
            parts.append(f"  {self.decisions_detected} decisions detected")
        if self.errors:
            parts.append(f"  {len(self.errors)} errors:")
            for err in self.errors[:5]:
                parts.append(f"    - {err}")
        return "\n".join(parts)


@dataclass
class IngestionConfig:
    """Base configuration for ingestion adapters."""

    max_segment_length: int = 2000
    min_segment_length: int = 50
    llm_temperature: float = 0.3
    fallback_to_heuristic: bool = True
