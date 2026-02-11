"""Transparency service for "why do you know" queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import UserContext

if TYPE_CHECKING:
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend


class ProvenanceType(Enum):
    """Type of provenance event."""

    CREATED = "created"
    UPDATED = "updated"
    ACCESSED = "accessed"
    DERIVED = "derived"  # Created from another memory
    IMPORTED = "imported"
    EXTERNAL = "external"  # From external source


@dataclass
class ProvenanceEvent:
    """A single event in the provenance chain."""

    event_type: ProvenanceType
    timestamp: datetime
    actor_id: str  # User or system that performed the action
    actor_type: str  # user, system, llm, external
    source_id: str | None = None  # For derived/imported events
    source_name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceEvent":
        """Create from dictionary."""
        return cls(
            event_type=ProvenanceType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            actor_id=data["actor_id"],
            actor_type=data["actor_type"],
            source_id=data.get("source_id"),
            source_name=data.get("source_name"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AccessLogEntry:
    """An entry in the access log."""

    entity_id: str
    entity_type: str
    accessed_by: str
    accessed_at: datetime
    access_type: str  # read, write, delete, export
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "accessed_by": self.accessed_by,
            "accessed_at": self.accessed_at.isoformat(),
            "access_type": self.access_type,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessLogEntry":
        """Create from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            accessed_by=data["accessed_by"],
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_type=data["access_type"],
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProvenanceChain:
    """Complete provenance chain for a memory item.

    Answers "why do you know this?" by tracing the origin
    and history of a piece of knowledge.
    """

    entity_id: str
    entity_type: str
    content_summary: str
    events: list[ProvenanceEvent] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: datetime | None = None
    created_at: datetime | None = None
    created_by: str | None = None

    @property
    def origin(self) -> ProvenanceEvent | None:
        """Get the original creation event."""
        for event in self.events:
            if event.event_type == ProvenanceType.CREATED:
                return event
        return self.events[0] if self.events else None

    @property
    def sources(self) -> list[str]:
        """Get all source IDs in the chain."""
        return [e.source_id for e in self.events if e.source_id]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "content_summary": self.content_summary,
            "events": [e.to_dict() for e in self.events],
            "related_entities": self.related_entities,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }

    def as_narrative(self) -> str:
        """Generate a human-readable narrative of the provenance."""
        if not self.events:
            return f"No provenance information available for '{self.content_summary}'."

        lines = [f"Provenance for: {self.content_summary}"]
        lines.append("-" * 50)

        origin = self.origin
        if origin:
            lines.append(f"Originally {origin.event_type.value} by {origin.actor_type} "
                        f"'{origin.actor_id}' on {origin.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if origin.source_name:
                lines.append(f"  Source: {origin.source_name}")

        lines.append("")
        lines.append("Event history:")
        for event in self.events:
            timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{timestamp}] {event.event_type.value} by {event.actor_id}")
            if event.description:
                lines.append(f"      {event.description}")

        if self.related_entities:
            lines.append("")
            lines.append(f"Related to: {', '.join(self.related_entities[:5])}")
            if len(self.related_entities) > 5:
                lines.append(f"  ... and {len(self.related_entities) - 5} more")

        lines.append("")
        lines.append(f"Access count: {self.access_count}")

        return "\n".join(lines)


class TransparencyService:
    """Service for transparency and provenance tracking.

    Provides "why do you know" functionality by tracking:
    - Where memories come from
    - How they've been modified
    - Who has accessed them
    - Related memories

    Example:
        >>> service = TransparencyService(backend)
        >>>
        >>> # Why do you know about Python?
        >>> chains = await service.why_do_you_know(user_ctx, "Python")
        >>> for chain in chains:
        ...     print(chain.as_narrative())
        >>>
        >>> # Get access log for a specific entity
        >>> log = await service.get_access_log(user_ctx, "belief-123")
    """

    def __init__(self, backend: "SiliconDBBackend") -> None:
        self._backend = backend
        self._access_log: list[AccessLogEntry] = []  # In-memory for now

    async def why_do_you_know(
        self,
        user_ctx: UserContext,
        query: str,
        limit: int = 10,
    ) -> list[ProvenanceChain]:
        """Answer "why do you know about X?".

        Searches for memories matching the query and returns
        their provenance chains.

        Args:
            user_ctx: User context
            query: The topic/query to explain
            limit: Maximum number of chains to return

        Returns:
            List of ProvenanceChain objects
        """
        chains = []

        # Search for matching memories
        prefix = f"{user_ctx.tenant_id}/{user_ctx.user_id}/"
        search_results = self._backend._db.search(query=query, k=limit * 2)

        for doc in search_results:
            if not doc.external_id.startswith(prefix):
                continue

            chain = await self._build_provenance_chain(doc)
            if chain:
                chains.append(chain)

            if len(chains) >= limit:
                break

        return chains

    async def get_provenance(
        self,
        user_ctx: UserContext,
        entity_id: str,
    ) -> ProvenanceChain | None:
        """Get the provenance chain for a specific entity.

        Args:
            user_ctx: User context
            entity_id: The entity ID

        Returns:
            ProvenanceChain or None if not found
        """
        # Build full external ID if needed
        if "/" not in entity_id:
            entity_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_id}"

        try:
            doc = self._backend._db.get(entity_id)
            if doc:
                return await self._build_provenance_chain_from_doc(doc, entity_id)
        except Exception:
            pass

        return None

    async def get_access_log(
        self,
        user_ctx: UserContext,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[AccessLogEntry]:
        """Get the access log for a user or specific entity.

        Args:
            user_ctx: User context
            entity_id: Optional entity ID to filter by
            limit: Maximum entries to return

        Returns:
            List of AccessLogEntry objects
        """
        # Filter access log
        entries = []
        for entry in reversed(self._access_log):
            if entry.accessed_by != user_ctx.user_id:
                # Only show own access log unless admin
                if not user_ctx.is_admin():
                    continue

            if entity_id and entry.entity_id != entity_id:
                continue

            entries.append(entry)
            if len(entries) >= limit:
                break

        return entries

    async def log_access(
        self,
        user_ctx: UserContext,
        entity_id: str,
        entity_type: str,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an access event.

        Args:
            user_ctx: User context
            entity_id: The entity being accessed
            entity_type: Type of entity
            access_type: Type of access (read, write, delete, export)
            metadata: Additional metadata
        """
        entry = AccessLogEntry(
            entity_id=entity_id,
            entity_type=entity_type,
            accessed_by=user_ctx.user_id,
            accessed_at=utc_now(),
            access_type=access_type,
            session_id=user_ctx.session_id,
            metadata=metadata or {},
        )
        self._access_log.append(entry)

        # Also update the entity's access count
        try:
            full_id = entity_id
            if "/" not in entity_id:
                full_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_id}"

            doc = self._backend._db.get(full_id)
            if doc:
                current_metadata = doc.get("metadata", {})
                current_metadata["access_count"] = current_metadata.get("access_count", 0) + 1
                current_metadata["last_accessed"] = utc_now().isoformat()
                self._backend._db.update(full_id, metadata=current_metadata)
        except Exception:
            pass

    async def add_provenance_event(
        self,
        user_ctx: UserContext,
        entity_id: str,
        event: ProvenanceEvent,
    ) -> bool:
        """Add a provenance event to an entity.

        Args:
            user_ctx: User context
            entity_id: The entity ID
            event: The provenance event to add

        Returns:
            True if successful
        """
        try:
            full_id = entity_id
            if "/" not in entity_id:
                full_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_id}"

            doc = self._backend._db.get(full_id)
            if not doc:
                return False

            metadata = doc.get("metadata", {})
            provenance = metadata.get("provenance", [])
            provenance.append(event.to_dict())
            metadata["provenance"] = provenance

            self._backend._db.update(full_id, metadata=metadata)
            return True
        except Exception:
            return False

    async def get_related_entities(
        self,
        user_ctx: UserContext,
        entity_id: str,
        limit: int = 20,
    ) -> list[str]:
        """Get entities related to the given entity.

        Uses the graph layer to find connections.

        Args:
            user_ctx: User context
            entity_id: The entity ID
            limit: Maximum related entities

        Returns:
            List of related entity IDs
        """
        related = []

        try:
            full_id = entity_id
            if "/" not in entity_id:
                full_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_id}"

            # Get neighbors from graph
            neighbors = self._backend._db.get_neighbors(full_id, k=limit)
            prefix = f"{user_ctx.tenant_id}/{user_ctx.user_id}/"

            for neighbor in neighbors:
                if neighbor.external_id.startswith(prefix):
                    related.append(neighbor.external_id)
        except Exception:
            pass

        return related

    async def _build_provenance_chain(self, doc: Any) -> ProvenanceChain | None:
        """Build provenance chain from a search result document."""
        try:
            return await self._build_provenance_chain_from_doc(
                {"metadata": doc.metadata, "text": doc.text},
                doc.external_id,
            )
        except Exception:
            return None

    async def _build_provenance_chain_from_doc(
        self,
        doc: dict[str, Any],
        external_id: str,
    ) -> ProvenanceChain | None:
        """Build provenance chain from a document."""
        try:
            metadata = doc.get("metadata", {})

            # Extract entity type from external_id
            parts = external_id.split("/")
            entity_part = parts[-1] if parts else external_id
            entity_type = entity_part.split("-")[0] if "-" in entity_part else "unknown"

            # Get content summary
            content = doc.get("text", "")
            content_summary = content[:100] + "..." if len(content) > 100 else content

            # Build events from provenance metadata
            events = []
            provenance_data = metadata.get("provenance", [])
            for event_data in provenance_data:
                try:
                    events.append(ProvenanceEvent.from_dict(event_data))
                except Exception:
                    pass

            # If no provenance, create initial event from metadata
            if not events:
                created_at_str = metadata.get("created_at") or metadata.get("observed_at")
                created_by = metadata.get("created_by") or metadata.get("owner_id")

                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except (ValueError, TypeError):
                        created_at = utc_now()
                else:
                    created_at = utc_now()

                # Determine source
                source_type = ProvenanceType.CREATED
                source_name = None
                source_id = metadata.get("source_id")

                if metadata.get("source"):
                    source_info = metadata["source"]
                    if isinstance(source_info, dict):
                        source_name = source_info.get("name") or source_info.get("id")
                        if source_info.get("type") == "external":
                            source_type = ProvenanceType.EXTERNAL

                events.append(ProvenanceEvent(
                    event_type=source_type,
                    timestamp=created_at,
                    actor_id=created_by or "unknown",
                    actor_type="user" if created_by else "system",
                    source_id=source_id,
                    source_name=source_name,
                ))

            # Get related entities
            related = []
            evidence_for = metadata.get("evidence_for", [])
            evidence_against = metadata.get("evidence_against", [])
            related.extend(evidence_for[:5])
            related.extend(evidence_against[:5])

            # Build chain
            created_at = None
            created_by = None
            for event in events:
                if event.event_type == ProvenanceType.CREATED:
                    created_at = event.timestamp
                    created_by = event.actor_id
                    break

            return ProvenanceChain(
                entity_id=external_id,
                entity_type=entity_type,
                content_summary=content_summary,
                events=events,
                related_entities=related,
                access_count=metadata.get("access_count", 0),
                last_accessed=datetime.fromisoformat(metadata["last_accessed"]) if metadata.get("last_accessed") else None,
                created_at=created_at,
                created_by=created_by,
            )

        except Exception:
            return None
