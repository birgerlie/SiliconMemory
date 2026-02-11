"""Memory inspector for export/import/correct operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import UserContext, PrivacyMetadata

if TYPE_CHECKING:
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend


class ExportFormat(Enum):
    """Format for data export."""

    JSON = "json"
    JSONL = "jsonl"  # JSON Lines


@dataclass
class MemoryRecord:
    """A single memory record for export/import.

    Standardized format for data portability.
    """

    external_id: str
    entity_type: str  # belief, experience, procedure, working
    content: str
    metadata: dict[str, Any]
    privacy: PrivacyMetadata | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "external_id": self.external_id,
            "entity_type": self.entity_type,
            "content": self.content,
            "metadata": self.metadata,
            "privacy": self.privacy.to_dict() if self.privacy else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Create from dictionary."""
        privacy = None
        if data.get("privacy"):
            privacy = PrivacyMetadata.from_dict(data["privacy"])

        created_at = utc_now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        updated_at = None
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])

        return cls(
            external_id=data["external_id"],
            entity_type=data["entity_type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            privacy=privacy,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class MemoryInspection:
    """Overview of a user's memory contents."""

    user_id: str
    tenant_id: str
    inspected_at: datetime = field(default_factory=utc_now)

    # Counts by type
    belief_count: int = 0
    experience_count: int = 0
    procedure_count: int = 0
    working_count: int = 0

    # Privacy breakdown
    private_count: int = 0
    workspace_count: int = 0
    public_count: int = 0

    # Classification breakdown
    classification_counts: dict[str, int] = field(default_factory=dict)

    # Storage stats
    total_items: int = 0
    oldest_item: datetime | None = None
    newest_item: datetime | None = None

    # Tags
    top_tags: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "inspected_at": self.inspected_at.isoformat(),
            "counts": {
                "beliefs": self.belief_count,
                "experiences": self.experience_count,
                "procedures": self.procedure_count,
                "working": self.working_count,
                "total": self.total_items,
            },
            "privacy": {
                "private": self.private_count,
                "workspace": self.workspace_count,
                "public": self.public_count,
            },
            "classifications": self.classification_counts,
            "time_range": {
                "oldest": self.oldest_item.isoformat() if self.oldest_item else None,
                "newest": self.newest_item.isoformat() if self.newest_item else None,
            },
            "top_tags": self.top_tags,
        }

    def as_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Memory Inspection for {self.user_id}",
            "=" * 50,
            f"Inspected at: {self.inspected_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Item Counts:",
            f"  Beliefs: {self.belief_count}",
            f"  Experiences: {self.experience_count}",
            f"  Procedures: {self.procedure_count}",
            f"  Working Memory: {self.working_count}",
            f"  Total: {self.total_items}",
            "",
            "Privacy Levels:",
            f"  Private: {self.private_count}",
            f"  Workspace: {self.workspace_count}",
            f"  Public: {self.public_count}",
        ]

        if self.classification_counts:
            lines.append("")
            lines.append("Classifications:")
            for cls, count in sorted(self.classification_counts.items()):
                lines.append(f"  {cls}: {count}")

        if self.oldest_item or self.newest_item:
            lines.append("")
            lines.append("Time Range:")
            if self.oldest_item:
                lines.append(f"  Oldest: {self.oldest_item.strftime('%Y-%m-%d')}")
            if self.newest_item:
                lines.append(f"  Newest: {self.newest_item.strftime('%Y-%m-%d')}")

        if self.top_tags:
            lines.append("")
            lines.append("Top Tags:")
            for tag, count in self.top_tags[:10]:
                lines.append(f"  {tag}: {count}")

        return "\n".join(lines)


@dataclass
class CorrectionResult:
    """Result of a memory correction operation."""

    entity_id: str
    success: bool
    original_content: str | None = None
    corrected_content: str | None = None
    corrections_applied: list[str] = field(default_factory=list)
    error_message: str | None = None
    corrected_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "success": self.success,
            "original_content": self.original_content,
            "corrected_content": self.corrected_content,
            "corrections_applied": self.corrections_applied,
            "error_message": self.error_message,
            "corrected_at": self.corrected_at.isoformat(),
        }


class MemoryInspector:
    """Service for inspecting, exporting, importing, and correcting memories.

    Provides data portability and control over stored memories:
    - Inspect: See an overview of what's stored
    - Export: Download all user data (GDPR data portability)
    - Import: Restore previously exported data
    - Correct: Edit specific memories

    Example:
        >>> inspector = MemoryInspector(backend)
        >>>
        >>> # Get overview
        >>> inspection = await inspector.inspect_memories(user_ctx)
        >>> print(inspection.as_summary())
        >>>
        >>> # Export all data
        >>> async for record in inspector.export_memories(user_ctx):
        ...     save_to_file(record)
        >>>
        >>> # Import data
        >>> await inspector.import_memories(user_ctx, records)
        >>>
        >>> # Correct a memory
        >>> result = await inspector.correct_memory(
        ...     user_ctx, "belief-123", {"content": "corrected text"}
        ... )
    """

    def __init__(self, backend: "SiliconDBBackend") -> None:
        self._backend = backend

    async def inspect_memories(
        self,
        user_ctx: UserContext,
    ) -> MemoryInspection:
        """Get an overview of the user's memory contents.

        Args:
            user_ctx: User context

        Returns:
            MemoryInspection with statistics
        """
        inspection = MemoryInspection(
            user_id=user_ctx.user_id,
            tenant_id=user_ctx.tenant_id,
        )

        prefix = f"{user_ctx.tenant_id}/{user_ctx.user_id}/"
        tag_counts: dict[str, int] = {}

        try:
            # Search all user documents
            search_results = self._backend._db.search(query="", k=100000)

            for doc in search_results:
                if not doc.external_id.startswith(prefix):
                    continue

                inspection.total_items += 1
                metadata = doc.metadata or {}

                # Count by type
                if "/belief-" in doc.external_id or doc.node_type == "belief":
                    inspection.belief_count += 1
                elif "/experience-" in doc.external_id or doc.node_type == "experience":
                    inspection.experience_count += 1
                elif "/procedure-" in doc.external_id or doc.node_type == "procedure":
                    inspection.procedure_count += 1
                elif "/working-" in doc.external_id or doc.node_type == "working":
                    inspection.working_count += 1

                # Count by privacy level
                privacy_level = metadata.get("privacy_level", "private")
                if privacy_level == "private":
                    inspection.private_count += 1
                elif privacy_level == "workspace":
                    inspection.workspace_count += 1
                elif privacy_level == "public":
                    inspection.public_count += 1

                # Count by classification
                classification = metadata.get("classification", "internal")
                inspection.classification_counts[classification] = (
                    inspection.classification_counts.get(classification, 0) + 1
                )

                # Track timestamps
                created_at_str = (
                    metadata.get("created_at") or
                    metadata.get("observed_at") or
                    metadata.get("occurred_at")
                )
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                        if inspection.oldest_item is None or created_at < inspection.oldest_item:
                            inspection.oldest_item = created_at
                        if inspection.newest_item is None or created_at > inspection.newest_item:
                            inspection.newest_item = created_at
                    except (ValueError, TypeError):
                        pass

                # Count tags
                tags = metadata.get("tags", [])
                if isinstance(tags, set):
                    tags = list(tags)
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Get top tags
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            inspection.top_tags = sorted_tags[:20]

        except Exception:
            pass

        return inspection

    async def export_memories(
        self,
        user_ctx: UserContext,
        entity_types: list[str] | None = None,
    ) -> AsyncIterator[MemoryRecord]:
        """Export user's memories as an async iterator.

        Args:
            user_ctx: User context
            entity_types: Optional filter by entity types

        Yields:
            MemoryRecord objects
        """
        prefix = f"{user_ctx.tenant_id}/{user_ctx.user_id}/"
        valid_types = set(entity_types) if entity_types else None

        try:
            search_results = self._backend._db.search(query="", k=100000)

            for doc in search_results:
                if not doc.external_id.startswith(prefix):
                    continue

                # Determine entity type
                entity_type = self._get_entity_type(doc.external_id, doc.node_type)

                if valid_types and entity_type not in valid_types:
                    continue

                # Build privacy metadata
                metadata = doc.metadata or {}
                privacy = None
                if metadata.get("owner_id") and metadata.get("tenant_id"):
                    try:
                        privacy = PrivacyMetadata.from_dict(metadata)
                    except Exception:
                        pass

                # Get timestamps
                created_at_str = (
                    metadata.get("created_at") or
                    metadata.get("observed_at") or
                    metadata.get("occurred_at")
                )
                created_at = utc_now()
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except (ValueError, TypeError):
                        pass

                updated_at = None
                if metadata.get("updated_at"):
                    try:
                        updated_at = datetime.fromisoformat(metadata["updated_at"])
                    except (ValueError, TypeError):
                        pass

                yield MemoryRecord(
                    external_id=doc.external_id,
                    entity_type=entity_type,
                    content=doc.text or "",
                    metadata=metadata,
                    privacy=privacy,
                    created_at=created_at,
                    updated_at=updated_at,
                )

        except Exception:
            pass

    async def import_memories(
        self,
        user_ctx: UserContext,
        records: list[MemoryRecord],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Import memories from records.

        Args:
            user_ctx: User context
            records: List of MemoryRecord objects to import
            overwrite: Whether to overwrite existing records

        Returns:
            Summary of import operation
        """
        imported = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        for record in records:
            try:
                # Build external ID with user's prefix
                parts = record.external_id.split("/")
                entity_part = parts[-1] if parts else record.external_id
                new_external_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_part}"

                # Check if exists
                exists = False
                try:
                    existing = self._backend._db.get(new_external_id)
                    exists = existing is not None
                except Exception:
                    pass

                if exists and not overwrite:
                    skipped += 1
                    continue

                # Build metadata
                metadata = dict(record.metadata)
                metadata["owner_id"] = user_ctx.user_id
                metadata["tenant_id"] = user_ctx.tenant_id
                metadata["imported_at"] = utc_now().isoformat()
                metadata["original_external_id"] = record.external_id

                # Import
                if exists:
                    self._backend._db.update(
                        new_external_id,
                        text=record.content,
                        metadata=metadata,
                    )
                else:
                    self._backend._db.ingest(
                        external_id=new_external_id,
                        text=record.content,
                        metadata=metadata,
                        node_type=record.entity_type,
                    )

                imported += 1

            except Exception as e:
                failed += 1
                errors.append(f"{record.external_id}: {str(e)}")

        return {
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "errors": errors[:10],  # Limit errors
            "total": len(records),
        }

    async def correct_memory(
        self,
        user_ctx: UserContext,
        entity_id: str,
        corrections: dict[str, Any],
    ) -> CorrectionResult:
        """Correct a specific memory.

        Args:
            user_ctx: User context
            entity_id: The entity to correct
            corrections: Dictionary of fields to correct

        Returns:
            CorrectionResult
        """
        # Build full external ID if needed
        if "/" not in entity_id:
            entity_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_id}"

        result = CorrectionResult(entity_id=entity_id, success=False)

        try:
            # Get existing document
            doc = self._backend._db.get(entity_id)
            if not doc:
                result.error_message = "Entity not found"
                return result

            # Verify ownership
            metadata = doc.get("metadata", {})
            if metadata.get("owner_id") and metadata["owner_id"] != user_ctx.user_id:
                if not user_ctx.is_admin():
                    result.error_message = "Access denied"
                    return result

            result.original_content = doc.get("text")

            # Apply corrections
            new_text = doc.get("text")
            new_metadata = dict(metadata)
            applied = []

            if "content" in corrections:
                new_text = corrections["content"]
                applied.append("content")
                result.corrected_content = new_text

            # Apply metadata corrections
            for key in ["confidence", "status", "tags"]:
                if key in corrections:
                    new_metadata[key] = corrections[key]
                    applied.append(key)

            # Record the correction
            new_metadata["corrected_at"] = utc_now().isoformat()
            new_metadata["corrected_by"] = user_ctx.user_id
            new_metadata["updated_at"] = utc_now().isoformat()

            # Update
            self._backend._db.update(
                entity_id,
                text=new_text,
                metadata=new_metadata,
            )

            result.success = True
            result.corrections_applied = applied

        except Exception as e:
            result.error_message = str(e)

        return result

    async def get_memory(
        self,
        user_ctx: UserContext,
        entity_id: str,
    ) -> MemoryRecord | None:
        """Get a specific memory record.

        Args:
            user_ctx: User context
            entity_id: The entity ID

        Returns:
            MemoryRecord or None
        """
        # Build full external ID if needed
        if "/" not in entity_id:
            entity_id = f"{user_ctx.tenant_id}/{user_ctx.user_id}/{entity_id}"

        try:
            doc = self._backend._db.get(entity_id)
            if not doc:
                return None

            metadata = doc.get("metadata", {})

            # Verify access
            if metadata.get("owner_id") and metadata["owner_id"] != user_ctx.user_id:
                if not user_ctx.is_admin():
                    # Check if shared or workspace/public
                    privacy_level = metadata.get("privacy_level", "private")
                    shared_with = metadata.get("shared_with", [])

                    if privacy_level == "private" and user_ctx.user_id not in shared_with:
                        return None

            entity_type = self._get_entity_type(entity_id, doc.get("node_type"))

            privacy = None
            if metadata.get("owner_id") and metadata.get("tenant_id"):
                try:
                    privacy = PrivacyMetadata.from_dict(metadata)
                except Exception:
                    pass

            created_at = utc_now()
            created_at_str = metadata.get("created_at") or metadata.get("observed_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except (ValueError, TypeError):
                    pass

            return MemoryRecord(
                external_id=entity_id,
                entity_type=entity_type,
                content=doc.get("text", ""),
                metadata=metadata,
                privacy=privacy,
                created_at=created_at,
            )

        except Exception:
            return None

    def _get_entity_type(self, external_id: str, node_type: str | None) -> str:
        """Determine entity type from ID or node_type."""
        if node_type:
            return node_type

        if "/belief-" in external_id or external_id.startswith("belief-"):
            return "belief"
        elif "/experience-" in external_id or external_id.startswith("experience-"):
            return "experience"
        elif "/procedure-" in external_id or external_id.startswith("procedure-"):
            return "procedure"
        elif "/working-" in external_id or external_id.startswith("working-"):
            return "working"

        return "unknown"
