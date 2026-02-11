"""Forgetting service for GDPR-compliant data deletion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import UserContext

if TYPE_CHECKING:
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend


class ForgetScope(Enum):
    """Scope of forgetting operation."""

    ENTITY = "entity"  # Single entity by ID
    SESSION = "session"  # All data from a session
    TIME_RANGE = "time_range"  # All data before/after a timestamp
    QUERY = "query"  # By semantic search
    TOPIC = "topic"  # By topic/tag
    ALL = "all"  # All user data (GDPR erasure)


class ForgetStatus(Enum):
    """Status of a forget request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class ForgetRequest:
    """Request to forget data.

    Supports various scopes of forgetting:
    - Single entity by ID
    - All data from a session
    - All data before a timestamp (GDPR)
    - Selective by semantic query
    - By topic/tag
    """

    user_ctx: UserContext
    scope: ForgetScope
    request_id: str = field(default_factory=lambda: f"forget-{utc_now().timestamp()}")

    # Scope-specific parameters
    entity_id: str | None = None
    entity_type: str | None = None  # belief, experience, procedure
    session_id: str | None = None
    before_timestamp: datetime | None = None
    after_timestamp: datetime | None = None
    query: str | None = None
    topics: list[str] | None = None

    # Options
    hard_delete: bool = True  # True = actual deletion, False = soft delete (mark as deleted)
    cascade: bool = True  # Delete related data (e.g., evidence links)
    include_audit: bool = False  # Also delete audit entries (usually False for compliance)

    # Metadata
    reason: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def validate(self) -> list[str]:
        """Validate the request and return list of issues."""
        issues = []

        if self.scope == ForgetScope.ENTITY:
            if not self.entity_id:
                issues.append("entity_id is required for ENTITY scope")

        elif self.scope == ForgetScope.SESSION:
            if not self.session_id:
                issues.append("session_id is required for SESSION scope")

        elif self.scope == ForgetScope.TIME_RANGE:
            if not self.before_timestamp and not self.after_timestamp:
                issues.append("before_timestamp or after_timestamp is required for TIME_RANGE scope")

        elif self.scope == ForgetScope.QUERY:
            if not self.query:
                issues.append("query is required for QUERY scope")

        elif self.scope == ForgetScope.TOPIC:
            if not self.topics:
                issues.append("topics is required for TOPIC scope")

        return issues


@dataclass
class ForgetResult:
    """Result of a forget operation."""

    request_id: str
    status: ForgetStatus
    deleted_count: int = 0
    failed_count: int = 0
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if operation was successful."""
        return self.status == ForgetStatus.COMPLETED

    @property
    def partial_success(self) -> bool:
        """Check if operation was partially successful."""
        return self.status == ForgetStatus.PARTIALLY_COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "deleted_count": self.deleted_count,
            "failed_count": self.failed_count,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "details": self.details,
        }


class ForgettingService:
    """Service for forgetting (deleting) data.

    Provides GDPR-compliant data deletion with various scopes:
    - Entity: Delete a single entity by ID
    - Session: Delete all data from a session
    - Time range: Delete all data before/after a timestamp
    - Query: Delete by semantic search
    - Topic: Delete by topic/tag
    - All: Delete all user data (GDPR erasure request)

    Example:
        >>> service = ForgettingService(backend)
        >>>
        >>> # Forget a single entity
        >>> result = await service.forget_entity(user_ctx, "belief-123", "belief")
        >>>
        >>> # Forget all session data
        >>> result = await service.forget_session(user_ctx, "session-abc")
        >>>
        >>> # GDPR erasure
        >>> result = await service.forget_all(user_ctx)
    """

    def __init__(self, backend: "SiliconDBBackend") -> None:
        self._backend = backend

    async def forget(self, request: ForgetRequest) -> ForgetResult:
        """Execute a forget request.

        Args:
            request: The forget request

        Returns:
            ForgetResult with operation status
        """
        # Validate request
        issues = request.validate()
        if issues:
            return ForgetResult(
                request_id=request.request_id,
                status=ForgetStatus.FAILED,
                error_message=f"Invalid request: {', '.join(issues)}",
            )

        # Dispatch to appropriate handler
        handlers = {
            ForgetScope.ENTITY: self._forget_entity,
            ForgetScope.SESSION: self._forget_session,
            ForgetScope.TIME_RANGE: self._forget_time_range,
            ForgetScope.QUERY: self._forget_by_query,
            ForgetScope.TOPIC: self._forget_by_topic,
            ForgetScope.ALL: self._forget_all,
        }

        handler = handlers.get(request.scope)
        if not handler:
            return ForgetResult(
                request_id=request.request_id,
                status=ForgetStatus.FAILED,
                error_message=f"Unknown scope: {request.scope}",
            )

        try:
            return await handler(request)
        except Exception as e:
            return ForgetResult(
                request_id=request.request_id,
                status=ForgetStatus.FAILED,
                error_message=str(e),
                completed_at=utc_now(),
            )

    async def forget_entity(
        self,
        user_ctx: UserContext,
        entity_id: str,
        entity_type: str | None = None,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget a single entity by ID.

        Args:
            user_ctx: User context
            entity_id: The entity ID to delete
            entity_type: Optional entity type (belief, experience, procedure)
            reason: Optional reason for deletion

        Returns:
            ForgetResult
        """
        request = ForgetRequest(
            user_ctx=user_ctx,
            scope=ForgetScope.ENTITY,
            entity_id=entity_id,
            entity_type=entity_type,
            reason=reason,
        )
        return await self.forget(request)

    async def forget_session(
        self,
        user_ctx: UserContext,
        session_id: str,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all data from a session.

        Args:
            user_ctx: User context
            session_id: The session ID
            reason: Optional reason for deletion

        Returns:
            ForgetResult
        """
        request = ForgetRequest(
            user_ctx=user_ctx,
            scope=ForgetScope.SESSION,
            session_id=session_id,
            reason=reason,
        )
        return await self.forget(request)

    async def forget_before(
        self,
        user_ctx: UserContext,
        timestamp: datetime,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all data before a timestamp (GDPR).

        Args:
            user_ctx: User context
            timestamp: Delete everything before this time
            reason: Optional reason for deletion

        Returns:
            ForgetResult
        """
        request = ForgetRequest(
            user_ctx=user_ctx,
            scope=ForgetScope.TIME_RANGE,
            before_timestamp=timestamp,
            reason=reason,
        )
        return await self.forget(request)

    async def selective_forget(
        self,
        user_ctx: UserContext,
        query: str,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget data matching a semantic query.

        Args:
            user_ctx: User context
            query: Semantic search query
            reason: Optional reason for deletion

        Returns:
            ForgetResult
        """
        request = ForgetRequest(
            user_ctx=user_ctx,
            scope=ForgetScope.QUERY,
            query=query,
            reason=reason,
        )
        return await self.forget(request)

    async def forget_topics(
        self,
        user_ctx: UserContext,
        topics: list[str],
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all data with specified topics.

        Args:
            user_ctx: User context
            topics: List of topics to delete
            reason: Optional reason for deletion

        Returns:
            ForgetResult
        """
        request = ForgetRequest(
            user_ctx=user_ctx,
            scope=ForgetScope.TOPIC,
            topics=topics,
            reason=reason,
        )
        return await self.forget(request)

    async def forget_all(
        self,
        user_ctx: UserContext,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget all user data (GDPR erasure request).

        This is a complete data erasure operation.

        Args:
            user_ctx: User context
            reason: Optional reason for deletion

        Returns:
            ForgetResult
        """
        request = ForgetRequest(
            user_ctx=user_ctx,
            scope=ForgetScope.ALL,
            reason=reason or "GDPR erasure request",
        )
        return await self.forget(request)

    # ========== Internal Handlers ==========

    async def _forget_entity(self, request: ForgetRequest) -> ForgetResult:
        """Handle entity deletion."""
        result = ForgetResult(
            request_id=request.request_id,
            status=ForgetStatus.IN_PROGRESS,
        )

        try:
            # Build the external ID
            external_id = self._build_external_id(
                request.user_ctx,
                request.entity_id,
                request.entity_type,
            )

            # Delete from backend
            deleted = await self._delete_document(external_id, request.hard_delete)

            if deleted:
                result.deleted_count = 1
                result.status = ForgetStatus.COMPLETED
            else:
                result.status = ForgetStatus.FAILED
                result.error_message = "Entity not found or access denied"

        except Exception as e:
            result.status = ForgetStatus.FAILED
            result.error_message = str(e)

        result.completed_at = utc_now()
        return result

    async def _forget_session(self, request: ForgetRequest) -> ForgetResult:
        """Handle session data deletion."""
        result = ForgetResult(
            request_id=request.request_id,
            status=ForgetStatus.IN_PROGRESS,
        )

        try:
            # Search for all documents with this session_id
            # Use the backend to find and delete
            deleted = 0
            failed = 0

            # Query experiences with this session_id
            prefix = f"{request.user_ctx.tenant_id}/{request.user_ctx.user_id}/"
            search_results = self._backend._db.search(
                query="",
                k=10000,
                filter={"session_id": request.session_id},
            )

            for doc in search_results:
                if doc.external_id.startswith(prefix):
                    try:
                        if request.hard_delete:
                            self._backend._db.delete(doc.external_id)
                        else:
                            self._backend._db.update(
                                doc.external_id,
                                metadata={"deleted": True, "deleted_at": utc_now().isoformat()},
                            )
                        deleted += 1
                    except Exception:
                        failed += 1

            result.deleted_count = deleted
            result.failed_count = failed
            result.status = (
                ForgetStatus.COMPLETED if failed == 0
                else ForgetStatus.PARTIALLY_COMPLETED if deleted > 0
                else ForgetStatus.FAILED
            )

        except Exception as e:
            result.status = ForgetStatus.FAILED
            result.error_message = str(e)

        result.completed_at = utc_now()
        return result

    async def _forget_time_range(self, request: ForgetRequest) -> ForgetResult:
        """Handle time-range deletion."""
        result = ForgetResult(
            request_id=request.request_id,
            status=ForgetStatus.IN_PROGRESS,
        )

        try:
            deleted = 0
            failed = 0

            # Search for all user documents
            prefix = f"{request.user_ctx.tenant_id}/{request.user_ctx.user_id}/"
            search_results = self._backend._db.search(query="", k=10000)

            for doc in search_results:
                if not doc.external_id.startswith(prefix):
                    continue

                # Check timestamp
                metadata = doc.metadata or {}
                created_at_str = metadata.get("created_at") or metadata.get("observed_at") or metadata.get("occurred_at")

                if not created_at_str:
                    continue

                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except (ValueError, TypeError):
                    continue

                should_delete = False
                if request.before_timestamp and created_at < request.before_timestamp:
                    should_delete = True
                if request.after_timestamp and created_at > request.after_timestamp:
                    should_delete = True

                if should_delete:
                    try:
                        if request.hard_delete:
                            self._backend._db.delete(doc.external_id)
                        else:
                            self._backend._db.update(
                                doc.external_id,
                                metadata={"deleted": True, "deleted_at": utc_now().isoformat()},
                            )
                        deleted += 1
                    except Exception:
                        failed += 1

            result.deleted_count = deleted
            result.failed_count = failed
            result.status = (
                ForgetStatus.COMPLETED if failed == 0
                else ForgetStatus.PARTIALLY_COMPLETED if deleted > 0
                else ForgetStatus.FAILED
            )

        except Exception as e:
            result.status = ForgetStatus.FAILED
            result.error_message = str(e)

        result.completed_at = utc_now()
        return result

    async def _forget_by_query(self, request: ForgetRequest) -> ForgetResult:
        """Handle query-based deletion."""
        result = ForgetResult(
            request_id=request.request_id,
            status=ForgetStatus.IN_PROGRESS,
        )

        try:
            deleted = 0
            failed = 0

            # Semantic search
            prefix = f"{request.user_ctx.tenant_id}/{request.user_ctx.user_id}/"
            search_results = self._backend._db.search(
                query=request.query or "",
                k=1000,
            )

            for doc in search_results:
                if not doc.external_id.startswith(prefix):
                    continue

                try:
                    if request.hard_delete:
                        self._backend._db.delete(doc.external_id)
                    else:
                        self._backend._db.update(
                            doc.external_id,
                            metadata={"deleted": True, "deleted_at": utc_now().isoformat()},
                        )
                    deleted += 1
                except Exception:
                    failed += 1

            result.deleted_count = deleted
            result.failed_count = failed
            result.status = (
                ForgetStatus.COMPLETED if failed == 0
                else ForgetStatus.PARTIALLY_COMPLETED if deleted > 0
                else ForgetStatus.FAILED
            )

        except Exception as e:
            result.status = ForgetStatus.FAILED
            result.error_message = str(e)

        result.completed_at = utc_now()
        return result

    async def _forget_by_topic(self, request: ForgetRequest) -> ForgetResult:
        """Handle topic-based deletion."""
        result = ForgetResult(
            request_id=request.request_id,
            status=ForgetStatus.IN_PROGRESS,
        )

        try:
            deleted = 0
            failed = 0

            # Search for documents with matching tags
            prefix = f"{request.user_ctx.tenant_id}/{request.user_ctx.user_id}/"
            topics_lower = {t.lower() for t in (request.topics or [])}

            search_results = self._backend._db.search(query="", k=10000)

            for doc in search_results:
                if not doc.external_id.startswith(prefix):
                    continue

                # Check tags
                metadata = doc.metadata or {}
                tags = metadata.get("tags", [])
                if isinstance(tags, set):
                    tags = list(tags)

                doc_tags_lower = {t.lower() for t in tags}

                if doc_tags_lower & topics_lower:
                    try:
                        if request.hard_delete:
                            self._backend._db.delete(doc.external_id)
                        else:
                            self._backend._db.update(
                                doc.external_id,
                                metadata={"deleted": True, "deleted_at": utc_now().isoformat()},
                            )
                        deleted += 1
                    except Exception:
                        failed += 1

            result.deleted_count = deleted
            result.failed_count = failed
            result.status = (
                ForgetStatus.COMPLETED if failed == 0
                else ForgetStatus.PARTIALLY_COMPLETED if deleted > 0
                else ForgetStatus.FAILED
            )

        except Exception as e:
            result.status = ForgetStatus.FAILED
            result.error_message = str(e)

        result.completed_at = utc_now()
        return result

    async def _forget_all(self, request: ForgetRequest) -> ForgetResult:
        """Handle complete user data erasure."""
        result = ForgetResult(
            request_id=request.request_id,
            status=ForgetStatus.IN_PROGRESS,
        )

        try:
            deleted = 0
            failed = 0

            # Find all documents for this user
            prefix = f"{request.user_ctx.tenant_id}/{request.user_ctx.user_id}/"
            search_results = self._backend._db.search(query="", k=100000)

            for doc in search_results:
                if not doc.external_id.startswith(prefix):
                    continue

                try:
                    if request.hard_delete:
                        self._backend._db.delete(doc.external_id)
                    else:
                        self._backend._db.update(
                            doc.external_id,
                            metadata={"deleted": True, "deleted_at": utc_now().isoformat()},
                        )
                    deleted += 1
                except Exception:
                    failed += 1

            result.deleted_count = deleted
            result.failed_count = failed
            result.status = (
                ForgetStatus.COMPLETED if failed == 0
                else ForgetStatus.PARTIALLY_COMPLETED if deleted > 0
                else ForgetStatus.FAILED
            )
            result.details["erasure_type"] = "complete"

        except Exception as e:
            result.status = ForgetStatus.FAILED
            result.error_message = str(e)

        result.completed_at = utc_now()
        return result

    def _build_external_id(
        self,
        user_ctx: UserContext,
        entity_id: str | None,
        entity_type: str | None,
    ) -> str:
        """Build external ID for an entity."""
        # If entity_id already has the full format, use it
        if entity_id and "/" in entity_id:
            return entity_id

        # Build the new format
        prefix = f"{user_ctx.tenant_id}/{user_ctx.user_id}/"

        if entity_type and entity_id:
            return f"{prefix}{entity_type}-{entity_id}"
        elif entity_id:
            return f"{prefix}{entity_id}"
        else:
            raise ValueError("entity_id is required")

    async def _delete_document(self, external_id: str, hard_delete: bool) -> bool:
        """Delete a document from the backend."""
        try:
            if hard_delete:
                self._backend._db.delete(external_id)
            else:
                self._backend._db.update(
                    external_id,
                    metadata={"deleted": True, "deleted_at": utc_now().isoformat()},
                )
            return True
        except Exception:
            return False
