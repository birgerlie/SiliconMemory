"""Audit logging for Silicon Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import UserContext

if TYPE_CHECKING:
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend


class AuditAction(Enum):
    """Types of auditable actions."""

    # Memory operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Batch operations
    QUERY = "query"
    SEARCH = "search"
    RECALL = "recall"

    # Privacy operations
    SHARE = "share"
    UNSHARE = "unshare"
    EXPORT = "export"
    IMPORT = "import"

    # Forget operations
    FORGET = "forget"
    FORGET_SESSION = "forget_session"
    FORGET_ALL = "forget_all"

    # Admin operations
    POLICY_CHANGE = "policy_change"
    CONFIG_CHANGE = "config_change"

    # Access control
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"

    # Consent
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"


class AuditSeverity(Enum):
    """Severity level for audit entries."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=utc_now)

    # Who
    user_id: str = ""
    tenant_id: str = ""
    session_id: str | None = None

    # What
    action: AuditAction = AuditAction.READ
    severity: AuditSeverity = AuditSeverity.INFO

    # Where
    resource_id: str | None = None
    resource_type: str | None = None

    # Result
    success: bool = True
    error_message: str | None = None

    # Details
    details: dict[str, Any] = field(default_factory=dict)

    # Request context
    ip_address: str | None = None
    user_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "action": self.action.value,
            "severity": self.severity.value,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "success": self.success,
            "error_message": self.error_message,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else utc_now(),
            user_id=data.get("user_id", ""),
            tenant_id=data.get("tenant_id", ""),
            session_id=data.get("session_id"),
            action=AuditAction(data.get("action", "read")),
            severity=AuditSeverity(data.get("severity", "info")),
            resource_id=data.get("resource_id"),
            resource_type=data.get("resource_type"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
        )

    def as_log_line(self) -> str:
        """Format as a log line."""
        status = "OK" if self.success else "FAILED"
        resource = f"{self.resource_type}:{self.resource_id}" if self.resource_id else "-"
        return (
            f"[{self.timestamp.isoformat()}] "
            f"{self.severity.value.upper():8} "
            f"{self.action.value:15} "
            f"user={self.user_id} "
            f"resource={resource} "
            f"status={status}"
        )


class AuditLogger:
    """Audit logging service.

    Records all security-relevant operations for compliance
    and forensic analysis.

    Example:
        >>> logger = AuditLogger(backend)
        >>>
        >>> # Log an operation
        >>> await logger.log(
        ...     user_ctx,
        ...     AuditAction.CREATE,
        ...     resource_id="belief-123",
        ...     resource_type="belief",
        ... )
        >>>
        >>> # Query audit log
        >>> entries = await logger.query(
        ...     user_ctx,
        ...     actions=[AuditAction.DELETE, AuditAction.FORGET],
        ...     hours=24,
        ... )
    """

    def __init__(
        self,
        backend: "SiliconDBBackend | None" = None,
        retention_days: int = 90,
        log_reads: bool = False,
    ) -> None:
        self._backend = backend
        self._retention_days = retention_days
        self._log_reads = log_reads
        self._entries: list[AuditEntry] = []  # In-memory buffer

    async def log(
        self,
        user_ctx: UserContext,
        action: AuditAction,
        resource_id: str | None = None,
        resource_type: str | None = None,
        success: bool = True,
        error_message: str | None = None,
        details: dict[str, Any] | None = None,
        severity: AuditSeverity | None = None,
    ) -> AuditEntry:
        """Log an audit event.

        Args:
            user_ctx: User context
            action: The action being logged
            resource_id: Optional resource ID
            resource_type: Optional resource type
            success: Whether the operation succeeded
            error_message: Optional error message
            details: Additional details
            severity: Override default severity

        Returns:
            The created AuditEntry
        """
        # Skip read operations if not configured
        if action == AuditAction.READ and not self._log_reads:
            entry = AuditEntry(
                user_id=user_ctx.user_id,
                tenant_id=user_ctx.tenant_id,
                session_id=user_ctx.session_id,
                action=action,
                resource_id=resource_id,
                resource_type=resource_type,
                success=success,
            )
            return entry

        # Determine severity
        if severity is None:
            severity = self._get_default_severity(action, success)

        entry = AuditEntry(
            user_id=user_ctx.user_id,
            tenant_id=user_ctx.tenant_id,
            session_id=user_ctx.session_id,
            action=action,
            severity=severity,
            resource_id=resource_id,
            resource_type=resource_type,
            success=success,
            error_message=error_message,
            details=details or {},
        )

        # Store in memory
        self._entries.append(entry)

        # Store in backend if available
        if self._backend:
            await self._persist_entry(entry)

        return entry

    async def log_access_granted(
        self,
        user_ctx: UserContext,
        resource_id: str,
        resource_type: str,
        permission: str,
    ) -> AuditEntry:
        """Log an access granted event."""
        return await self.log(
            user_ctx,
            AuditAction.ACCESS_GRANTED,
            resource_id=resource_id,
            resource_type=resource_type,
            details={"permission": permission},
        )

    async def log_access_denied(
        self,
        user_ctx: UserContext,
        resource_id: str,
        resource_type: str,
        permission: str,
        reason: str,
    ) -> AuditEntry:
        """Log an access denied event."""
        return await self.log(
            user_ctx,
            AuditAction.ACCESS_DENIED,
            resource_id=resource_id,
            resource_type=resource_type,
            success=False,
            error_message=reason,
            details={"permission": permission},
            severity=AuditSeverity.WARNING,
        )

    async def log_forget(
        self,
        user_ctx: UserContext,
        scope: str,
        deleted_count: int,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a forget operation."""
        return await self.log(
            user_ctx,
            AuditAction.FORGET,
            details={
                "scope": scope,
                "deleted_count": deleted_count,
                **(details or {}),
            },
            severity=AuditSeverity.WARNING,
        )

    async def log_export(
        self,
        user_ctx: UserContext,
        record_count: int,
        format: str = "json",
    ) -> AuditEntry:
        """Log a data export operation."""
        return await self.log(
            user_ctx,
            AuditAction.EXPORT,
            details={
                "record_count": record_count,
                "format": format,
            },
        )

    async def log_import(
        self,
        user_ctx: UserContext,
        imported_count: int,
        failed_count: int,
    ) -> AuditEntry:
        """Log a data import operation."""
        return await self.log(
            user_ctx,
            AuditAction.IMPORT,
            success=failed_count == 0,
            details={
                "imported_count": imported_count,
                "failed_count": failed_count,
            },
        )

    async def query(
        self,
        user_ctx: UserContext,
        actions: list[AuditAction] | None = None,
        severity_min: AuditSeverity | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        hours: int | None = None,
        success_only: bool | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query the audit log.

        Args:
            user_ctx: User context
            actions: Filter by actions
            severity_min: Minimum severity
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            hours: Filter by time (last N hours)
            success_only: Filter by success status
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        results = []
        cutoff = None
        if hours:
            cutoff = utc_now() - timedelta(hours=hours)

        severity_order = {
            AuditSeverity.DEBUG: 0,
            AuditSeverity.INFO: 1,
            AuditSeverity.WARNING: 2,
            AuditSeverity.ERROR: 3,
            AuditSeverity.CRITICAL: 4,
        }
        min_severity_level = severity_order.get(severity_min, 0) if severity_min else 0

        # Search in-memory entries
        for entry in reversed(self._entries):
            # Filter by tenant (unless admin)
            if entry.tenant_id != user_ctx.tenant_id and not user_ctx.is_admin():
                continue

            # Filter by actions
            if actions and entry.action not in actions:
                continue

            # Filter by severity
            if severity_order.get(entry.severity, 0) < min_severity_level:
                continue

            # Filter by resource
            if resource_type and entry.resource_type != resource_type:
                continue
            if resource_id and entry.resource_id != resource_id:
                continue

            # Filter by time
            if cutoff and entry.timestamp < cutoff:
                continue

            # Filter by success
            if success_only is not None and entry.success != success_only:
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results

    async def get_entry(self, entry_id: str) -> AuditEntry | None:
        """Get a specific audit entry by ID."""
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    async def cleanup_old_entries(self) -> int:
        """Clean up entries older than retention period.

        Returns:
            Number of entries deleted
        """
        cutoff = utc_now() - timedelta(days=self._retention_days)
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        return original_count - len(self._entries)

    async def get_statistics(
        self,
        user_ctx: UserContext,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get audit statistics.

        Args:
            user_ctx: User context
            hours: Time window

        Returns:
            Statistics dictionary
        """
        cutoff = utc_now() - timedelta(hours=hours)

        stats = {
            "period_hours": hours,
            "total_events": 0,
            "by_action": {},
            "by_severity": {},
            "success_count": 0,
            "failure_count": 0,
            "unique_users": set(),
        }

        for entry in self._entries:
            if entry.timestamp < cutoff:
                continue
            if entry.tenant_id != user_ctx.tenant_id and not user_ctx.is_admin():
                continue

            stats["total_events"] += 1

            action_name = entry.action.value
            stats["by_action"][action_name] = stats["by_action"].get(action_name, 0) + 1

            severity_name = entry.severity.value
            stats["by_severity"][severity_name] = stats["by_severity"].get(severity_name, 0) + 1

            if entry.success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1

            stats["unique_users"].add(entry.user_id)

        stats["unique_users"] = len(stats["unique_users"])
        return stats

    def _get_default_severity(self, action: AuditAction, success: bool) -> AuditSeverity:
        """Get default severity for an action."""
        if not success:
            return AuditSeverity.WARNING

        critical_actions = {
            AuditAction.FORGET_ALL,
            AuditAction.POLICY_CHANGE,
            AuditAction.CONFIG_CHANGE,
        }
        if action in critical_actions:
            return AuditSeverity.WARNING

        warning_actions = {
            AuditAction.DELETE,
            AuditAction.FORGET,
            AuditAction.FORGET_SESSION,
            AuditAction.EXPORT,
            AuditAction.ACCESS_DENIED,
        }
        if action in warning_actions:
            return AuditSeverity.WARNING

        debug_actions = {AuditAction.READ, AuditAction.QUERY, AuditAction.SEARCH}
        if action in debug_actions:
            return AuditSeverity.DEBUG

        return AuditSeverity.INFO

    async def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist an audit entry to the backend."""
        if not self._backend:
            return

        try:
            external_id = f"audit/{entry.tenant_id}/{entry.id}"
            self._backend._db.ingest(
                external_id=external_id,
                text=entry.as_log_line(),
                metadata=entry.to_dict(),
                node_type="audit",
            )
        except Exception:
            pass  # Don't fail operations due to audit failures
