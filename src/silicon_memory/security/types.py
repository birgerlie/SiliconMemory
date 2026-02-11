"""Security and privacy types for Silicon Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from silicon_memory.core.utils import utc_now
from typing import Any
from uuid import uuid4


class PrivacyLevel(Enum):
    """Privacy level for memory items."""

    PRIVATE = "private"  # User only
    WORKSPACE = "workspace"  # Shared within tenant
    PUBLIC = "public"  # All users


class DataClassification(Enum):
    """Data classification for sensitivity handling."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PERSONAL = "personal"  # PII
    SENSITIVE = "sensitive"  # PII + health/financial


@dataclass
class UserContext:
    """User context required for all memory operations.

    All memory operations require a UserContext to enforce
    multi-tenant isolation and access control.

    Example:
        >>> user_ctx = UserContext(
        ...     user_id="user-123",
        ...     tenant_id="acme-corp",
        ...     default_privacy=PrivacyLevel.PRIVATE,
        ... )
        >>> memory = SiliconMemory("/path/to/db", user_context=user_ctx)
    """

    user_id: str
    tenant_id: str
    session_id: str = field(default_factory=lambda: f"session-{uuid4().hex[:12]}")
    roles: set[str] = field(default_factory=lambda: {"member"})
    default_privacy: PrivacyLevel = PrivacyLevel.PRIVATE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.tenant_id:
            raise ValueError("tenant_id is required")

    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return "admin" in self.roles

    def is_workspace_admin(self) -> bool:
        """Check if user has workspace-admin role."""
        return "workspace-admin" in self.roles or self.is_admin()

    def can_access_workspace(self) -> bool:
        """Check if user can access workspace-level resources."""
        return "member" in self.roles or "viewer" in self.roles or self.is_admin()


@dataclass
class PrivacyMetadata:
    """Privacy metadata attached to all memory items.

    Tracks ownership, privacy level, consent, and retention.
    """

    owner_id: str
    tenant_id: str
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    classification: DataClassification = DataClassification.INTERNAL
    consents: dict[str, datetime] = field(default_factory=dict)
    shared_with: list[str] = field(default_factory=list)
    do_not_remember: bool = False
    retention_until: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    created_by: str = ""
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.created_by:
            self.created_by = self.owner_id

    def is_accessible_by(self, user_ctx: UserContext) -> bool:
        """Check if accessible by given user context.

        Access rules:
        1. Owners always have access
        2. Admins have access within their tenant
        3. Workspace members can access workspace/public in same tenant
        4. Anyone can access public resources
        """
        # Owner always has access
        if user_ctx.user_id == self.owner_id:
            return True

        # Different tenant - only public is accessible
        if user_ctx.tenant_id != self.tenant_id:
            return self.privacy_level == PrivacyLevel.PUBLIC

        # Same tenant
        if user_ctx.is_admin():
            return True

        if self.privacy_level == PrivacyLevel.PUBLIC:
            return True

        if self.privacy_level == PrivacyLevel.WORKSPACE:
            return user_ctx.can_access_workspace()

        # Private - only accessible if explicitly shared
        return user_ctx.user_id in self.shared_with

    def grant_consent(self, consent_type: str) -> None:
        """Grant a consent."""
        self.consents[consent_type] = utc_now()

    def revoke_consent(self, consent_type: str) -> bool:
        """Revoke a consent."""
        if consent_type in self.consents:
            del self.consents[consent_type]
            return True
        return False

    def has_consent(self, consent_type: str) -> bool:
        """Check if consent is granted."""
        return consent_type in self.consents

    def share_with(self, user_id: str) -> None:
        """Explicitly share with a user."""
        if user_id not in self.shared_with:
            self.shared_with.append(user_id)

    def unshare_with(self, user_id: str) -> bool:
        """Remove explicit sharing with a user."""
        if user_id in self.shared_with:
            self.shared_with.remove(user_id)
            return True
        return False

    def is_expired(self) -> bool:
        """Check if retention period has expired."""
        if self.retention_until is None:
            return False
        return utc_now() > self.retention_until

    def record_access(self) -> None:
        """Record an access to this item."""
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "privacy_level": self.privacy_level.value,
            "classification": self.classification.value,
            "consents": {k: v.isoformat() for k, v in self.consents.items()},
            "shared_with": self.shared_with,
            "do_not_remember": self.do_not_remember,
            "retention_until": self.retention_until.isoformat() if self.retention_until else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrivacyMetadata":
        """Create from dictionary."""
        consents = {}
        for k, v in data.get("consents", {}).items():
            if isinstance(v, str):
                consents[k] = datetime.fromisoformat(v)
            else:
                consents[k] = v

        retention_until = None
        if data.get("retention_until"):
            retention_until = datetime.fromisoformat(data["retention_until"])

        created_at = utc_now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            owner_id=data["owner_id"],
            tenant_id=data["tenant_id"],
            privacy_level=PrivacyLevel(data.get("privacy_level", "private")),
            classification=DataClassification(data.get("classification", "internal")),
            consents=consents,
            shared_with=data.get("shared_with", []),
            do_not_remember=data.get("do_not_remember", False),
            retention_until=retention_until,
            created_at=created_at,
            created_by=data.get("created_by", data["owner_id"]),
            access_count=data.get("access_count", 0),
        )

    @classmethod
    def create_for_user(
        cls,
        user_ctx: UserContext,
        privacy_level: PrivacyLevel | None = None,
        classification: DataClassification = DataClassification.INTERNAL,
        retention_days: int | None = None,
    ) -> "PrivacyMetadata":
        """Create privacy metadata for a user context."""
        retention_until = None
        if retention_days:
            retention_until = utc_now() + timedelta(days=retention_days)

        return cls(
            owner_id=user_ctx.user_id,
            tenant_id=user_ctx.tenant_id,
            privacy_level=privacy_level or user_ctx.default_privacy,
            classification=classification,
            created_by=user_ctx.user_id,
            retention_until=retention_until,
        )
