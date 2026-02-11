"""Memory preferences for what to remember/not remember."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import DataClassification, PrivacyLevel


@dataclass
class MemoryPreferences:
    """User preferences for memory storage and handling.

    Controls what topics/patterns should not be remembered,
    default retention periods, and other privacy preferences.

    Example:
        >>> prefs = MemoryPreferences(
        ...     user_id="user-123",
        ...     do_not_remember_topics=["medical", "financial"],
        ...     do_not_remember_patterns=[r"\\d{3}-\\d{2}-\\d{4}"],  # SSN
        ...     default_retention_days=365,
        ... )
        >>> await memory.update_preferences(prefs)
    """

    user_id: str
    tenant_id: str | None = None

    # Topics to never remember
    do_not_remember_topics: list[str] = field(default_factory=list)

    # Regex patterns to never remember (e.g., SSN, credit cards)
    do_not_remember_patterns: list[str] = field(default_factory=list)

    # Default privacy level for new memories
    default_privacy: PrivacyLevel = PrivacyLevel.PRIVATE

    # Default classification for new memories
    default_classification: DataClassification = DataClassification.INTERNAL

    # Default retention in days (None = no expiry)
    default_retention_days: int | None = None

    # Maximum retention in days (enforced limit)
    max_retention_days: int | None = None

    # Consent preferences
    allow_storage: bool = True
    allow_processing: bool = True
    allow_sharing: bool = False

    # Auto-forget settings
    auto_forget_sessions: bool = False  # Forget session data after session ends
    session_retention_hours: int = 24  # How long to keep session data

    # Transparency preferences
    track_provenance: bool = True  # Track where memories come from
    track_access: bool = True  # Log who accesses memories

    # Notification preferences
    notify_on_access: bool = False
    notify_on_share: bool = False
    notify_on_export: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    # Compiled patterns (transient, not stored)
    _compiled_patterns: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id is required")
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns = []
        for pattern in self.do_not_remember_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass  # Skip invalid patterns

    def should_remember(self, content: str, topics: list[str] | None = None) -> bool:
        """Check if content should be remembered based on preferences.

        Args:
            content: The content to check
            topics: Optional list of topics associated with the content

        Returns:
            True if the content should be remembered, False otherwise
        """
        if not self.allow_storage:
            return False

        # Check topics
        if topics:
            content_topics_lower = {t.lower() for t in topics}
            blocked_topics_lower = {t.lower() for t in self.do_not_remember_topics}
            if content_topics_lower & blocked_topics_lower:
                return False

        # Check content against blocked topics (keyword matching)
        content_lower = content.lower()
        for topic in self.do_not_remember_topics:
            if topic.lower() in content_lower:
                return False

        # Check patterns
        for pattern in self._compiled_patterns:
            if pattern.search(content):
                return False

        return True

    def get_retention_days(self, requested_days: int | None = None) -> int | None:
        """Get effective retention days, respecting limits.

        Args:
            requested_days: Requested retention period

        Returns:
            Effective retention days, or None for no expiry
        """
        if requested_days is None:
            days = self.default_retention_days
        else:
            days = requested_days

        if days is not None and self.max_retention_days is not None:
            days = min(days, self.max_retention_days)

        return days

    def add_blocked_topic(self, topic: str) -> None:
        """Add a topic to the do-not-remember list."""
        topic_lower = topic.lower()
        if topic_lower not in [t.lower() for t in self.do_not_remember_topics]:
            self.do_not_remember_topics.append(topic)
            self.updated_at = utc_now()

    def remove_blocked_topic(self, topic: str) -> bool:
        """Remove a topic from the do-not-remember list."""
        topic_lower = topic.lower()
        for i, t in enumerate(self.do_not_remember_topics):
            if t.lower() == topic_lower:
                self.do_not_remember_topics.pop(i)
                self.updated_at = utc_now()
                return True
        return False

    def add_blocked_pattern(self, pattern: str) -> bool:
        """Add a regex pattern to the do-not-remember list.

        Returns:
            True if pattern was valid and added, False otherwise
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            if pattern not in self.do_not_remember_patterns:
                self.do_not_remember_patterns.append(pattern)
                self._compiled_patterns.append(compiled)
                self.updated_at = utc_now()
            return True
        except re.error:
            return False

    def remove_blocked_pattern(self, pattern: str) -> bool:
        """Remove a pattern from the do-not-remember list."""
        if pattern in self.do_not_remember_patterns:
            idx = self.do_not_remember_patterns.index(pattern)
            self.do_not_remember_patterns.pop(idx)
            if idx < len(self._compiled_patterns):
                self._compiled_patterns.pop(idx)
            self.updated_at = utc_now()
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "do_not_remember_topics": self.do_not_remember_topics,
            "do_not_remember_patterns": self.do_not_remember_patterns,
            "default_privacy": self.default_privacy.value,
            "default_classification": self.default_classification.value,
            "default_retention_days": self.default_retention_days,
            "max_retention_days": self.max_retention_days,
            "allow_storage": self.allow_storage,
            "allow_processing": self.allow_processing,
            "allow_sharing": self.allow_sharing,
            "auto_forget_sessions": self.auto_forget_sessions,
            "session_retention_hours": self.session_retention_hours,
            "track_provenance": self.track_provenance,
            "track_access": self.track_access,
            "notify_on_access": self.notify_on_access,
            "notify_on_share": self.notify_on_share,
            "notify_on_export": self.notify_on_export,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryPreferences":
        """Create from dictionary."""
        created_at = utc_now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        updated_at = utc_now()
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])

        return cls(
            user_id=data["user_id"],
            tenant_id=data.get("tenant_id"),
            do_not_remember_topics=data.get("do_not_remember_topics", []),
            do_not_remember_patterns=data.get("do_not_remember_patterns", []),
            default_privacy=PrivacyLevel(data.get("default_privacy", "private")),
            default_classification=DataClassification(
                data.get("default_classification", "internal")
            ),
            default_retention_days=data.get("default_retention_days"),
            max_retention_days=data.get("max_retention_days"),
            allow_storage=data.get("allow_storage", True),
            allow_processing=data.get("allow_processing", True),
            allow_sharing=data.get("allow_sharing", False),
            auto_forget_sessions=data.get("auto_forget_sessions", False),
            session_retention_hours=data.get("session_retention_hours", 24),
            track_provenance=data.get("track_provenance", True),
            track_access=data.get("track_access", True),
            notify_on_access=data.get("notify_on_access", False),
            notify_on_share=data.get("notify_on_share", False),
            notify_on_export=data.get("notify_on_export", True),
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    def default_for_user(cls, user_id: str, tenant_id: str | None = None) -> "MemoryPreferences":
        """Create default preferences for a user."""
        return cls(
            user_id=user_id,
            tenant_id=tenant_id,
            default_privacy=PrivacyLevel.PRIVATE,
            allow_storage=True,
            allow_processing=True,
            allow_sharing=False,
        )

    @classmethod
    def privacy_focused(cls, user_id: str, tenant_id: str | None = None) -> "MemoryPreferences":
        """Create privacy-focused preferences.

        Suitable for users who want minimal data retention.
        """
        return cls(
            user_id=user_id,
            tenant_id=tenant_id,
            default_privacy=PrivacyLevel.PRIVATE,
            default_retention_days=30,
            max_retention_days=90,
            allow_storage=True,
            allow_processing=True,
            allow_sharing=False,
            auto_forget_sessions=True,
            session_retention_hours=1,
            track_provenance=True,
            track_access=True,
            notify_on_access=True,
            notify_on_share=True,
            notify_on_export=True,
            do_not_remember_patterns=[
                r"\d{3}-\d{2}-\d{4}",  # SSN
                r"\b\d{16}\b",  # Credit card
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ],
        )
