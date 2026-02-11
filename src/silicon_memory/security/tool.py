"""Privacy tool for LLM function calling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import (
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)
from silicon_memory.security.preferences import MemoryPreferences

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class PrivacyAction(str, Enum):
    """Actions available through the privacy tool."""

    # Privacy controls
    SET_PRIVACY = "set_privacy"
    GET_PRIVACY = "get_privacy"
    SHARE_WITH = "share_with"
    UNSHARE = "unshare"

    # Preferences
    GET_PREFERENCES = "get_preferences"
    UPDATE_PREFERENCES = "update_preferences"
    ADD_DO_NOT_REMEMBER = "add_do_not_remember"
    REMOVE_DO_NOT_REMEMBER = "remove_do_not_remember"

    # Consent
    GRANT_CONSENT = "grant_consent"
    REVOKE_CONSENT = "revoke_consent"
    CHECK_CONSENT = "check_consent"

    # Forgetting
    FORGET_ENTITY = "forget_entity"
    FORGET_SESSION = "forget_session"
    FORGET_TOPIC = "forget_topic"

    # Transparency
    WHY_DO_YOU_KNOW = "why_do_you_know"
    GET_ACCESS_LOG = "get_access_log"

    # Inspection
    INSPECT_MEMORIES = "inspect_memories"
    EXPORT_MEMORIES = "export_memories"


@dataclass
class PrivacyToolResponse:
    """Response from privacy tool invocation."""

    success: bool
    action: PrivacyAction
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        result = {
            "success": self.success,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


class PrivacyTool:
    """LLM-callable privacy and security tool.

    Provides a function-calling interface for LLMs to:
    - Manage privacy settings for memories
    - Update user preferences (what to remember/not)
    - Handle consent management
    - Execute forgetting operations
    - Query provenance ("why do you know")
    - Inspect and export memories

    Example:
        >>> tool = PrivacyTool(memory)
        >>>
        >>> # Set privacy level
        >>> response = await tool.invoke(
        ...     "set_privacy",
        ...     entity_id="belief-123",
        ...     privacy_level="workspace",
        ... )
        >>>
        >>> # Add do-not-remember topic
        >>> response = await tool.invoke(
        ...     "add_do_not_remember",
        ...     topic="medical",
        ... )
        >>>
        >>> # Why do you know about X?
        >>> response = await tool.invoke(
        ...     "why_do_you_know",
        ...     query="Python programming",
        ... )
    """

    def __init__(self, memory: "SiliconMemory") -> None:
        self._memory = memory

    async def invoke(
        self,
        action: str,
        **kwargs: Any,
    ) -> PrivacyToolResponse:
        """Invoke the privacy tool with an action.

        Args:
            action: The action to perform
            **kwargs: Action-specific parameters

        Returns:
            PrivacyToolResponse with results or error
        """
        try:
            action_enum = PrivacyAction(action.lower())
        except ValueError:
            return PrivacyToolResponse(
                success=False,
                action=PrivacyAction.GET_PRIVACY,
                error=f"Unknown action: {action}. Valid actions: {[a.value for a in PrivacyAction]}",
            )

        handlers = {
            PrivacyAction.SET_PRIVACY: self._handle_set_privacy,
            PrivacyAction.GET_PRIVACY: self._handle_get_privacy,
            PrivacyAction.SHARE_WITH: self._handle_share_with,
            PrivacyAction.UNSHARE: self._handle_unshare,
            PrivacyAction.GET_PREFERENCES: self._handle_get_preferences,
            PrivacyAction.UPDATE_PREFERENCES: self._handle_update_preferences,
            PrivacyAction.ADD_DO_NOT_REMEMBER: self._handle_add_do_not_remember,
            PrivacyAction.REMOVE_DO_NOT_REMEMBER: self._handle_remove_do_not_remember,
            PrivacyAction.GRANT_CONSENT: self._handle_grant_consent,
            PrivacyAction.REVOKE_CONSENT: self._handle_revoke_consent,
            PrivacyAction.CHECK_CONSENT: self._handle_check_consent,
            PrivacyAction.FORGET_ENTITY: self._handle_forget_entity,
            PrivacyAction.FORGET_SESSION: self._handle_forget_session,
            PrivacyAction.FORGET_TOPIC: self._handle_forget_topic,
            PrivacyAction.WHY_DO_YOU_KNOW: self._handle_why_do_you_know,
            PrivacyAction.GET_ACCESS_LOG: self._handle_get_access_log,
            PrivacyAction.INSPECT_MEMORIES: self._handle_inspect_memories,
            PrivacyAction.EXPORT_MEMORIES: self._handle_export_memories,
        }

        handler = handlers[action_enum]
        try:
            return await handler(**kwargs)
        except Exception as e:
            return PrivacyToolResponse(
                success=False,
                action=action_enum,
                error=str(e),
            )

    # ========== Privacy Handlers ==========

    async def _handle_set_privacy(
        self,
        entity_id: str,
        privacy_level: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Set privacy level for an entity."""
        try:
            level = PrivacyLevel(privacy_level.lower())
        except ValueError:
            return PrivacyToolResponse(
                success=False,
                action=PrivacyAction.SET_PRIVACY,
                error=f"Invalid privacy level: {privacy_level}",
            )

        # Update privacy via memory
        success = await self._memory.set_entity_privacy(entity_id, level)

        return PrivacyToolResponse(
            success=success,
            action=PrivacyAction.SET_PRIVACY,
            data={"entity_id": entity_id, "privacy_level": level.value},
        )

    async def _handle_get_privacy(
        self,
        entity_id: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Get privacy metadata for an entity."""
        privacy = await self._memory.get_entity_privacy(entity_id)

        if privacy:
            return PrivacyToolResponse(
                success=True,
                action=PrivacyAction.GET_PRIVACY,
                data=privacy.to_dict(),
            )
        else:
            return PrivacyToolResponse(
                success=False,
                action=PrivacyAction.GET_PRIVACY,
                error="Entity not found or access denied",
            )

    async def _handle_share_with(
        self,
        entity_id: str,
        user_id: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Share an entity with another user."""
        success = await self._memory.share_entity(entity_id, user_id)

        return PrivacyToolResponse(
            success=success,
            action=PrivacyAction.SHARE_WITH,
            data={"entity_id": entity_id, "shared_with": user_id},
        )

    async def _handle_unshare(
        self,
        entity_id: str,
        user_id: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Remove sharing from an entity."""
        success = await self._memory.unshare_entity(entity_id, user_id)

        return PrivacyToolResponse(
            success=success,
            action=PrivacyAction.UNSHARE,
            data={"entity_id": entity_id, "unshared_from": user_id},
        )

    # ========== Preference Handlers ==========

    async def _handle_get_preferences(self, **_: Any) -> PrivacyToolResponse:
        """Get user preferences."""
        prefs = await self._memory.get_preferences()

        return PrivacyToolResponse(
            success=True,
            action=PrivacyAction.GET_PREFERENCES,
            data=prefs.to_dict() if prefs else {},
        )

    async def _handle_update_preferences(
        self,
        default_privacy: str | None = None,
        default_retention_days: int | None = None,
        allow_sharing: bool | None = None,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Update user preferences."""
        updates = {}
        if default_privacy:
            try:
                updates["default_privacy"] = PrivacyLevel(default_privacy.lower())
            except ValueError:
                pass
        if default_retention_days is not None:
            updates["default_retention_days"] = default_retention_days
        if allow_sharing is not None:
            updates["allow_sharing"] = allow_sharing

        success = await self._memory.update_preferences_partial(updates)

        return PrivacyToolResponse(
            success=success,
            action=PrivacyAction.UPDATE_PREFERENCES,
            data={"updated": list(updates.keys())},
        )

    async def _handle_add_do_not_remember(
        self,
        topic: str | None = None,
        pattern: str | None = None,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Add a do-not-remember topic or pattern."""
        if topic:
            success = await self._memory.add_blocked_topic(topic)
            return PrivacyToolResponse(
                success=success,
                action=PrivacyAction.ADD_DO_NOT_REMEMBER,
                data={"type": "topic", "value": topic},
            )
        elif pattern:
            success = await self._memory.add_blocked_pattern(pattern)
            return PrivacyToolResponse(
                success=success,
                action=PrivacyAction.ADD_DO_NOT_REMEMBER,
                data={"type": "pattern", "value": pattern},
            )
        else:
            return PrivacyToolResponse(
                success=False,
                action=PrivacyAction.ADD_DO_NOT_REMEMBER,
                error="Either topic or pattern is required",
            )

    async def _handle_remove_do_not_remember(
        self,
        topic: str | None = None,
        pattern: str | None = None,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Remove a do-not-remember topic or pattern."""
        if topic:
            success = await self._memory.remove_blocked_topic(topic)
            return PrivacyToolResponse(
                success=success,
                action=PrivacyAction.REMOVE_DO_NOT_REMEMBER,
                data={"type": "topic", "value": topic},
            )
        elif pattern:
            success = await self._memory.remove_blocked_pattern(pattern)
            return PrivacyToolResponse(
                success=success,
                action=PrivacyAction.REMOVE_DO_NOT_REMEMBER,
                data={"type": "pattern", "value": pattern},
            )
        else:
            return PrivacyToolResponse(
                success=False,
                action=PrivacyAction.REMOVE_DO_NOT_REMEMBER,
                error="Either topic or pattern is required",
            )

    # ========== Consent Handlers ==========

    async def _handle_grant_consent(
        self,
        entity_id: str,
        consent_type: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Grant consent for an entity."""
        success = await self._memory.grant_consent(entity_id, consent_type)

        return PrivacyToolResponse(
            success=success,
            action=PrivacyAction.GRANT_CONSENT,
            data={"entity_id": entity_id, "consent_type": consent_type},
        )

    async def _handle_revoke_consent(
        self,
        entity_id: str,
        consent_type: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Revoke consent for an entity."""
        success = await self._memory.revoke_consent(entity_id, consent_type)

        return PrivacyToolResponse(
            success=success,
            action=PrivacyAction.REVOKE_CONSENT,
            data={"entity_id": entity_id, "consent_type": consent_type},
        )

    async def _handle_check_consent(
        self,
        entity_id: str,
        consent_type: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Check if consent is granted."""
        has_consent = await self._memory.check_consent(entity_id, consent_type)

        return PrivacyToolResponse(
            success=True,
            action=PrivacyAction.CHECK_CONSENT,
            data={
                "entity_id": entity_id,
                "consent_type": consent_type,
                "granted": has_consent,
            },
        )

    # ========== Forget Handlers ==========

    async def _handle_forget_entity(
        self,
        entity_id: str,
        entity_type: str | None = None,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Forget a specific entity."""
        result = await self._memory.forget_entity(entity_id, entity_type)

        return PrivacyToolResponse(
            success=result.success,
            action=PrivacyAction.FORGET_ENTITY,
            data=result.to_dict(),
            error=result.error_message,
        )

    async def _handle_forget_session(
        self,
        session_id: str,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Forget all data from a session."""
        result = await self._memory.forget_session(session_id)

        return PrivacyToolResponse(
            success=result.success,
            action=PrivacyAction.FORGET_SESSION,
            data=result.to_dict(),
            error=result.error_message,
        )

    async def _handle_forget_topic(
        self,
        topics: list[str],
        **_: Any,
    ) -> PrivacyToolResponse:
        """Forget all data with specified topics."""
        result = await self._memory.forget_topics(topics)

        return PrivacyToolResponse(
            success=result.success,
            action=PrivacyAction.FORGET_TOPIC,
            data=result.to_dict(),
            error=result.error_message,
        )

    # ========== Transparency Handlers ==========

    async def _handle_why_do_you_know(
        self,
        query: str,
        limit: int = 5,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Get provenance for a query."""
        chains = await self._memory.why_do_you_know(query, limit)

        return PrivacyToolResponse(
            success=True,
            action=PrivacyAction.WHY_DO_YOU_KNOW,
            data={
                "query": query,
                "chains": [c.to_dict() for c in chains],
                "narratives": [c.as_narrative() for c in chains],
            },
        )

    async def _handle_get_access_log(
        self,
        entity_id: str | None = None,
        limit: int = 20,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Get access log."""
        log = await self._memory.get_access_log(entity_id, limit)

        return PrivacyToolResponse(
            success=True,
            action=PrivacyAction.GET_ACCESS_LOG,
            data={
                "entity_id": entity_id,
                "entries": [e.to_dict() for e in log],
            },
        )

    # ========== Inspection Handlers ==========

    async def _handle_inspect_memories(self, **_: Any) -> PrivacyToolResponse:
        """Inspect all memories."""
        inspection = await self._memory.inspect_memories()

        return PrivacyToolResponse(
            success=True,
            action=PrivacyAction.INSPECT_MEMORIES,
            data=inspection.to_dict(),
        )

    async def _handle_export_memories(
        self,
        entity_types: list[str] | None = None,
        **_: Any,
    ) -> PrivacyToolResponse:
        """Export memories (returns count, actual export happens through iterator)."""
        inspection = await self._memory.inspect_memories()

        return PrivacyToolResponse(
            success=True,
            action=PrivacyAction.EXPORT_MEMORIES,
            data={
                "total_items": inspection.total_items,
                "note": "Use memory.export_memories() to get the actual data stream",
            },
        )

    @staticmethod
    def get_openai_schema() -> dict[str, Any]:
        """Get OpenAI function calling schema for this tool."""
        return {
            "name": "privacy",
            "description": (
                "Manage privacy, preferences, and data control. Use this to "
                "set privacy levels, manage what to remember, handle consent, "
                "forget data, and understand why certain information is known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [a.value for a in PrivacyAction],
                        "description": (
                            "The privacy action to perform: "
                            "set_privacy/get_privacy (manage entity privacy), "
                            "share_with/unshare (manage sharing), "
                            "get_preferences/update_preferences (user settings), "
                            "add_do_not_remember/remove_do_not_remember (blocked topics), "
                            "grant_consent/revoke_consent/check_consent (consent management), "
                            "forget_entity/forget_session/forget_topic (data deletion), "
                            "why_do_you_know/get_access_log (transparency), "
                            "inspect_memories/export_memories (data control)"
                        ),
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID for privacy/consent/forget operations",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Entity type (belief, experience, procedure)",
                    },
                    "privacy_level": {
                        "type": "string",
                        "enum": ["private", "workspace", "public"],
                        "description": "Privacy level for set_privacy",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID for share_with/unshare",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic for add/remove_do_not_remember",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern for add/remove_do_not_remember",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of topics for forget_topic",
                    },
                    "consent_type": {
                        "type": "string",
                        "description": "Type of consent (storage, processing, sharing)",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for forget_session",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query for why_do_you_know",
                    },
                    "default_privacy": {
                        "type": "string",
                        "description": "Default privacy level for update_preferences",
                    },
                    "default_retention_days": {
                        "type": "integer",
                        "description": "Default retention days for update_preferences",
                    },
                    "allow_sharing": {
                        "type": "boolean",
                        "description": "Allow sharing flag for update_preferences",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Limit for query results",
                    },
                },
                "required": ["action"],
            },
        }
