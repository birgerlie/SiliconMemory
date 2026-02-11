"""Unit tests for the security module."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import (
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)
from silicon_memory.security.config import SecurityConfig
from silicon_memory.security.preferences import MemoryPreferences
from silicon_memory.security.authorization import (
    AccessDecision,
    AccessDecisionResult,
    Permission,
    Policy,
    PolicyEngine,
    create_custom_policy,
)
from silicon_memory.security.forgetting import (
    ForgetRequest,
    ForgetScope,
    ForgetStatus,
)
from silicon_memory.security.transparency import (
    ProvenanceEvent,
    ProvenanceType,
    ProvenanceChain,
    AccessLogEntry,
)
from silicon_memory.security.inspector import (
    MemoryRecord,
    MemoryInspection,
    CorrectionResult,
)
from silicon_memory.security.audit import (
    AuditAction,
    AuditEntry,
    AuditLogger,
    AuditSeverity,
)


# ============================================================================
# UserContext Tests
# ============================================================================


class TestUserContext:
    """Tests for UserContext."""

    def test_create_user_context(self):
        """Test creating a user context."""
        ctx = UserContext(
            user_id="user-123",
            tenant_id="tenant-abc",
        )
        assert ctx.user_id == "user-123"
        assert ctx.tenant_id == "tenant-abc"
        assert ctx.session_id.startswith("session-")
        assert "member" in ctx.roles
        assert ctx.default_privacy == PrivacyLevel.PRIVATE

    def test_user_context_requires_user_id(self):
        """Test that user_id is required."""
        with pytest.raises(ValueError, match="user_id is required"):
            UserContext(user_id="", tenant_id="tenant-abc")

    def test_user_context_requires_tenant_id(self):
        """Test that tenant_id is required."""
        with pytest.raises(ValueError, match="tenant_id is required"):
            UserContext(user_id="user-123", tenant_id="")

    def test_is_admin(self):
        """Test admin role check."""
        ctx = UserContext(
            user_id="admin-1",
            tenant_id="tenant-abc",
            roles={"admin"},
        )
        assert ctx.is_admin() is True

        ctx2 = UserContext(
            user_id="user-123",
            tenant_id="tenant-abc",
        )
        assert ctx2.is_admin() is False

    def test_is_workspace_admin(self):
        """Test workspace admin role check."""
        ctx = UserContext(
            user_id="ws-admin",
            tenant_id="tenant-abc",
            roles={"workspace-admin"},
        )
        assert ctx.is_workspace_admin() is True
        assert ctx.is_admin() is False

    def test_can_access_workspace(self):
        """Test workspace access check."""
        member = UserContext(
            user_id="member-1",
            tenant_id="tenant-abc",
            roles={"member"},
        )
        assert member.can_access_workspace() is True

        viewer = UserContext(
            user_id="viewer-1",
            tenant_id="tenant-abc",
            roles={"viewer"},
        )
        assert viewer.can_access_workspace() is True


# ============================================================================
# PrivacyMetadata Tests
# ============================================================================


class TestPrivacyMetadata:
    """Tests for PrivacyMetadata."""

    def test_create_privacy_metadata(self):
        """Test creating privacy metadata."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
        )
        assert meta.owner_id == "user-123"
        assert meta.tenant_id == "tenant-abc"
        assert meta.privacy_level == PrivacyLevel.PRIVATE
        assert meta.created_by == "user-123"

    def test_is_accessible_by_owner(self):
        """Test owner always has access."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")
        assert meta.is_accessible_by(ctx) is True

    def test_private_not_accessible_by_others(self):
        """Test private resources are not accessible by others."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )
        ctx = UserContext(user_id="user-456", tenant_id="tenant-abc")
        assert meta.is_accessible_by(ctx) is False

    def test_workspace_accessible_by_tenant_members(self):
        """Test workspace resources are accessible by tenant members."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.WORKSPACE,
        )
        ctx = UserContext(
            user_id="user-456",
            tenant_id="tenant-abc",
            roles={"member"},
        )
        assert meta.is_accessible_by(ctx) is True

    def test_public_accessible_by_anyone(self):
        """Test public resources are accessible by anyone."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PUBLIC,
        )
        ctx = UserContext(user_id="user-456", tenant_id="other-tenant")
        assert meta.is_accessible_by(ctx) is True

    def test_cross_tenant_private_not_accessible(self):
        """Test cross-tenant private access is denied."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )
        ctx = UserContext(user_id="user-456", tenant_id="other-tenant")
        assert meta.is_accessible_by(ctx) is False

    def test_admin_has_full_access_in_tenant(self):
        """Test admin has full access within tenant."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )
        ctx = UserContext(
            user_id="admin-1",
            tenant_id="tenant-abc",
            roles={"admin"},
        )
        assert meta.is_accessible_by(ctx) is True

    def test_explicit_sharing(self):
        """Test explicit sharing grants access."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )
        meta.share_with("user-456")

        ctx = UserContext(user_id="user-456", tenant_id="tenant-abc")
        assert meta.is_accessible_by(ctx) is True

    def test_consent_management(self):
        """Test consent granting and revoking."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
        )

        assert meta.has_consent("storage") is False
        meta.grant_consent("storage")
        assert meta.has_consent("storage") is True

        meta.revoke_consent("storage")
        assert meta.has_consent("storage") is False

    def test_retention_expiry(self):
        """Test retention period expiry check."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            retention_until=utc_now() - timedelta(days=1),
        )
        assert meta.is_expired() is True

        meta2 = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            retention_until=utc_now() + timedelta(days=1),
        )
        assert meta2.is_expired() is False

    def test_serialization(self):
        """Test to_dict and from_dict."""
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.WORKSPACE,
            classification=DataClassification.CONFIDENTIAL,
        )
        meta.grant_consent("storage")
        meta.share_with("user-456")

        data = meta.to_dict()
        restored = PrivacyMetadata.from_dict(data)

        assert restored.owner_id == meta.owner_id
        assert restored.tenant_id == meta.tenant_id
        assert restored.privacy_level == meta.privacy_level
        assert restored.classification == meta.classification
        assert restored.has_consent("storage")
        assert "user-456" in restored.shared_with

    def test_create_for_user(self):
        """Test creating privacy metadata from user context."""
        ctx = UserContext(
            user_id="user-123",
            tenant_id="tenant-abc",
            default_privacy=PrivacyLevel.WORKSPACE,
        )
        meta = PrivacyMetadata.create_for_user(ctx, retention_days=30)

        assert meta.owner_id == "user-123"
        assert meta.tenant_id == "tenant-abc"
        assert meta.privacy_level == PrivacyLevel.WORKSPACE
        assert meta.retention_until is not None


# ============================================================================
# SecurityConfig Tests
# ============================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SecurityConfig()
        assert config.enforce_tenant_isolation is True
        assert config.default_privacy_level == "private"
        assert config.audit_enabled is True

    def test_strict_config(self):
        """Test strict configuration preset."""
        config = SecurityConfig.strict()
        assert config.allow_cross_tenant_public is False
        assert config.require_storage_consent is True
        assert config.admin_bypass_enabled is False

    def test_permissive_config(self):
        """Test permissive configuration preset."""
        config = SecurityConfig.permissive()
        assert config.default_privacy_level == "workspace"
        assert config.require_storage_consent is False

    def test_validation(self):
        """Test configuration validation."""
        config = SecurityConfig(
            default_retention_days=100,
            max_retention_days=50,
        )
        issues = config.validate()
        assert len(issues) > 0
        assert "cannot exceed" in issues[0]


# ============================================================================
# MemoryPreferences Tests
# ============================================================================


class TestMemoryPreferences:
    """Tests for MemoryPreferences."""

    def test_create_preferences(self):
        """Test creating preferences."""
        prefs = MemoryPreferences(user_id="user-123")
        assert prefs.user_id == "user-123"
        assert prefs.allow_storage is True

    def test_should_remember_blocked_topic(self):
        """Test blocking by topic."""
        prefs = MemoryPreferences(
            user_id="user-123",
            do_not_remember_topics=["medical", "financial"],
        )

        assert prefs.should_remember("This is about coding") is True
        assert prefs.should_remember("My medical history") is False
        assert prefs.should_remember("Financial transactions") is False

    def test_should_remember_blocked_pattern(self):
        """Test blocking by regex pattern."""
        prefs = MemoryPreferences(
            user_id="user-123",
            do_not_remember_patterns=[r"\d{3}-\d{2}-\d{4}"],  # SSN pattern
        )

        assert prefs.should_remember("Normal text") is True
        assert prefs.should_remember("My SSN is 123-45-6789") is False

    def test_add_remove_blocked_topic(self):
        """Test adding and removing blocked topics."""
        prefs = MemoryPreferences(user_id="user-123")

        prefs.add_blocked_topic("health")
        assert "health" in prefs.do_not_remember_topics

        prefs.remove_blocked_topic("health")
        assert "health" not in prefs.do_not_remember_topics

    def test_retention_days_limit(self):
        """Test retention days respect limits."""
        prefs = MemoryPreferences(
            user_id="user-123",
            default_retention_days=90,
            max_retention_days=30,
        )

        # Request 60 days, but max is 30
        assert prefs.get_retention_days(60) == 30
        # Default is 90, but capped at 30
        assert prefs.get_retention_days() == 30

    def test_serialization(self):
        """Test to_dict and from_dict."""
        prefs = MemoryPreferences(
            user_id="user-123",
            tenant_id="tenant-abc",
            do_not_remember_topics=["health"],
            default_privacy=PrivacyLevel.WORKSPACE,
        )

        data = prefs.to_dict()
        restored = MemoryPreferences.from_dict(data)

        assert restored.user_id == prefs.user_id
        assert restored.tenant_id == prefs.tenant_id
        assert "health" in restored.do_not_remember_topics
        assert restored.default_privacy == PrivacyLevel.WORKSPACE


# ============================================================================
# Authorization Tests
# ============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_owner_has_full_access(self):
        """Test that owners have full access."""
        engine = PolicyEngine()

        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")
        meta = PrivacyMetadata(owner_id="user-123", tenant_id="tenant-abc")

        for perm in Permission:
            decision = engine.evaluate(ctx, meta, perm)
            assert decision.allowed, f"Owner should have {perm.value} access"

    def test_admin_has_tenant_access(self):
        """Test that admins have full access within tenant."""
        engine = PolicyEngine()

        ctx = UserContext(
            user_id="admin-1",
            tenant_id="tenant-abc",
            roles={"admin"},
        )
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.allowed

    def test_workspace_member_read_access(self):
        """Test workspace members can read workspace resources."""
        engine = PolicyEngine()

        ctx = UserContext(
            user_id="user-456",
            tenant_id="tenant-abc",
            roles={"member"},
        )
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.WORKSPACE,
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.allowed

    def test_public_read_access(self):
        """Test anyone can read public resources."""
        engine = PolicyEngine()

        ctx = UserContext(user_id="user-456", tenant_id="other-tenant")
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PUBLIC,
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.allowed

    def test_private_denied_to_others(self):
        """Test private resources are denied to non-owners."""
        engine = PolicyEngine()

        ctx = UserContext(user_id="user-456", tenant_id="tenant-abc")
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.denied

    def test_explicit_share_grants_read(self):
        """Test explicit sharing grants read access."""
        engine = PolicyEngine()

        ctx = UserContext(user_id="user-456", tenant_id="tenant-abc")
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
            shared_with=["user-456"],
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.allowed

    def test_sensitive_data_restriction(self):
        """Test sensitive data is restricted to admins/owners."""
        engine = PolicyEngine()

        ctx = UserContext(
            user_id="user-456",
            tenant_id="tenant-abc",
            roles={"member"},
        )
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.WORKSPACE,
            classification=DataClassification.SENSITIVE,
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.denied

    def test_custom_policy(self):
        """Test adding a custom policy."""
        engine = PolicyEngine()

        # Add policy to allow managers to read confidential data
        policy = create_custom_policy(
            "manager-confidential",
            "Manager Confidential Access",
            "Allow managers to read confidential data",
            allow=True,
            priority=85,
            roles={"manager"},
            classifications={DataClassification.CONFIDENTIAL},
        )
        engine.add_policy(policy)

        ctx = UserContext(
            user_id="manager-1",
            tenant_id="tenant-abc",
            roles={"manager"},
        )
        meta = PrivacyMetadata(
            owner_id="user-123",
            tenant_id="tenant-abc",
            privacy_level=PrivacyLevel.PRIVATE,
            classification=DataClassification.CONFIDENTIAL,
        )

        decision = engine.evaluate(ctx, meta, Permission.READ)
        assert decision.allowed


# ============================================================================
# Forgetting Tests
# ============================================================================


class TestForgetRequest:
    """Tests for ForgetRequest."""

    def test_entity_request_validation(self):
        """Test entity scope requires entity_id."""
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")
        req = ForgetRequest(user_ctx=ctx, scope=ForgetScope.ENTITY)

        issues = req.validate()
        assert "entity_id is required" in issues[0]

    def test_session_request_validation(self):
        """Test session scope requires session_id."""
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")
        req = ForgetRequest(user_ctx=ctx, scope=ForgetScope.SESSION)

        issues = req.validate()
        assert "session_id is required" in issues[0]

    def test_valid_entity_request(self):
        """Test valid entity request passes validation."""
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")
        req = ForgetRequest(
            user_ctx=ctx,
            scope=ForgetScope.ENTITY,
            entity_id="belief-123",
        )

        issues = req.validate()
        assert len(issues) == 0


# ============================================================================
# Transparency Tests
# ============================================================================


class TestProvenanceChain:
    """Tests for ProvenanceChain."""

    def test_create_provenance_chain(self):
        """Test creating a provenance chain."""
        event = ProvenanceEvent(
            event_type=ProvenanceType.CREATED,
            timestamp=utc_now(),
            actor_id="user-123",
            actor_type="user",
        )

        chain = ProvenanceChain(
            entity_id="belief-123",
            entity_type="belief",
            content_summary="Python is a programming language",
            events=[event],
        )

        assert chain.origin == event
        assert len(chain.events) == 1

    def test_provenance_narrative(self):
        """Test generating a narrative."""
        event = ProvenanceEvent(
            event_type=ProvenanceType.CREATED,
            timestamp=utc_now(),
            actor_id="user-123",
            actor_type="user",
            source_name="User Input",
        )

        chain = ProvenanceChain(
            entity_id="belief-123",
            entity_type="belief",
            content_summary="Python is great",
            events=[event],
        )

        narrative = chain.as_narrative()
        assert "Python is great" in narrative
        assert "user-123" in narrative


class TestAccessLogEntry:
    """Tests for AccessLogEntry."""

    def test_create_access_log_entry(self):
        """Test creating an access log entry."""
        entry = AccessLogEntry(
            entity_id="belief-123",
            entity_type="belief",
            accessed_by="user-456",
            accessed_at=utc_now(),
            access_type="read",
        )

        assert entry.entity_id == "belief-123"
        assert entry.accessed_by == "user-456"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        entry = AccessLogEntry(
            entity_id="belief-123",
            entity_type="belief",
            accessed_by="user-456",
            accessed_at=utc_now(),
            access_type="read",
        )

        data = entry.to_dict()
        restored = AccessLogEntry.from_dict(data)

        assert restored.entity_id == entry.entity_id
        assert restored.accessed_by == entry.accessed_by


# ============================================================================
# Inspector Tests
# ============================================================================


class TestMemoryRecord:
    """Tests for MemoryRecord."""

    def test_create_memory_record(self):
        """Test creating a memory record."""
        record = MemoryRecord(
            external_id="tenant/user/belief-123",
            entity_type="belief",
            content="Python is great",
            metadata={"confidence": 0.9},
        )

        assert record.entity_type == "belief"
        assert record.content == "Python is great"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        meta = PrivacyMetadata(owner_id="user-123", tenant_id="tenant-abc")
        record = MemoryRecord(
            external_id="tenant/user/belief-123",
            entity_type="belief",
            content="Test content",
            metadata={},
            privacy=meta,
        )

        data = record.to_dict()
        restored = MemoryRecord.from_dict(data)

        assert restored.external_id == record.external_id
        assert restored.privacy.owner_id == "user-123"


class TestMemoryInspection:
    """Tests for MemoryInspection."""

    def test_create_inspection(self):
        """Test creating an inspection."""
        inspection = MemoryInspection(
            user_id="user-123",
            tenant_id="tenant-abc",
            belief_count=10,
            experience_count=5,
            total_items=15,
        )

        assert inspection.total_items == 15
        assert inspection.belief_count == 10

    def test_inspection_summary(self):
        """Test generating a summary."""
        inspection = MemoryInspection(
            user_id="user-123",
            tenant_id="tenant-abc",
            belief_count=10,
            experience_count=5,
            private_count=12,
            total_items=15,
        )

        summary = inspection.as_summary()
        assert "user-123" in summary
        assert "Beliefs: 10" in summary


# ============================================================================
# Audit Tests
# ============================================================================


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.mark.asyncio
    async def test_log_action(self):
        """Test logging an action."""
        logger = AuditLogger()
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")

        entry = await logger.log(
            ctx,
            AuditAction.CREATE,
            resource_id="belief-123",
            resource_type="belief",
        )

        assert entry.action == AuditAction.CREATE
        assert entry.user_id == "user-123"
        assert entry.success is True

    @pytest.mark.asyncio
    async def test_query_audit_log(self):
        """Test querying the audit log."""
        logger = AuditLogger()
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")

        await logger.log(ctx, AuditAction.CREATE, resource_id="belief-1")
        await logger.log(ctx, AuditAction.DELETE, resource_id="belief-2")
        await logger.log(ctx, AuditAction.READ, resource_id="belief-3")

        # Query all
        results = await logger.query(ctx)
        assert len(results) >= 2  # Reads might be skipped

        # Query only deletes
        results = await logger.query(ctx, actions=[AuditAction.DELETE])
        assert len(results) == 1
        assert results[0].action == AuditAction.DELETE

    @pytest.mark.asyncio
    async def test_log_access_denied(self):
        """Test logging access denied events."""
        logger = AuditLogger()
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")

        entry = await logger.log_access_denied(
            ctx,
            resource_id="secret-doc",
            resource_type="document",
            permission="read",
            reason="Insufficient permissions",
        )

        assert entry.action == AuditAction.ACCESS_DENIED
        assert entry.success is False
        assert entry.severity == AuditSeverity.WARNING

    @pytest.mark.asyncio
    async def test_audit_statistics(self):
        """Test audit statistics."""
        logger = AuditLogger()
        ctx = UserContext(user_id="user-123", tenant_id="tenant-abc")

        await logger.log(ctx, AuditAction.CREATE, resource_id="belief-1")
        await logger.log(ctx, AuditAction.UPDATE, resource_id="belief-1")
        await logger.log(ctx, AuditAction.DELETE, resource_id="belief-2", success=False)

        stats = await logger.get_statistics(ctx)
        assert stats["total_events"] >= 3
        assert stats["failure_count"] >= 1


class TestAuditEntry:
    """Tests for AuditEntry."""

    def test_create_entry(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            user_id="user-123",
            tenant_id="tenant-abc",
            action=AuditAction.CREATE,
            resource_id="belief-123",
        )

        assert entry.action == AuditAction.CREATE
        assert entry.success is True

    def test_entry_log_line(self):
        """Test formatting as log line."""
        entry = AuditEntry(
            user_id="user-123",
            tenant_id="tenant-abc",
            action=AuditAction.DELETE,
            resource_id="belief-123",
            resource_type="belief",
            success=True,
        )

        line = entry.as_log_line()
        assert "delete" in line
        assert "user=user-123" in line
        assert "OK" in line

    def test_serialization(self):
        """Test to_dict and from_dict."""
        entry = AuditEntry(
            user_id="user-123",
            tenant_id="tenant-abc",
            action=AuditAction.UPDATE,
            severity=AuditSeverity.WARNING,
        )

        data = entry.to_dict()
        restored = AuditEntry.from_dict(data)

        assert restored.user_id == entry.user_id
        assert restored.action == entry.action
        assert restored.severity == entry.severity
