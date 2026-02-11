"""Attribute-Based Access Control (ABAC) authorization for Silicon Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import (
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)


class Permission(Enum):
    """Permissions for memory operations."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    EXPORT = "export"
    ADMIN = "admin"


class AccessDecisionResult(Enum):
    """Result of an access decision."""

    ALLOW = "allow"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class AccessDecision:
    """Result of an authorization decision.

    Contains the decision, reason, and any relevant metadata.
    """

    result: AccessDecisionResult
    permission: Permission
    reason: str
    policy_id: str | None = None
    evaluated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        """Check if access is allowed."""
        return self.result == AccessDecisionResult.ALLOW

    @property
    def denied(self) -> bool:
        """Check if access is explicitly denied."""
        return self.result == AccessDecisionResult.DENY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "permission": self.permission.value,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Policy:
    """An access control policy.

    Policies are evaluated in order of priority (higher first).
    First matching ALLOW or DENY wins.
    """

    id: str
    name: str
    description: str
    priority: int = 0  # Higher = evaluated first
    enabled: bool = True

    # Conditions for when policy applies
    target_permissions: set[Permission] = field(default_factory=set)
    target_roles: set[str] = field(default_factory=set)
    target_privacy_levels: set[PrivacyLevel] = field(default_factory=set)
    target_classifications: set[DataClassification] = field(default_factory=set)

    # Custom condition function
    condition: Callable[[UserContext, PrivacyMetadata, Permission], bool] | None = None

    # Effect when policy matches
    effect: AccessDecisionResult = AccessDecisionResult.ALLOW

    def matches(
        self,
        user_ctx: UserContext,
        resource: PrivacyMetadata | None,
        permission: Permission,
    ) -> bool:
        """Check if this policy applies to the request."""
        # Check permission
        if self.target_permissions and permission not in self.target_permissions:
            return False

        # Check roles
        if self.target_roles and not (user_ctx.roles & self.target_roles):
            return False

        # Check privacy level
        if resource and self.target_privacy_levels:
            if resource.privacy_level not in self.target_privacy_levels:
                return False

        # Check classification
        if resource and self.target_classifications:
            if resource.classification not in self.target_classifications:
                return False

        # Check custom condition
        if self.condition and resource:
            if not self.condition(user_ctx, resource, permission):
                return False

        return True

    def evaluate(
        self,
        user_ctx: UserContext,
        resource: PrivacyMetadata | None,
        permission: Permission,
    ) -> AccessDecision | None:
        """Evaluate this policy.

        Returns:
            AccessDecision if policy matches, None otherwise
        """
        if not self.enabled:
            return None

        if not self.matches(user_ctx, resource, permission):
            return None

        return AccessDecision(
            result=self.effect,
            permission=permission,
            reason=f"Policy '{self.name}' matched",
            policy_id=self.id,
        )


class PolicyEngine:
    """ABAC policy engine for access control.

    Evaluates policies in priority order. First explicit ALLOW or DENY wins.
    If no policy matches, default decision is applied.

    Default Policies:
    1. Owners have full access to their resources
    2. Workspace members can read workspace/public resources in same tenant
    3. Anyone can read public resources
    4. Admins have full access within their tenant

    Example:
        >>> engine = PolicyEngine()
        >>> decision = engine.evaluate(user_ctx, resource_privacy, Permission.READ)
        >>> if decision.allowed:
        ...     # proceed with operation
    """

    def __init__(self, default_deny: bool = True) -> None:
        self._policies: list[Policy] = []
        self._default_deny = default_deny
        self._setup_default_policies()

    def _setup_default_policies(self) -> None:
        """Set up default access control policies."""
        # Policy 1: Owner has full access
        self.add_policy(Policy(
            id="owner-full-access",
            name="Owner Full Access",
            description="Owners have full access to their own resources",
            priority=100,
            condition=lambda ctx, res, _: ctx.user_id == res.owner_id,
            effect=AccessDecisionResult.ALLOW,
        ))

        # Policy 2: Admin has full access within tenant
        self.add_policy(Policy(
            id="admin-tenant-access",
            name="Admin Tenant Access",
            description="Admins have full access within their tenant",
            priority=90,
            target_roles={"admin"},
            condition=lambda ctx, res, _: ctx.tenant_id == res.tenant_id,
            effect=AccessDecisionResult.ALLOW,
        ))

        # Policy 3: Workspace admin has access to workspace resources
        self.add_policy(Policy(
            id="workspace-admin-access",
            name="Workspace Admin Access",
            description="Workspace admins can manage workspace resources",
            priority=80,
            target_roles={"workspace-admin"},
            target_privacy_levels={PrivacyLevel.WORKSPACE, PrivacyLevel.PUBLIC},
            condition=lambda ctx, res, _: ctx.tenant_id == res.tenant_id,
            effect=AccessDecisionResult.ALLOW,
        ))

        # Policy 4: Members can read workspace/public in same tenant
        self.add_policy(Policy(
            id="member-workspace-read",
            name="Member Workspace Read",
            description="Members can read workspace and public resources in their tenant",
            priority=70,
            target_permissions={Permission.READ},
            target_roles={"member", "viewer"},
            target_privacy_levels={PrivacyLevel.WORKSPACE, PrivacyLevel.PUBLIC},
            condition=lambda ctx, res, _: ctx.tenant_id == res.tenant_id,
            effect=AccessDecisionResult.ALLOW,
        ))

        # Policy 5: Anyone can read public resources
        self.add_policy(Policy(
            id="public-read",
            name="Public Read",
            description="Anyone can read public resources",
            priority=60,
            target_permissions={Permission.READ},
            target_privacy_levels={PrivacyLevel.PUBLIC},
            effect=AccessDecisionResult.ALLOW,
        ))

        # Policy 6: Explicit sharing
        self.add_policy(Policy(
            id="explicit-share-read",
            name="Explicit Share Read",
            description="Users can read resources explicitly shared with them",
            priority=50,
            target_permissions={Permission.READ},
            condition=lambda ctx, res, _: ctx.user_id in res.shared_with,
            effect=AccessDecisionResult.ALLOW,
        ))

        # Policy 7: Deny access to sensitive data for non-admin
        self.add_policy(Policy(
            id="sensitive-data-restriction",
            name="Sensitive Data Restriction",
            description="Only admins can access sensitive data",
            priority=95,
            target_classifications={DataClassification.SENSITIVE},
            condition=lambda ctx, res, _: not ctx.is_admin() and ctx.user_id != res.owner_id,
            effect=AccessDecisionResult.DENY,
        ))

    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the engine."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy by ID."""
        for i, p in enumerate(self._policies):
            if p.id == policy_id:
                self._policies.pop(i)
                return True
        return False

    def get_policy(self, policy_id: str) -> Policy | None:
        """Get a policy by ID."""
        for p in self._policies:
            if p.id == policy_id:
                return p
        return None

    def list_policies(self) -> list[Policy]:
        """List all policies."""
        return list(self._policies)

    def evaluate(
        self,
        user_ctx: UserContext,
        resource: PrivacyMetadata | None,
        permission: Permission,
    ) -> AccessDecision:
        """Evaluate access request against all policies.

        Args:
            user_ctx: The user context making the request
            resource: The resource's privacy metadata (None for new resources)
            permission: The permission being requested

        Returns:
            AccessDecision with the result
        """
        # For new resources (no privacy metadata), allow if user has write permission
        if resource is None:
            return AccessDecision(
                result=AccessDecisionResult.ALLOW,
                permission=permission,
                reason="New resource creation allowed",
                policy_id=None,
            )

        # Evaluate policies in priority order
        for policy in self._policies:
            decision = policy.evaluate(user_ctx, resource, permission)
            if decision is not None:
                return decision

        # No policy matched - apply default
        if self._default_deny:
            return AccessDecision(
                result=AccessDecisionResult.DENY,
                permission=permission,
                reason="No policy matched, default deny",
                policy_id=None,
            )
        else:
            return AccessDecision(
                result=AccessDecisionResult.ALLOW,
                permission=permission,
                reason="No policy matched, default allow",
                policy_id=None,
            )

    def can_read(self, user_ctx: UserContext, resource: PrivacyMetadata) -> bool:
        """Check if user can read the resource."""
        return self.evaluate(user_ctx, resource, Permission.READ).allowed

    def can_write(self, user_ctx: UserContext, resource: PrivacyMetadata) -> bool:
        """Check if user can write to the resource."""
        return self.evaluate(user_ctx, resource, Permission.WRITE).allowed

    def can_delete(self, user_ctx: UserContext, resource: PrivacyMetadata) -> bool:
        """Check if user can delete the resource."""
        return self.evaluate(user_ctx, resource, Permission.DELETE).allowed

    def can_share(self, user_ctx: UserContext, resource: PrivacyMetadata) -> bool:
        """Check if user can share the resource."""
        return self.evaluate(user_ctx, resource, Permission.SHARE).allowed

    def can_export(self, user_ctx: UserContext, resource: PrivacyMetadata) -> bool:
        """Check if user can export the resource."""
        return self.evaluate(user_ctx, resource, Permission.EXPORT).allowed


def create_custom_policy(
    policy_id: str,
    name: str,
    description: str,
    allow: bool = True,
    priority: int = 50,
    permissions: set[Permission] | None = None,
    roles: set[str] | None = None,
    privacy_levels: set[PrivacyLevel] | None = None,
    classifications: set[DataClassification] | None = None,
    condition: Callable[[UserContext, PrivacyMetadata, Permission], bool] | None = None,
) -> Policy:
    """Helper to create a custom policy.

    Example:
        >>> policy = create_custom_policy(
        ...     "my-policy",
        ...     "My Custom Policy",
        ...     "Allow managers to read confidential data",
        ...     allow=True,
        ...     roles={"manager"},
        ...     classifications={DataClassification.CONFIDENTIAL},
        ... )
        >>> engine.add_policy(policy)
    """
    return Policy(
        id=policy_id,
        name=name,
        description=description,
        priority=priority,
        enabled=True,
        target_permissions=permissions or set(),
        target_roles=roles or set(),
        target_privacy_levels=privacy_levels or set(),
        target_classifications=classifications or set(),
        condition=condition,
        effect=AccessDecisionResult.ALLOW if allow else AccessDecisionResult.DENY,
    )
