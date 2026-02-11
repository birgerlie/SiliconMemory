"""Security and privacy module for Silicon Memory.

This module provides multi-user security, privacy controls, and
data governance features:

- Multi-tenant isolation with UserContext
- Privacy levels (private, workspace, public)
- Data classification (public, internal, confidential, personal, sensitive)
- ABAC authorization with customizable policies
- Forgetting service (GDPR-compliant deletion)
- Transparency service ("why do you know")
- Memory inspector (export/import/correct)
- Audit logging
- LLM-callable privacy tool

Example:
    >>> from silicon_memory.security import (
    ...     UserContext,
    ...     PrivacyLevel,
    ...     PolicyEngine,
    ... )
    >>>
    >>> # Create user context (required for all operations)
    >>> user_ctx = UserContext(
    ...     user_id="user-123",
    ...     tenant_id="acme-corp",
    ...     default_privacy=PrivacyLevel.PRIVATE,
    ... )
    >>>
    >>> # Initialize memory with user context
    >>> memory = SiliconMemory("/path/to/db", user_context=user_ctx)
"""

# Core types
from silicon_memory.security.types import (
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)

# Configuration
from silicon_memory.security.config import SecurityConfig

# Preferences
from silicon_memory.security.preferences import MemoryPreferences

# Authorization
from silicon_memory.security.authorization import (
    AccessDecision,
    AccessDecisionResult,
    Permission,
    Policy,
    PolicyEngine,
    create_custom_policy,
)

# Forgetting
from silicon_memory.security.forgetting import (
    ForgetRequest,
    ForgetResult,
    ForgetScope,
    ForgetStatus,
    ForgettingService,
)

# Transparency
from silicon_memory.security.transparency import (
    AccessLogEntry,
    ProvenanceChain,
    ProvenanceEvent,
    ProvenanceType,
    TransparencyService,
)

# Inspector
from silicon_memory.security.inspector import (
    CorrectionResult,
    ExportFormat,
    MemoryInspection,
    MemoryInspector,
    MemoryRecord,
)

# Audit
from silicon_memory.security.audit import (
    AuditAction,
    AuditEntry,
    AuditLogger,
    AuditSeverity,
)

# Tool
from silicon_memory.security.tool import (
    PrivacyAction,
    PrivacyTool,
    PrivacyToolResponse,
)

__all__ = [
    # Core types
    "DataClassification",
    "PrivacyLevel",
    "PrivacyMetadata",
    "UserContext",
    # Configuration
    "SecurityConfig",
    # Preferences
    "MemoryPreferences",
    # Authorization
    "AccessDecision",
    "AccessDecisionResult",
    "Permission",
    "Policy",
    "PolicyEngine",
    "create_custom_policy",
    # Forgetting
    "ForgetRequest",
    "ForgetResult",
    "ForgetScope",
    "ForgetStatus",
    "ForgettingService",
    # Transparency
    "AccessLogEntry",
    "ProvenanceChain",
    "ProvenanceEvent",
    "ProvenanceType",
    "TransparencyService",
    # Inspector
    "CorrectionResult",
    "ExportFormat",
    "MemoryInspection",
    "MemoryInspector",
    "MemoryRecord",
    # Audit
    "AuditAction",
    "AuditEntry",
    "AuditLogger",
    "AuditSeverity",
    # Tool
    "PrivacyAction",
    "PrivacyTool",
    "PrivacyToolResponse",
]
