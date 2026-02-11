"""Silicon Memory - Multi-layered memory system backed by SiliconDB.

A cognitive memory system for LLM augmentation with:
- Semantic memory (beliefs/facts as triplets)
- Episodic memory (experiences/events)
- Procedural memory (how-to knowledge)
- Working memory (short-term context with TTL)

Plus:
- Multi-user security with tenant isolation
- Privacy controls (private, workspace, public)
- Forgetting API (GDPR-compliant deletion)
- Transparency ("why do you know")
- Reflection engine for experience -> belief extraction
- Graph query layer for relationship traversal
- LLM client integrations (OpenAI, Anthropic)

Example:
    >>> from silicon_memory import SiliconMemory, Belief, Triplet, UserContext, PrivacyLevel
    >>> from uuid import uuid4
    >>>
    >>> # Create user context (required)
    >>> user_ctx = UserContext(
    ...     user_id="user-123",
    ...     tenant_id="acme-corp",
    ...     default_privacy=PrivacyLevel.PRIVATE,
    ... )
    >>>
    >>> with SiliconMemory("/path/to/db", user_context=user_ctx) as memory:
    ...     # Store a belief
    ...     belief = Belief(
    ...         id=uuid4(),
    ...         triplet=Triplet("Python", "is", "programming language"),
    ...         confidence=0.9,
    ...     )
    ...     await memory.commit_belief(belief)
    ...
    ...     # Query what we know
    ...     proof = await memory.what_do_you_know("Python")
    ...     print(proof.as_report())
    ...
    ...     # Forget a session
    ...     await memory.forget_session("session-abc")
"""

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Experience,
    KnowledgeProof,
    Procedure,
    RecallResult,
    Source,
    SourceType,
    TemporalContext,
    Triplet,
)
from silicon_memory.memory.silicondb_router import (
    SiliconMemory,
    RecallContext,
    RecallResponse,
)
from silicon_memory.storage.silicondb_backend import SiliconDBBackend, SiliconDBConfig
from silicon_memory.temporal.decay import DecayConfig, DecayFunction
from silicon_memory.tools.memory_tool import MemoryTool, MemoryToolResponse
from silicon_memory.tools.query_tool import QueryTool, QueryResponse

# Security module
from silicon_memory.security import (
    # Core types
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
    # Configuration
    SecurityConfig,
    # Preferences
    MemoryPreferences,
    # Authorization
    AccessDecision,
    Permission,
    PolicyEngine,
    # Forgetting
    ForgetResult,
    ForgetScope,
    ForgettingService,
    # Transparency
    ProvenanceChain,
    TransparencyService,
    # Inspector
    MemoryInspection,
    MemoryInspector,
    MemoryRecord,
    # Audit
    AuditAction,
    AuditEntry,
    AuditLogger,
    # Tool
    PrivacyTool,
)

# Reflection engine
from silicon_memory.reflection import (
    ReflectionEngine,
    ReflectionConfig,
    ReflectionResult,
    ReflectionTool,
    Pattern,
    PatternType,
    BeliefCandidate,
)

# Graph query layer
from silicon_memory.graph import (
    GraphQuery,
    GraphQueryBuilder,
    EntityExplorer,
    GraphTool,
    GraphNode,
    GraphEdge,
    GraphPath,
    EntityProfile,
)

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "SiliconMemory",
    "RecallContext",
    "RecallResponse",
    # Storage
    "SiliconDBBackend",
    "SiliconDBConfig",
    # Core types
    "Belief",
    "BeliefStatus",
    "Experience",
    "KnowledgeProof",
    "Procedure",
    "RecallResult",
    "Source",
    "SourceType",
    "TemporalContext",
    "Triplet",
    # Temporal
    "DecayConfig",
    "DecayFunction",
    # Tools
    "MemoryTool",
    "MemoryToolResponse",
    "QueryTool",
    "QueryResponse",
    # Reflection
    "ReflectionEngine",
    "ReflectionConfig",
    "ReflectionResult",
    "ReflectionTool",
    "Pattern",
    "PatternType",
    "BeliefCandidate",
    # Graph
    "GraphQuery",
    "GraphQueryBuilder",
    "EntityExplorer",
    "GraphTool",
    "GraphNode",
    "GraphEdge",
    "GraphPath",
    "EntityProfile",
    # Security - Core types
    "DataClassification",
    "PrivacyLevel",
    "PrivacyMetadata",
    "UserContext",
    # Security - Configuration
    "SecurityConfig",
    # Security - Preferences
    "MemoryPreferences",
    # Security - Authorization
    "AccessDecision",
    "Permission",
    "PolicyEngine",
    # Security - Forgetting
    "ForgetResult",
    "ForgetScope",
    "ForgettingService",
    # Security - Transparency
    "ProvenanceChain",
    "TransparencyService",
    # Security - Inspector
    "MemoryInspection",
    "MemoryInspector",
    "MemoryRecord",
    # Security - Audit
    "AuditAction",
    "AuditEntry",
    "AuditLogger",
    # Security - Tool
    "PrivacyTool",
]
