"""Core types and protocols for Silicon Memory."""

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Experience,
    Procedure,
    RecallResult,
    Source,
    SourceType,
    TemporalContext,
    Triplet,
)
from silicon_memory.core.claims import (
    AntiTrivialityPolicy,
    AntiTrivialityResult,
    ClaimDecision,
    ClaimDecisionLabel,
    ClaimMergeCandidate,
    ClaimPrecisionJudge,
    ProvenanceContract,
    ProvenanceValidation,
    build_provenance_contract,
    validate_provenance_contract,
)
from silicon_memory.core.protocols import (
    EpisodicMemory,
    LLMProvider,
    MemoryStore,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from silicon_memory.core.exceptions import (
    MemoryError,
    BeliefConflictError,
    StorageError,
    ValidationError,
)

__all__ = [
    # Types
    "Belief",
    "BeliefStatus",
    "Experience",
    "Procedure",
    "RecallResult",
    "Source",
    "SourceType",
    "TemporalContext",
    "Triplet",
    # Protocols
    "EpisodicMemory",
    "LLMProvider",
    "MemoryStore",
    "ProceduralMemory",
    "SemanticMemory",
    "WorkingMemory",
    # Exceptions
    "MemoryError",
    "BeliefConflictError",
    "StorageError",
    "ValidationError",
    # Claim strategy
    "AntiTrivialityPolicy",
    "AntiTrivialityResult",
    "ClaimDecision",
    "ClaimDecisionLabel",
    "ClaimMergeCandidate",
    "ClaimPrecisionJudge",
    "ProvenanceContract",
    "ProvenanceValidation",
    "build_provenance_contract",
    "validate_provenance_contract",
]
