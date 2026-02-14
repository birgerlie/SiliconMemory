"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ========== Common ==========

class ErrorResponse(BaseModel):
    error: str
    detail: str


# ========== Recall ==========

class RecallRequest(BaseModel):
    query: str
    max_facts: int = 20
    max_experiences: int = 10
    max_procedures: int = 5
    min_confidence: float = 0.3
    include_episodic: bool = True
    include_procedural: bool = True
    include_working: bool = True
    salience_profile: str | None = None


class RecallItem(BaseModel):
    content: str
    confidence: float
    memory_type: str
    relevance_score: float
    belief_id: str | None = None


class RecallResponse(BaseModel):
    facts: list[RecallItem]
    experiences: list[RecallItem]
    procedures: list[RecallItem]
    working_context: dict[str, Any]
    total_items: int
    query: str


# ========== Store ==========

class StoreRequest(BaseModel):
    """Store a memory item. Type determines what kind."""

    type: str = Field(default="auto", description="belief | experience | procedure | auto (LLM classifies)")
    content: str
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Belief-specific
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None

    # Experience-specific
    outcome: str | None = None
    session_id: str | None = None

    # Procedure-specific
    name: str | None = None
    description: str | None = None
    trigger: str | None = None
    steps: list[str] = Field(default_factory=list)


class StoreResponse(BaseModel):
    id: str
    type: str
    stored: bool = True


# ========== Query ==========

class QueryRequest(BaseModel):
    query: str
    limit: int = 10
    min_confidence: float = 0.0


class BeliefItem(BaseModel):
    id: str
    content: str
    confidence: float
    status: str
    tags: list[str] = Field(default_factory=list)
    subject: str | None = None
    predicate: str | None = None
    object_: str | None = Field(None, alias="object")


class QueryResponse(BaseModel):
    beliefs: list[BeliefItem]
    query: str
    count: int


# ========== Get ==========

class MemoryItem(BaseModel):
    id: str
    type: str
    content: str
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ========== Working Memory ==========

class WorkingMemoryEntry(BaseModel):
    key: str
    value: Any


class SetWorkingRequest(BaseModel):
    value: Any
    ttl_seconds: int = 300


# ========== Decisions ==========

class AssumptionInput(BaseModel):
    belief_id: str
    description: str
    confidence_at_decision: float
    is_critical: bool = False


class AlternativeInput(BaseModel):
    title: str
    description: str
    rejection_reason: str = ""


class DecisionStoreRequest(BaseModel):
    title: str
    description: str
    assumptions: list[AssumptionInput] = Field(default_factory=list)
    alternatives: list[AlternativeInput] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionSearchRequest(BaseModel):
    query: str
    limit: int = 10
    min_confidence: float = 0.0


class DecisionItem(BaseModel):
    id: str
    title: str
    description: str
    status: str
    decided_at: str
    outcome: str | None = None
    tags: list[str] = Field(default_factory=list)


# ========== Ingestion ==========

class IngestRequest(BaseModel):
    source_type: str = Field(description="meeting | chat | email | document | news")
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    experiences_created: int
    entities_resolved: int
    decisions_detected: int
    action_items_detected: int
    errors: list[str]
    source_type: str


# ========== Forget ==========

class ForgetRequest(BaseModel):
    scope: str = Field(description="entity | session | topic | query | all")
    entity_id: str | None = None
    session_id: str | None = None
    topics: list[str] = Field(default_factory=list)
    query: str | None = None
    reason: str | None = None


class ForgetResponse(BaseModel):
    deleted_count: int
    scope: str
    success: bool


# ========== Reflection ==========

class ReflectRequest(BaseModel):
    max_experiences: int = 100
    auto_commit: bool = True


class ReflectResponse(BaseModel):
    experiences_processed: int
    patterns_found: int
    new_beliefs: int
    updated_beliefs: int
    contradictions: int
    summary: str


# ========== Health ==========

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    uptime_seconds: float


class StatusResponse(BaseModel):
    status: str = "ok"
    version: str
    uptime_seconds: float
    active_users: int
    last_reflection: str | None = None
    reflection_count: int = 0
    mode: str
