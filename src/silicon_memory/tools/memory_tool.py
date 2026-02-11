"""Memory tool for LLM function calling integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.utils import utc_now
from silicon_memory.core.types import (
    Belief,
    Experience,
    Procedure,
    RecallResult,
    Source,
    SourceType,
    Triplet,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory, RecallContext


class MemoryAction(str, Enum):
    """Actions available through the memory tool."""

    RECALL = "recall"
    STORE_FACT = "store_fact"
    STORE_EXPERIENCE = "store_experience"
    STORE_PROCEDURE = "store_procedure"
    SET_CONTEXT = "set_context"
    GET_CONTEXT = "get_context"
    WHAT_DO_YOU_KNOW = "what_do_you_know"
    STORE_DECISION = "store_decision"
    RECALL_DECISIONS = "recall_decisions"
    SWITCH_CONTEXT = "switch_context"
    RESUME_CONTEXT = "resume_context"


@dataclass
class MemoryToolResponse:
    """Response from memory tool invocation."""

    success: bool
    action: MemoryAction
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


class MemoryTool:
    """LLM-callable memory tool for context enhancement.

    This tool provides a function-calling interface for LLMs to:
    - Recall relevant memories for context
    - Store new facts, experiences, and procedures
    - Manage working memory context
    - Query knowledge with proofs

    Example:
        >>> from silicon_memory import SiliconMemory, MemoryTool
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     tool = MemoryTool(memory)
        ...     response = await tool.invoke("recall", query="Python")
        ...     print(response.data)
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        default_source: Source | None = None,
    ) -> None:
        self._memory = memory
        self._default_source = default_source or Source(
            id="llm_tool",
            type=SourceType.EXTERNAL,
            reliability=0.7,
        )

    async def invoke(
        self,
        action: str,
        **kwargs: Any,
    ) -> MemoryToolResponse:
        """Invoke the memory tool with an action.

        Args:
            action: The action to perform
            **kwargs: Action-specific parameters

        Returns:
            MemoryToolResponse with results or error
        """
        try:
            action_enum = MemoryAction(action.lower())
        except ValueError:
            return MemoryToolResponse(
                success=False,
                action=MemoryAction.RECALL,
                error=f"Unknown action: {action}. Valid actions: {[a.value for a in MemoryAction]}",
            )

        handlers = {
            MemoryAction.RECALL: self._handle_recall,
            MemoryAction.STORE_FACT: self._handle_store_fact,
            MemoryAction.STORE_EXPERIENCE: self._handle_store_experience,
            MemoryAction.STORE_PROCEDURE: self._handle_store_procedure,
            MemoryAction.SET_CONTEXT: self._handle_set_context,
            MemoryAction.GET_CONTEXT: self._handle_get_context,
            MemoryAction.WHAT_DO_YOU_KNOW: self._handle_what_do_you_know,
            MemoryAction.STORE_DECISION: self._handle_store_decision,
            MemoryAction.RECALL_DECISIONS: self._handle_recall_decisions,
            MemoryAction.SWITCH_CONTEXT: self._handle_switch_context,
            MemoryAction.RESUME_CONTEXT: self._handle_resume_context,
        }

        handler = handlers[action_enum]
        try:
            return await handler(**kwargs)
        except Exception as e:
            return MemoryToolResponse(
                success=False,
                action=action_enum,
                error=str(e),
            )

    async def _handle_recall(
        self,
        query: str = "",
        max_facts: int = 10,
        max_experiences: int = 5,
        max_procedures: int = 3,
        min_confidence: float = 0.3,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle recall action."""
        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(
            query=query,
            max_facts=max_facts,
            max_experiences=max_experiences,
            max_procedures=max_procedures,
            min_confidence=min_confidence,
        )

        response = await self._memory.recall(ctx)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.RECALL,
            data={
                "facts": [self._recall_to_dict(r) for r in response.facts],
                "experiences": [self._recall_to_dict(r) for r in response.experiences],
                "procedures": [self._recall_to_dict(r) for r in response.procedures],
                "working_context": response.working_context,
                "total_items": response.total_items,
                "query": response.query,
            },
        )

    async def _handle_store_fact(
        self,
        content: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,  # noqa: A002
        confidence: float = 0.7,
        source_id: str | None = None,
        source_name: str | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle store_fact action."""
        triplet = None
        if subject and predicate and object:
            triplet = Triplet(
                subject=subject,
                predicate=predicate,
                object=object,
            )

        source = self._default_source
        if source_id or source_name:
            source = Source(
                id=source_id or str(uuid4()),
                type=SourceType.HUMAN,
                reliability=0.8,
            )

        # Get user context for security metadata
        user_ctx = getattr(self._memory, 'user_context', None)

        belief = Belief(
            id=uuid4(),
            content=content,
            triplet=triplet,
            confidence=min(1.0, max(0.0, confidence)),
            source=source,
            tags=set(tags) if tags else set(),
            user_id=user_ctx.user_id if user_ctx else None,
            tenant_id=user_ctx.tenant_id if user_ctx else None,
        )

        await self._memory.commit_belief(belief)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.STORE_FACT,
            data={
                "belief_id": str(belief.id),
                "content": content,
                "triplet": triplet.as_text() if triplet else None,
                "confidence": belief.confidence,
            },
        )

    async def _handle_store_experience(
        self,
        content: str,
        outcome: str | None = None,
        session_id: str | None = None,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle store_experience action."""
        # Get user context for security metadata
        user_ctx = getattr(self._memory, 'user_context', None)

        experience = Experience(
            id=uuid4(),
            content=content,
            outcome=outcome,
            session_id=session_id or (user_ctx.session_id if user_ctx else None),
            user_id=user_ctx.user_id if user_ctx else None,
            tenant_id=user_ctx.tenant_id if user_ctx else None,
        )

        await self._memory.record_experience(experience)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.STORE_EXPERIENCE,
            data={
                "experience_id": str(experience.id),
                "content": content,
                "outcome": outcome,
            },
        )

    async def _handle_store_procedure(
        self,
        name: str,
        steps: list[str],
        trigger: str = "",
        description: str = "",
        confidence: float = 0.5,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle store_procedure action."""
        # Get user context for security metadata
        user_ctx = getattr(self._memory, 'user_context', None)

        procedure = Procedure(
            id=uuid4(),
            name=name,
            steps=steps,
            trigger=trigger,
            description=description,
            confidence=min(1.0, max(0.0, confidence)),
            user_id=user_ctx.user_id if user_ctx else None,
            tenant_id=user_ctx.tenant_id if user_ctx else None,
        )

        await self._memory.commit_procedure(procedure)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.STORE_PROCEDURE,
            data={
                "procedure_id": str(procedure.id),
                "name": name,
                "steps": steps,
            },
        )

    async def _handle_set_context(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle set_context action."""
        await self._memory.set_context(key, value, ttl_seconds)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.SET_CONTEXT,
            data={
                "key": key,
                "ttl_seconds": ttl_seconds,
            },
        )

    async def _handle_get_context(
        self,
        key: str | None = None,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle get_context action."""
        if key:
            value = await self._memory.get_context(key)
            data = {"key": key, "value": value}
        else:
            data = {"context": await self._memory.get_all_context()}

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.GET_CONTEXT,
            data=data,
        )

    async def _handle_what_do_you_know(
        self,
        query: str,
        min_confidence: float = 0.3,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle what_do_you_know action."""
        proof = await self._memory.what_do_you_know(query, min_confidence)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.WHAT_DO_YOU_KNOW,
            data={
                "query": proof.query,
                "total_confidence": proof.total_confidence,
                "belief_count": len(proof.beliefs),
                "source_count": len(proof.sources),
                "contradiction_count": len(proof.contradictions),
                "report": proof.as_report(),
                "beliefs": [
                    {
                        "id": str(b.id),
                        "content": b.content or (b.triplet.as_text() if b.triplet else ""),
                        "confidence": b.confidence,
                        "source": b.source.id if b.source else None,
                        "evidence_for": len(b.evidence_for),
                        "evidence_against": len(b.evidence_against),
                    }
                    for b in proof.beliefs
                ],
            },
        )

    async def _handle_store_decision(
        self,
        title: str = "",
        description: str = "",
        assumptions: list[dict] | None = None,
        alternatives: list[dict] | None = None,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle store_decision action."""
        from silicon_memory.core.decision import Decision, Assumption, Alternative
        from uuid import UUID

        parsed_assumptions = []
        for a in (assumptions or []):
            parsed_assumptions.append(Assumption(
                belief_id=UUID(a["belief_id"]) if "belief_id" in a else uuid4(),
                description=a.get("description", ""),
                confidence_at_decision=a.get("confidence_at_decision", 0.5),
                is_critical=a.get("is_critical", False),
            ))

        parsed_alternatives = []
        for alt in (alternatives or []):
            parsed_alternatives.append(Alternative(
                title=alt.get("title", ""),
                description=alt.get("description", ""),
                rejection_reason=alt.get("rejection_reason", ""),
            ))

        user_ctx = getattr(self._memory, 'user_context', None)
        decision = Decision(
            title=title,
            description=description,
            assumptions=parsed_assumptions,
            alternatives=parsed_alternatives,
            decided_by=user_ctx.user_id if user_ctx else None,
            session_id=getattr(user_ctx, 'session_id', None) if user_ctx else None,
            user_id=user_ctx.user_id if user_ctx else None,
            tenant_id=user_ctx.tenant_id if user_ctx else None,
        )

        snapshot_id = await self._memory.commit_decision(decision)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.STORE_DECISION,
            data={
                "decision_id": str(decision.id),
                "title": title,
                "belief_snapshot_id": snapshot_id,
            },
        )

    async def _handle_recall_decisions(
        self,
        query: str = "",
        k: int = 10,
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle recall_decisions action."""
        decisions = await self._memory.recall_decisions(query, k)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.RECALL_DECISIONS,
            data={
                "decisions": [
                    {
                        "id": str(d.id),
                        "title": d.title,
                        "description": d.description,
                        "status": d.status.value,
                        "decided_at": d.decided_at.isoformat(),
                        "assumption_count": len(d.assumptions),
                        "alternative_count": len(d.alternatives),
                    }
                    for d in decisions
                ],
                "count": len(decisions),
            },
        )

    async def _handle_switch_context(
        self,
        task_context: str = "",
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle switch_context action — snapshot current context."""
        snapshot = await self._memory.create_snapshot(task_context)

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.SWITCH_CONTEXT,
            data={
                "task_context": snapshot.task_context,
                "summary": snapshot.summary,
                "next_steps": snapshot.next_steps,
                "open_questions": getattr(snapshot, "open_questions", []),
            },
        )

    async def _handle_resume_context(
        self,
        task_context: str = "",
        **_: Any,
    ) -> MemoryToolResponse:
        """Handle resume_context action — retrieve latest snapshot."""
        snapshot = await self._memory.get_latest_snapshot(task_context)

        if snapshot is None:
            return MemoryToolResponse(
                success=True,
                action=MemoryAction.RESUME_CONTEXT,
                data={"found": False, "task_context": task_context},
            )

        return MemoryToolResponse(
            success=True,
            action=MemoryAction.RESUME_CONTEXT,
            data={
                "found": True,
                "task_context": snapshot.task_context,
                "summary": snapshot.summary,
                "working_memory": snapshot.working_memory,
                "next_steps": snapshot.next_steps,
                "open_questions": getattr(snapshot, "open_questions", []),
            },
        )

    def _recall_to_dict(self, result: RecallResult) -> dict[str, Any]:
        """Convert RecallResult to dictionary."""
        return {
            "content": result.content,
            "confidence": result.confidence,
            "memory_type": result.memory_type,
            "relevance_score": result.relevance_score,
            "source": result.source.id if result.source else None,
        }

    @staticmethod
    def get_openai_schema() -> dict[str, Any]:
        """Get OpenAI function calling schema for this tool."""
        return {
            "name": "memory",
            "description": (
                "Access and manage the memory system. Use this to recall relevant "
                "context, store new facts and experiences, and query knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [a.value for a in MemoryAction],
                        "description": (
                            "The action to perform: "
                            "recall (get relevant memories), "
                            "store_fact (save a fact/belief), "
                            "store_experience (save an experience), "
                            "store_procedure (save how-to knowledge), "
                            "set_context (set working memory), "
                            "get_context (get working memory), "
                            "what_do_you_know (query with proof)"
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for recall/what_do_you_know actions",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for store_fact/store_experience",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject of triplet (for store_fact)",
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Predicate of triplet (for store_fact)",
                    },
                    "object": {
                        "type": "string",
                        "description": "Object of triplet (for store_fact)",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence level (0-1)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for store_procedure",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps for store_procedure",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key for set_context/get_context",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value for set_context",
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "TTL in seconds for set_context (default: 300)",
                    },
                    "task_context": {
                        "type": "string",
                        "description": "Task context identifier for switch_context/resume_context",
                    },
                },
                "required": ["action"],
            },
        }
