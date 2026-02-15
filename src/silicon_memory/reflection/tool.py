"""Reflection tool for LLM function calling integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.reflection.types import (
    BeliefCandidate,
    Pattern,
    ReflectionConfig,
    ReflectionResult,
)
from silicon_memory.reflection.engine import ReflectionEngine

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class ReflectionAction(str, Enum):
    """Actions available through the reflection tool."""

    REFLECT = "reflect"
    ANALYZE_PATTERNS = "analyze_patterns"
    GET_PENDING = "get_pending"
    COMMIT_BELIEF = "commit_belief"
    COMMIT_ALL = "commit_all"


@dataclass
class ReflectionToolResponse:
    """Response from reflection tool invocation."""

    success: bool
    action: ReflectionAction
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


class ReflectionTool:
    """LLM-callable tool for reflection operations.

    Allows LLMs to trigger memory consolidation and manage
    the extraction of beliefs from experiences.

    Example:
        >>> from silicon_memory import SiliconMemory
        >>> from silicon_memory.reflection import ReflectionTool
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     tool = ReflectionTool(memory)
        ...     response = await tool.invoke("reflect")
        ...     print(f"Found {response.data['patterns_found']} patterns")
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        llm: Any = None,
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._engine = ReflectionEngine(memory, llm=llm, config=config)
        self._pending_candidates: list[BeliefCandidate] = []

    async def invoke(
        self,
        action: str,
        **kwargs: Any,
    ) -> ReflectionToolResponse:
        """Invoke the reflection tool with an action.

        Args:
            action: The action to perform
            **kwargs: Action-specific parameters

        Returns:
            ReflectionToolResponse with results or error
        """
        try:
            action_enum = ReflectionAction(action.lower())
        except ValueError:
            return ReflectionToolResponse(
                success=False,
                action=ReflectionAction.REFLECT,
                error=f"Unknown action: {action}. Valid: {[a.value for a in ReflectionAction]}",
            )

        handlers = {
            ReflectionAction.REFLECT: self._handle_reflect,
            ReflectionAction.ANALYZE_PATTERNS: self._handle_analyze_patterns,
            ReflectionAction.GET_PENDING: self._handle_get_pending,
            ReflectionAction.COMMIT_BELIEF: self._handle_commit_belief,
            ReflectionAction.COMMIT_ALL: self._handle_commit_all,
        }

        handler = handlers[action_enum]
        try:
            return await handler(**kwargs)
        except Exception as e:
            return ReflectionToolResponse(
                success=False,
                action=action_enum,
                error=str(e),
            )

    async def _handle_reflect(
        self,
        max_experiences: int = 100,
        auto_commit: bool = False,
        **_: Any,
    ) -> ReflectionToolResponse:
        """Handle reflect action."""
        result = await self._engine.reflect(
            max_experiences=max_experiences,
            auto_commit=auto_commit,
        )

        # Store pending candidates for later commit
        self._pending_candidates = result.new_beliefs

        return ReflectionToolResponse(
            success=True,
            action=ReflectionAction.REFLECT,
            data={
                "experiences_processed": result.experiences_processed,
                "patterns_found": len(result.patterns_found),
                "new_beliefs": len(result.new_beliefs),
                "contradictions": len(result.contradictions),
                "summary": result.summary(),
                "candidates": [
                    {
                        "id": str(c.id),
                        "content": c.content,
                        "confidence": c.confidence,
                        "is_contested": c.is_contested,
                        "reasoning": c.reasoning,
                    }
                    for c in result.new_beliefs
                ],
            },
        )

    async def _handle_analyze_patterns(
        self,
        max_experiences: int = 100,
        **_: Any,
    ) -> ReflectionToolResponse:
        """Handle analyze_patterns action."""
        patterns = await self._engine.analyze_patterns(max_experiences)

        return ReflectionToolResponse(
            success=True,
            action=ReflectionAction.ANALYZE_PATTERNS,
            data={
                "pattern_count": len(patterns),
                "patterns": [
                    {
                        "id": str(p.id),
                        "type": p.type.value,
                        "description": p.description,
                        "confidence": p.confidence,
                        "occurrences": p.occurrences,
                        "evidence_count": len(p.evidence),
                    }
                    for p in patterns
                ],
            },
        )

    async def _handle_get_pending(
        self,
        min_confidence: float = 0.5,
        **_: Any,
    ) -> ReflectionToolResponse:
        """Handle get_pending action."""
        candidates = await self._engine.get_pending_candidates(min_confidence)
        self._pending_candidates = candidates

        return ReflectionToolResponse(
            success=True,
            action=ReflectionAction.GET_PENDING,
            data={
                "candidate_count": len(candidates),
                "candidates": [
                    {
                        "id": str(c.id),
                        "content": c.content,
                        "confidence": c.confidence,
                        "is_contested": c.is_contested,
                        "supports": len(c.supports),
                        "contradicts": len(c.contradicts),
                        "reasoning": c.reasoning,
                    }
                    for c in candidates
                ],
            },
        )

    async def _handle_commit_belief(
        self,
        belief_id: str,
        **_: Any,
    ) -> ReflectionToolResponse:
        """Handle commit_belief action."""
        # Find the candidate
        candidate = None
        for c in self._pending_candidates:
            if str(c.id) == belief_id:
                candidate = c
                break

        if not candidate:
            return ReflectionToolResponse(
                success=False,
                action=ReflectionAction.COMMIT_BELIEF,
                error=f"Candidate {belief_id} not found in pending list",
            )

        belief = await self._engine.commit_belief(candidate)

        if belief:
            # Remove from pending
            self._pending_candidates = [
                c for c in self._pending_candidates if str(c.id) != belief_id
            ]

            return ReflectionToolResponse(
                success=True,
                action=ReflectionAction.COMMIT_BELIEF,
                data={
                    "belief_id": str(belief.id),
                    "content": belief.content or (
                        belief.triplet.as_text() if belief.triplet else ""
                    ),
                    "confidence": belief.confidence,
                },
            )

        return ReflectionToolResponse(
            success=False,
            action=ReflectionAction.COMMIT_BELIEF,
            error="Failed to commit belief",
        )

    async def _handle_commit_all(
        self,
        **_: Any,
    ) -> ReflectionToolResponse:
        """Handle commit_all action."""
        result = ReflectionResult(new_beliefs=self._pending_candidates)
        committed = await self._engine.commit_all_valid(result)

        # Clear pending
        self._pending_candidates = []

        return ReflectionToolResponse(
            success=True,
            action=ReflectionAction.COMMIT_ALL,
            data={
                "committed_count": len(committed),
                "beliefs": [
                    {
                        "id": str(b.id),
                        "content": b.content or (
                            b.triplet.as_text() if b.triplet else ""
                        ),
                        "confidence": b.confidence,
                    }
                    for b in committed
                ],
            },
        )

    @staticmethod
    def get_openai_schema() -> dict[str, Any]:
        """Get OpenAI function calling schema for this tool."""
        return {
            "name": "reflection",
            "description": (
                "Manage memory consolidation. Extract beliefs from experiences, "
                "analyze patterns, and commit new knowledge to memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [a.value for a in ReflectionAction],
                        "description": (
                            "The action to perform: "
                            "reflect (process experiences into beliefs), "
                            "analyze_patterns (find patterns without committing), "
                            "get_pending (list candidates awaiting approval), "
                            "commit_belief (approve and store a specific belief), "
                            "commit_all (approve all non-contested beliefs)"
                        ),
                    },
                    "max_experiences": {
                        "type": "integer",
                        "description": "Maximum experiences to process (default: 100)",
                    },
                    "auto_commit": {
                        "type": "boolean",
                        "description": "Auto-commit non-contested beliefs (default: false)",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence for pending beliefs (default: 0.5)",
                    },
                    "belief_id": {
                        "type": "string",
                        "description": "ID of belief to commit (for commit_belief action)",
                    },
                },
                "required": ["action"],
            },
        }
