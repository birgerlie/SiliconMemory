"""Decision store and search endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from silicon_memory.core.decision import Alternative, Assumption, Decision
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.server.dependencies import get_memory
from silicon_memory.server.schemas import (
    DecisionItem,
    DecisionSearchRequest,
    DecisionStoreRequest,
)

router = APIRouter()


@router.post("/decisions")
async def store_decision(
    body: DecisionStoreRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> DecisionItem:
    assumptions = [
        Assumption(
            belief_id=UUID(a.belief_id),
            description=a.description,
            confidence_at_decision=a.confidence_at_decision,
            is_critical=a.is_critical,
        )
        for a in body.assumptions
    ]
    alternatives = [
        Alternative(
            title=alt.title,
            description=alt.description,
            rejection_reason=alt.rejection_reason,
        )
        for alt in body.alternatives
    ]

    decision = Decision(
        title=body.title,
        description=body.description,
        assumptions=assumptions,
        alternatives=alternatives,
        tags=set(body.tags),
        metadata=body.metadata,
        user_id=memory.user_context.user_id,
        tenant_id=memory.user_context.tenant_id,
    )

    await memory.commit_decision(decision)

    return DecisionItem(
        id=str(decision.id),
        title=decision.title,
        description=decision.description,
        status=decision.status.value,
        decided_at=decision.decided_at.isoformat(),
        tags=list(decision.tags),
    )


@router.post("/decisions/search")
async def search_decisions(
    body: DecisionSearchRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> list[DecisionItem]:
    decisions = await memory.recall_decisions(
        body.query, k=body.limit, min_confidence=body.min_confidence
    )
    return [
        DecisionItem(
            id=str(d.id),
            title=d.title,
            description=d.description,
            status=d.status.value,
            decided_at=d.decided_at.isoformat(),
            outcome=d.outcome,
            tags=list(d.tags),
        )
        for d in decisions
    ]
