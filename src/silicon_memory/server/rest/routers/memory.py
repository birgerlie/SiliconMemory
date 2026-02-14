"""Core memory endpoints: recall, store, query, get."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends

from silicon_memory.llm.provider import SiliconLLMProvider, classify_memory_type
from silicon_memory.core.types import (
    Belief,
    Experience,
    Procedure,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.memory.silicondb_router import RecallContext, SiliconMemory
from silicon_memory.server.dependencies import get_llm, get_memory, resolve_user_context
from silicon_memory.server.errors import MemoryNotFoundError
from silicon_memory.server.schemas import (
    BeliefItem,
    MemoryItem,
    QueryRequest,
    QueryResponse,
    RecallItem,
    RecallRequest,
    RecallResponse,
    StoreRequest,
    StoreResponse,
)

router = APIRouter()


@router.post("/recall")
async def recall(
    body: RecallRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> RecallResponse:
    ctx = RecallContext(
        query=body.query,
        max_facts=body.max_facts,
        max_experiences=body.max_experiences,
        max_procedures=body.max_procedures,
        min_confidence=body.min_confidence,
        include_episodic=body.include_episodic,
        include_procedural=body.include_procedural,
        include_working=body.include_working,
        salience_profile=body.salience_profile,
    )
    result = await memory.recall(ctx)

    def _to_item(r) -> RecallItem:
        return RecallItem(
            content=r.content,
            confidence=r.confidence,
            memory_type=r.memory_type,
            relevance_score=r.relevance_score,
            belief_id=str(r.belief_id) if r.belief_id else None,
        )

    return RecallResponse(
        facts=[_to_item(f) for f in result.facts],
        experiences=[_to_item(e) for e in result.experiences],
        procedures=[_to_item(p) for p in result.procedures],
        working_context=result.working_context,
        total_items=result.total_items,
        query=result.query,
    )


@router.post("/store")
async def store(
    body: StoreRequest,
    memory: SiliconMemory = Depends(get_memory),
    llm: SiliconLLMProvider = Depends(get_llm),
) -> StoreResponse:
    item_id = uuid4()
    source = Source(id="api", type=SourceType.HUMAN, reliability=0.8)

    # Auto-classify using LLM if type not specified
    memory_type = body.type
    if memory_type == "auto":
        memory_type, _ = await classify_memory_type(llm, body.content)

    if memory_type == "belief":
        triplet = None
        if body.subject and body.predicate and body.object:
            triplet = Triplet(subject=body.subject, predicate=body.predicate, object=body.object)
        belief = Belief(
            id=item_id,
            content=body.content,
            triplet=triplet,
            confidence=body.confidence,
            source=source,
            tags=set(body.tags),
            metadata=body.metadata,
            user_id=memory.user_context.user_id,
            tenant_id=memory.user_context.tenant_id,
        )
        await memory.commit_belief(belief)

    elif memory_type == "experience":
        experience = Experience(
            id=item_id,
            content=body.content,
            outcome=body.outcome,
            session_id=body.session_id,
            tags=set(body.tags),
            user_id=memory.user_context.user_id,
            tenant_id=memory.user_context.tenant_id,
        )
        await memory.record_experience(experience)

    elif memory_type == "procedure":
        procedure = Procedure(
            id=item_id,
            name=body.name or body.content[:50],
            description=body.description or body.content,
            trigger=body.trigger or "",
            steps=body.steps,
            confidence=body.confidence,
            source=source,
            tags=set(body.tags),
            user_id=memory.user_context.user_id,
            tenant_id=memory.user_context.tenant_id,
        )
        await memory.commit_procedure(procedure)
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown type: {memory_type}")

    return StoreResponse(id=str(item_id), type=memory_type)


@router.get("/memory/{memory_type}/{memory_id}")
async def get_memory_item(
    memory_type: str,
    memory_id: str,
    memory: SiliconMemory = Depends(get_memory),
) -> MemoryItem:
    uid = UUID(memory_id)

    if memory_type == "belief":
        item = await memory.get_belief(uid)
        if not item:
            raise MemoryNotFoundError("belief", memory_id)
        return MemoryItem(
            id=str(item.id),
            type="belief",
            content=item.content or (item.triplet.as_text() if item.triplet else ""),
            confidence=item.confidence,
            metadata=item.metadata,
        )

    elif memory_type == "experience":
        item = await memory.get_experience(uid)
        if not item:
            raise MemoryNotFoundError("experience", memory_id)
        return MemoryItem(
            id=str(item.id),
            type="experience",
            content=item.content,
            metadata=item.context,
        )

    elif memory_type == "procedure":
        item = await memory.get_procedure(uid)
        if not item:
            raise MemoryNotFoundError("procedure", memory_id)
        return MemoryItem(
            id=str(item.id),
            type="procedure",
            content=item.description,
            confidence=item.confidence,
        )

    elif memory_type == "decision":
        item = await memory.get_decision(uid)
        if not item:
            raise MemoryNotFoundError("decision", memory_id)
        return MemoryItem(
            id=str(item.id),
            type="decision",
            content=item.description,
            metadata=item.metadata,
        )

    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown memory type: {memory_type}")


@router.post("/query")
async def query_beliefs(
    body: QueryRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> QueryResponse:
    beliefs = await memory.query_beliefs(
        body.query, limit=body.limit, min_confidence=body.min_confidence
    )
    items = [
        BeliefItem(
            id=str(b.id),
            content=b.content or (b.triplet.as_text() if b.triplet else ""),
            confidence=b.confidence,
            status=b.status.value,
            tags=list(b.tags),
            subject=b.triplet.subject if b.triplet else None,
            predicate=b.triplet.predicate if b.triplet else None,
            object=b.triplet.object if b.triplet else None,
        )
        for b in beliefs
    ]
    return QueryResponse(beliefs=items, query=body.query, count=len(items))
