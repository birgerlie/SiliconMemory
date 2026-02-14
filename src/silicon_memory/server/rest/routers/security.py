"""GDPR forgetting endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.server.dependencies import get_memory
from silicon_memory.server.schemas import ForgetRequest, ForgetResponse

router = APIRouter()


@router.post("/forget")
async def forget(
    body: ForgetRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> ForgetResponse:
    if body.scope == "entity":
        if not body.entity_id:
            raise HTTPException(status_code=400, detail="entity_id required for entity scope")
        result = await memory.forget_entity(body.entity_id, reason=body.reason)

    elif body.scope == "session":
        if not body.session_id:
            raise HTTPException(status_code=400, detail="session_id required for session scope")
        result = await memory.forget_session(body.session_id, reason=body.reason)

    elif body.scope == "topic":
        if not body.topics:
            raise HTTPException(status_code=400, detail="topics required for topic scope")
        result = await memory.forget_topics(body.topics, reason=body.reason)

    elif body.scope == "query":
        if not body.query:
            raise HTTPException(status_code=400, detail="query required for query scope")
        result = await memory.selective_forget(body.query, reason=body.reason)

    elif body.scope == "all":
        result = await memory.forget_all(reason=body.reason)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown scope: {body.scope}")

    return ForgetResponse(
        deleted_count=result.deleted_count,
        scope=body.scope,
        success=result.success,
    )
