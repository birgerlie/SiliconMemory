"""On-demand reflection endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.types import ReflectionConfig
from silicon_memory.server.dependencies import get_memory
from silicon_memory.server.schemas import ReflectRequest, ReflectResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reflect")
async def reflect(
    body: ReflectRequest,
    request: Request,
    memory: SiliconMemory = Depends(get_memory),
) -> ReflectResponse:
    """Trigger a reflection cycle on-demand.

    Processes unprocessed experiences into patterns and beliefs.
    """
    config = ReflectionConfig(
        max_experiences_per_batch=body.max_experiences,
        auto_commit_beliefs=body.auto_commit,
    )
    engine = ReflectionEngine(memory, config)
    result = await engine.reflect(auto_commit=body.auto_commit)

    # Update app-level stats
    request.app.state.reflection_count += 1
    request.app.state.last_reflection = result.timestamp.isoformat()

    return ReflectResponse(
        experiences_processed=result.experiences_processed,
        patterns_found=len(result.patterns_found),
        new_beliefs=len(result.new_beliefs),
        updated_beliefs=len(result.updated_beliefs),
        contradictions=len(result.contradictions),
        summary=result.summary(),
    )
