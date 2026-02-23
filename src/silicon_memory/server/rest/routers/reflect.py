"""On-demand reflection and RAPTOR endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.types import ReflectionConfig
from silicon_memory.server.dependencies import get_llm, get_memory
from silicon_memory.server.schemas import ConsolidationResponse, ReflectRequest, ReflectResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reflect")
async def reflect(
    body: ReflectRequest,
    request: Request,
    memory: SiliconMemory = Depends(get_memory),
    llm: SiliconLLMProvider = Depends(get_llm),
) -> ReflectResponse:
    """Trigger a reflection cycle on-demand.

    Processes unprocessed experiences into patterns and beliefs
    using LLM-powered multi-dimensional extraction.
    """
    config = ReflectionConfig(
        max_experiences_per_batch=body.max_experiences,
        auto_commit_beliefs=body.auto_commit,
    )
    engine = ReflectionEngine(memory=memory, llm=llm, config=config)
    result = await engine.reflect(auto_commit=body.auto_commit)

    # Update app-level stats
    request.app.state.reflection_count += 1
    request.app.state.last_reflection = result.timestamp.isoformat()

    c = result.consolidation
    return ReflectResponse(
        experiences_processed=result.experiences_processed,
        patterns_found=len(result.patterns_found),
        new_beliefs=len(result.new_beliefs),
        updated_beliefs=len(result.updated_beliefs),
        contradictions=len(result.contradictions),
        consolidation=ConsolidationResponse(
            cooccurrence_links=c.cooccurrence_links,
            mc_updates=c.mc_updates,
            propagations=c.propagations,
            mc_contradictions=c.mc_contradictions,
            triple_contradictions=c.triple_contradictions,
            uncertain_beliefs=c.uncertain_beliefs,
        ),
        summary=result.summary(),
    )


# ========== RAPTOR Hierarchical Retrieval ==========


class RaptorBuildRequest(BaseModel):
    cluster_size: int = Field(default=10, ge=2, le=100)
    max_levels: int = Field(default=5, ge=1, le=10)


class RaptorBuildResponse(BaseModel):
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class RaptorSearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)
    tree_boost: float = Field(default=0.3, ge=0.0, le=1.0)


class RaptorSearchItem(BaseModel):
    content: str
    confidence: float
    memory_type: str
    relevance_score: float


class RaptorSearchResponse(BaseModel):
    results: list[RaptorSearchItem]
    count: int


@router.post("/raptor/build")
async def raptor_build(
    body: RaptorBuildRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> RaptorBuildResponse:
    """Build a RAPTOR hierarchical tree over the document store.

    Expensive operation — run infrequently (e.g., after dream cycles).
    """
    result = await memory.build_knowledge_tree(
        cluster_size=body.cluster_size,
        max_levels=body.max_levels,
    )
    return RaptorBuildResponse(
        status=result.get("status", "built"),
        details=result,
    )


@router.post("/raptor/search")
async def raptor_search(
    body: RaptorSearchRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> RaptorSearchResponse:
    """Search using RAPTOR hierarchical retrieval.

    Returns multi-granularity results: summaries for overview,
    leaves for detail.
    """
    results = await memory.search_raptor(
        query=body.query,
        k=body.k,
        tree_boost=body.tree_boost,
    )
    items = [
        RaptorSearchItem(
            content=r.content,
            confidence=r.confidence,
            memory_type=r.memory_type,
            relevance_score=r.relevance_score,
        )
        for r in results
    ]
    return RaptorSearchResponse(results=items, count=len(items))
