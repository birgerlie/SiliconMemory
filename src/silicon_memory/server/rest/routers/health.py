"""Health and status endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from silicon_memory.server.schemas import HealthResponse, StatusResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    elapsed = time.monotonic() - request.app.state.start_time
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(elapsed, 1),
    )


@router.get("/status")
async def status(request: Request) -> StatusResponse:
    elapsed = time.monotonic() - request.app.state.start_time
    pool = request.app.state.pool
    last_ref = getattr(request.app.state, "last_reflection", None)
    ref_count = getattr(request.app.state, "reflection_count", 0)

    return StatusResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(elapsed, 1),
        active_users=len(pool.active_instances()),
        last_reflection=last_ref,
        reflection_count=ref_count,
        mode=request.app.state.config.mode,
    )
