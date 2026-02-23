"""Health and status endpoints."""

from __future__ import annotations

import contextlib
import time

from fastapi import APIRouter, Request

from silicon_memory.server.schemas import HealthResponse, StatusResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    elapsed = time.monotonic() - request.app.state.start_time

    worker_health = None
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is not None:
        worker_health = registry.health()

    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(elapsed, 1),
        workers=worker_health,
    )


@router.get("/status")
async def status(request: Request) -> StatusResponse:
    elapsed = time.monotonic() - request.app.state.start_time
    pool = request.app.state.pool
    last_ref = getattr(request.app.state, "last_reflection", None)
    ref_count = getattr(request.app.state, "reflection_count", 0)

    # Collect event stream health if available
    event_stream = None
    instances = pool.active_instances()
    if instances:
        backend = getattr(instances[0], "_backend", None)
        if backend:
            with contextlib.suppress(Exception):
                event_stream = await backend.get_event_stats()

    # Worker health from registry
    worker_health = None
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is not None:
        worker_health = registry.health()
        # Include event_driven_active summary in event_stream
        if event_stream is not None:
            active_streams = sum(
                1 for w in registry.workers if w.event_stream_active
            )
            event_stream["event_driven_workers"] = active_streams

    return StatusResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(elapsed, 1),
        active_users=len(instances),
        last_reflection=last_ref,
        reflection_count=ref_count,
        mode=request.app.state.config.mode,
        event_stream=event_stream if event_stream else None,
        workers=worker_health,
    )
