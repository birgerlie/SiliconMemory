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

    # Collect event stream health if available
    event_stream = None
    instances = pool.active_instances()
    if instances:
        backend = getattr(instances[0], "_backend", None)
        if backend:
            try:
                event_stream = await backend.get_event_stats()
                # Also include worker event status
                worker = getattr(request.app.state, "worker", None)
                if worker and event_stream is not None:
                    event_stream["event_driven_active"] = getattr(
                        worker, "event_stream_active", False
                    )
            except Exception:
                pass

    return StatusResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(elapsed, 1),
        active_users=len(instances),
        last_reflection=last_ref,
        reflection_count=ref_count,
        mode=request.app.state.config.mode,
        event_stream=event_stream if event_stream else None,
    )
