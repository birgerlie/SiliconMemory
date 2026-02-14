"""Error types and exception-to-HTTP mapping for the server."""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class MemoryNotFoundError(Exception):
    """A requested memory item was not found."""

    def __init__(self, memory_type: str, memory_id: str) -> None:
        self.memory_type = memory_type
        self.memory_id = memory_id
        super().__init__(f"{memory_type} '{memory_id}' not found")


class MemoryPoolError(Exception):
    """Error getting a memory instance from the pool."""


class IngestionError(Exception):
    """Error during content ingestion."""


async def memory_not_found_handler(request: Request, exc: MemoryNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "detail": str(exc),
            "type": exc.memory_type,
            "id": exc.memory_id,
        },
    )


async def memory_pool_error_handler(request: Request, exc: MemoryPoolError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": "service_unavailable", "detail": str(exc)},
    )


async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "ingestion_failed", "detail": str(exc)},
    )


EXCEPTION_HANDLERS = {
    MemoryNotFoundError: memory_not_found_handler,
    MemoryPoolError: memory_pool_error_handler,
    IngestionError: ingestion_error_handler,
}
