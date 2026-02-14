"""Working memory CRUD endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.server.dependencies import get_memory
from silicon_memory.server.schemas import SetWorkingRequest, WorkingMemoryEntry

router = APIRouter()


@router.get("/working")
async def get_all_working(
    memory: SiliconMemory = Depends(get_memory),
) -> dict[str, Any]:
    return await memory.get_all_context()


@router.put("/working/{key}")
async def set_working(
    key: str,
    body: SetWorkingRequest,
    memory: SiliconMemory = Depends(get_memory),
) -> WorkingMemoryEntry:
    await memory.set_context(key, body.value, body.ttl_seconds)
    return WorkingMemoryEntry(key=key, value=body.value)


@router.delete("/working/{key}")
async def delete_working(
    key: str,
    memory: SiliconMemory = Depends(get_memory),
) -> dict[str, bool]:
    deleted = await memory.delete_context(key)
    return {"deleted": deleted}
