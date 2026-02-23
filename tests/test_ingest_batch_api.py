"""Unit tests for SiliconMemory ingest batch API."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from silicon_memory.core.types import Experience
from silicon_memory.memory.silicondb_router import SiliconMemory


def _exp() -> Experience:
    return Experience(id=uuid4(), content="doc text", context={}, processed=False)


@pytest.mark.asyncio
async def test_ingest_experiences_batch_succeeded() -> None:
    memory = object.__new__(SiliconMemory)
    memory.record_experience = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    memory.wait_for_ingest_visibility = AsyncMock(return_value=True)  # type: ignore[attr-defined]

    exps = [_exp(), _exp()]
    out = await SiliconMemory.ingest_experiences_batch(memory, exps, wait_for_visibility=True)

    assert out.state == "SUCCEEDED"
    assert out.visibility_ok is True
    assert out.receipt.accepted_count == 2
    assert out.receipt.failed_count == 0
    assert len(out.receipt.accepted_experience_ids) == 2


@pytest.mark.asyncio
async def test_ingest_experiences_batch_partial() -> None:
    memory = object.__new__(SiliconMemory)
    fail_id = None

    async def _record(exp: Experience) -> None:
        if str(exp.id) == fail_id:
            raise RuntimeError("write failed")

    memory.record_experience = AsyncMock(side_effect=_record)  # type: ignore[attr-defined]
    memory.wait_for_ingest_visibility = AsyncMock(return_value=True)  # type: ignore[attr-defined]

    exps = [_exp(), _exp()]
    fail_id = str(exps[1].id)
    out = await SiliconMemory.ingest_experiences_batch(memory, exps, wait_for_visibility=True)

    assert out.state == "PARTIAL"
    assert out.visibility_ok is True
    assert out.receipt.accepted_count == 1
    assert out.receipt.failed_count == 1
    assert len(out.receipt.errors) == 1


@pytest.mark.asyncio
async def test_ingest_experiences_batch_timeout() -> None:
    memory = object.__new__(SiliconMemory)
    memory.record_experience = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    memory.wait_for_ingest_visibility = AsyncMock(return_value=False)  # type: ignore[attr-defined]

    out = await SiliconMemory.ingest_experiences_batch(memory, [_exp()], wait_for_visibility=True)

    assert out.state == "TIMEOUT"
    assert out.visibility_ok is False
    assert out.receipt.accepted_count == 1


@pytest.mark.asyncio
async def test_ingest_experiences_batch_failed_when_all_writes_fail() -> None:
    memory = object.__new__(SiliconMemory)
    memory.record_experience = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]
    memory.wait_for_ingest_visibility = AsyncMock(return_value=True)  # type: ignore[attr-defined]

    out = await SiliconMemory.ingest_experiences_batch(memory, [_exp()], wait_for_visibility=True)

    assert out.state == "FAILED"
    assert out.receipt.accepted_count == 0
    assert out.receipt.failed_count == 1
