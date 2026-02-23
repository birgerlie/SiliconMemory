"""Tests for procedure detector experience grouping fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from silicon_memory.reflection.procedure_detector import ProcedureDetector


@dataclass
class _StubExperience:
    content: str
    session_id: str | None = None
    context: dict = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime(2026, 2, 19, tzinfo=timezone.utc))


@pytest.mark.asyncio
async def test_detect_from_experiences_groups_by_document_id_when_session_missing() -> None:
    memory = AsyncMock()
    memory.get_recent_experiences = AsyncMock(
        return_value=[
            _StubExperience(content="Step 1 run tests", context={"document_id": "a"}),
            _StubExperience(content="Step 2 deploy", context={"document_id": "b"}),
        ],
    )
    detector = ProcedureDetector(memory=memory, llm=AsyncMock())
    detector._extract_procedures = AsyncMock(return_value=[])  # noqa: SLF001

    await detector.detect_from_experiences(limit=10)

    detector._extract_procedures.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_detect_from_experiences_uses_day_fallback_grouping() -> None:
    memory = AsyncMock()
    memory.get_recent_experiences = AsyncMock(
        return_value=[
            _StubExperience(content="Action alpha", context={}),
            _StubExperience(
                content="Action beta",
                context={},
                occurred_at=datetime(2026, 2, 20, tzinfo=timezone.utc),
            ),
        ],
    )
    detector = ProcedureDetector(memory=memory, llm=AsyncMock())
    detector._extract_procedures = AsyncMock(return_value=[])  # noqa: SLF001

    await detector.detect_from_experiences(limit=10)

    detector._extract_procedures.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_derive_from_timeline_creates_procedure_from_ordered_events() -> None:
    class _Triple:
        def __init__(self, date_value: str, event_text: str) -> None:
            self.subject = "Case 22-1426"
            self.predicate = "occurred_on"
            self.object_value = date_value
            self.metadata = {"source_metadata": {"event_text": event_text}}

    backend = AsyncMock()
    backend._query_triples = lambda **kwargs: [  # noqa: ARG005
        _Triple("2024-09-17", "Decision issued"),
        _Triple("2024-03-12", "Oral argument heard"),
    ]
    memory = AsyncMock()
    memory._backend = backend

    detector = ProcedureDetector(memory=memory, llm=AsyncMock())
    procedures = await detector._derive_from_timeline(limit=10)  # noqa: SLF001

    assert len(procedures) == 1
    assert procedures[0].name.startswith("Case 22-1426")
    assert procedures[0].steps[0].startswith("2024-03-12")
    assert procedures[0].steps[1].startswith("2024-09-17")


@pytest.mark.asyncio
async def test_derive_from_repeated_predicates_creates_generic_procedure() -> None:
    class _Triple:
        def __init__(self, predicate: str) -> None:
            self.external_id = "t/u/belief-1"
            self.subject = "Case 22-1426"
            self.predicate = predicate
            self.object_value = "obj"
            self.metadata = {"owner_id": "u", "tenant_id": "t", "privacy_level": "private"}

    backend = AsyncMock()
    backend._query_triples = lambda **kwargs: [  # noqa: ARG005
        _Triple("appeals"),
        _Triple("appeals"),
        _Triple("challenges"),
        _Triple("challenges"),
    ]
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005, SLF001
    memory = AsyncMock()
    memory._backend = backend

    detector = ProcedureDetector(memory=memory, llm=AsyncMock())
    procedures = await detector._derive_from_repeated_predicates(limit=10)  # noqa: SLF001

    assert len(procedures) == 1
    assert procedures[0].name.startswith("Case 22-1426")
    assert len(procedures[0].steps) >= 2
