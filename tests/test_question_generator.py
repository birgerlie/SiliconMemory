"""Tests for question generator fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from silicon_memory.core.types import Belief
from silicon_memory.reflection.question_generator import Question, QuestionGenerator


@pytest.mark.asyncio
async def test_generate_questions_uses_heuristic_fallback_when_llm_empty() -> None:
    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(
        return_value=[
            Belief(id=uuid4(), content="A is likely connected to B", confidence=0.42, tags={"hypothesis"}),
            Belief(id=uuid4(), content="C may influence D", confidence=0.61, tags={"hypothesis"}),
        ],
    )

    memory = AsyncMock()
    memory._backend = backend
    memory.query_beliefs = AsyncMock(return_value=[])

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    questions = await qgen.generate_questions(max_questions=3)
    assert len(questions) >= 1
    assert questions[0].question.startswith("What evidence would confirm or refute:")


@pytest.mark.asyncio
async def test_generate_and_store_persists_heuristic_questions_to_working_memory() -> None:
    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(
        return_value=[
            Belief(id=uuid4(), content="X might lead to Y", confidence=0.49, tags={"hypothesis"}),
        ],
    )

    memory = AsyncMock()
    memory._backend = backend
    memory.query_beliefs = AsyncMock(return_value=[])
    memory.set_context = AsyncMock()
    memory.commit_belief = AsyncMock()

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    stored = await qgen.generate_and_store()
    assert stored >= 1
    memory.set_context.assert_awaited()


@pytest.mark.asyncio
async def test_generate_questions_falls_back_to_low_confidence_triples() -> None:
    class _Triple:
        external_id = "t/u/belief-1"
        subject = "A"
        predicate = "related_to"
        object_value = "B"
        probability = 0.41
        metadata = {
            "belief_id": "b1",
            "content": "A is likely related to B",
            "owner_id": "u",
            "tenant_id": "t",
            "privacy_level": "private",
        }

    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(return_value=[])
    backend._query_triples = lambda **kwargs: [_Triple()]  # noqa: ARG005, SLF001
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005, SLF001

    memory = AsyncMock()
    memory._backend = backend
    memory.query_beliefs = AsyncMock(return_value=[])

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    questions = await qgen.generate_questions(max_questions=3)
    assert len(questions) == 1
    assert "A is likely related to B" in questions[0].question


@pytest.mark.asyncio
async def test_generate_questions_uses_coverage_fallback_for_high_confidence_triples() -> None:
    class _Triple:
        external_id = "t/u/belief-2"
        subject = "Court"
        predicate = "issued"
        object_value = "ruling"
        probability = 0.92
        metadata = {
            "belief_id": "b2",
            "content": "Court issued ruling in case 22-1426",
            "owner_id": "u",
            "tenant_id": "t",
            "privacy_level": "private",
        }

    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(return_value=[])
    backend._query_triples = lambda **kwargs: [_Triple()]  # noqa: ARG005, SLF001
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005, SLF001

    memory = AsyncMock()
    memory._backend = backend
    memory.query_beliefs = AsyncMock(return_value=[])

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    questions = await qgen.generate_questions(max_questions=3)
    assert len(questions) == 1
    assert questions[0].priority == "low"
    assert questions[0].question.startswith("What independent source would corroborate:")


@pytest.mark.asyncio
async def test_generate_questions_uses_backend_query_when_router_signature_differs() -> None:
    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(
        return_value=[
            Belief(id=uuid4(), content="Potential contradiction in timeline", confidence=0.8, tags=set()),
        ],
    )
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(return_value=[])
    backend._query_triples = AsyncMock(return_value=[])

    memory = AsyncMock()
    memory._backend = backend
    memory.query_beliefs = AsyncMock(side_effect=TypeError("unexpected keyword argument"))

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    questions = await qgen.generate_questions(max_questions=2)
    assert len(questions) == 1
    assert "Potential contradiction in timeline" in questions[0].question


@pytest.mark.asyncio
async def test_generate_questions_retries_heuristic_before_llm() -> None:
    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(return_value=[])
    backend._query_triples = AsyncMock(return_value=[])

    memory = AsyncMock()
    memory._backend = backend

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._heuristic_questions = AsyncMock(  # noqa: SLF001
        side_effect=[[], [], [Question(question="Recovered question")]],
    )
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    questions = await qgen.generate_questions(max_questions=1)
    assert len(questions) == 1
    assert questions[0].question == "Recovered question"
    assert qgen._heuristic_questions.await_count >= 3  # noqa: SLF001
    qgen._llm_call.assert_not_awaited()  # noqa: SLF001


@pytest.mark.asyncio
async def test_generate_and_store_retries_heuristic_when_initial_generation_empty() -> None:
    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(return_value=[])
    backend._query_triples = AsyncMock(return_value=[])

    memory = AsyncMock()
    memory._backend = backend
    memory.set_context = AsyncMock()
    memory.commit_belief = AsyncMock()

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen.generate_questions = AsyncMock(return_value=[])  # noqa: SLF001
    qgen._heuristic_questions = AsyncMock(  # noqa: SLF001
        return_value=[Question(question="Late fallback question", priority="medium")],
    )

    stored = await qgen.generate_and_store()
    assert stored == 1
    memory.set_context.assert_awaited()


@pytest.mark.asyncio
async def test_generate_questions_falls_back_to_recent_experiences_when_signals_sparse() -> None:
    backend = AsyncMock()
    backend.get_uncertain_beliefs = AsyncMock(return_value=[])
    backend.query_beliefs = AsyncMock(return_value=[])
    backend.detect_triple_contradictions = AsyncMock(return_value=[])
    backend.get_beliefs_by_tag = AsyncMock(return_value=[])
    backend._query_triples = AsyncMock(return_value=[])
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005, SLF001

    memory = AsyncMock()
    memory._backend = backend
    memory.query_beliefs = AsyncMock(return_value=[])
    memory.get_recent_experiences = AsyncMock(
        return_value=[
            SimpleNamespace(content="A hearing was rescheduled to next week."),
        ],
    )

    qgen = QuestionGenerator(memory=memory, llm=AsyncMock())
    qgen._llm_call = AsyncMock(return_value='{"questions": []}')  # noqa: SLF001

    questions = await qgen.generate_questions(max_questions=3)
    assert len(questions) == 1
    assert questions[0].question.startswith("What external evidence can validate")
