"""Tests for conversation lifecycle scheduling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from silicon_memory.clients.base import (
    ChatMessage,
    ChatResponse,
    MemoryAugmentedClient,
    MemoryClientConfig,
)
from silicon_memory.orchestration import ConversationScheduler, ConversationSchedulerConfig
from silicon_memory.orchestration import SchedulerEventType


class _DummyClient(MemoryAugmentedClient):
    async def _call_api(self, messages: list[ChatMessage], **kwargs):  # noqa: ARG002
        return ChatResponse(content="assistant-reply")

    async def _stream_api(self, messages: list[ChatMessage], **kwargs):  # noqa: ARG002
        yield "assistant-reply"


def _mock_memory():
    user_ctx = SimpleNamespace(user_id="u1", tenant_id="t1", session_id="s1")
    snapshot = SimpleNamespace(
        id=uuid4(),
        summary="session summary",
        next_steps=["next"],
        open_questions=["why?"],
    )
    memory = SimpleNamespace(
        user_context=user_ctx,
        set_context=AsyncMock(),
        create_snapshot=AsyncMock(return_value=snapshot),
        record_experience=AsyncMock(),
        recall=AsyncMock(
            return_value=SimpleNamespace(
                facts=[],
                experiences=[],
                procedures=[],
                working_context={},
            )
        ),
    )
    return memory


@pytest.mark.asyncio
async def test_scheduler_runs_active_reflection_on_cadence():
    memory = _mock_memory()
    llm = object()
    config = ConversationSchedulerConfig(
        active_reflect_every_turns=2,
        active_reflect_min_interval_seconds=0,
    )
    scheduler = ConversationScheduler(memory=memory, llm=llm, config=config)

    with patch("silicon_memory.orchestration.conversation.ReflectionEngine") as engine_cls:
        engine = MagicMock()
        engine.reflect = AsyncMock(return_value=SimpleNamespace(experiences_processed=7))
        engine_cls.return_value = engine

        tick1 = await scheduler.on_interaction("session-a", "hello", "hi")
        tick2 = await scheduler.on_interaction("session-a", "status?", "ok")

    assert tick1.reflected is False
    assert tick2.reflected is True
    assert tick2.reflection_experiences_processed == 7
    assert memory.set_context.await_count == 2
    engine.reflect.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_conversation_end_runs_snapshot_reflect_and_dream():
    memory = _mock_memory()
    llm = object()
    config = ConversationSchedulerConfig(
        active_reflect_every_turns=100,
        active_reflect_min_interval_seconds=0,
        run_end_dream=True,
        end_dream_min_turns=1,
    )
    scheduler = ConversationScheduler(memory=memory, llm=llm, config=config)

    with patch("silicon_memory.orchestration.conversation.ReflectionEngine") as engine_cls:
        engine = MagicMock()
        engine.reflect = AsyncMock(return_value=SimpleNamespace(experiences_processed=4))
        engine.dream = AsyncMock(return_value={"hypotheses_generated": 2})
        engine_cls.return_value = engine

        await scheduler.on_interaction("session-b", "start", "ack")
        result = await scheduler.on_conversation_end("session-b", task_context="chat/task-b")

    assert result.snapshot_created is True
    assert result.reflected is True
    assert result.dreamed is True
    assert result.task_context == "chat/task-b"
    assert result.dream_stats.get("hypotheses_generated") == 2
    memory.create_snapshot.assert_awaited_once_with("chat/task-b", llm_provider=llm)
    assert memory.set_context.await_count >= 2


@pytest.mark.asyncio
async def test_scheduler_ingest_event_reflects_on_threshold_and_respects_guardrail():
    memory = _mock_memory()
    llm = object()
    config = ConversationSchedulerConfig(ingest_reflect_threshold=10)
    scheduler = ConversationScheduler(memory=memory, llm=llm, config=config)

    with patch("silicon_memory.orchestration.conversation.ReflectionEngine") as engine_cls:
        engine = MagicMock()
        engine.reflect = AsyncMock(return_value=SimpleNamespace(experiences_processed=11))
        engine_cls.return_value = engine

        outcome = await scheduler.emit_event(
            SchedulerEventType.INGEST_COMPLETED,
            session_id="session-c",
            payload={"new_extracted_count": 12},
        )
        blocked = await scheduler.emit_event(
            SchedulerEventType.INGEST_COMPLETED,
            session_id="session-c",
            payload={"new_extracted_count": 30, "backpressure": 0.95},
        )

    assert outcome.result["reflection_experiences_processed"] == 11
    assert blocked.decision.blocked is True
    assert "reflection_experiences_processed" not in blocked.result
    engine.reflect.assert_awaited_once()


@pytest.mark.asyncio
async def test_client_scheduler_hooks_and_end_conversation_fallback():
    memory = _mock_memory()

    scheduler = MagicMock()
    scheduler.on_interaction = AsyncMock()
    scheduler.on_conversation_end = AsyncMock(
        return_value=SimpleNamespace(
            to_dict=lambda: {
                "session_id": "s1",
                "turn_count": 1,
                "task_context": "task",
                "snapshot_created": True,
            }
        )
    )
    scheduler.emit_event = AsyncMock(
        return_value=SimpleNamespace(
            to_dict=lambda: {
                "event": {"type": SchedulerEventType.INGEST_COMPLETED.value},
                "decision": {"actions": [], "reason": "mock", "blocked": False},
                "result": {},
                "errors": [],
            }
        )
    )

    cfg = MemoryClientConfig(
        auto_recall=False,
        enable_conversation_scheduler=True,
        conversation_scheduler=scheduler,
    )
    client = _DummyClient(memory=memory, config=cfg)
    await client.chat([{"role": "user", "content": "hello"}], use_memory=False)

    scheduler.on_interaction.assert_awaited_once()
    end_outcome = await client.end_conversation(task_context="task")
    assert end_outcome["snapshot_created"] is True
    scheduler.on_conversation_end.assert_awaited_once()
    event_outcome = await client.emit_scheduler_event(
        SchedulerEventType.INGEST_COMPLETED,
        payload={"new_extracted_count": 5},
    )
    assert event_outcome["event"]["type"] == SchedulerEventType.INGEST_COMPLETED.value

    cfg_no_sched = MemoryClientConfig(
        auto_recall=False,
        enable_conversation_scheduler=False,
    )
    client_no_sched = _DummyClient(memory=memory, config=cfg_no_sched)
    fallback = await client_no_sched.end_conversation(task_context="task-fallback")
    assert fallback["snapshot_created"] is True
    memory.create_snapshot.assert_awaited_with("task-fallback")
