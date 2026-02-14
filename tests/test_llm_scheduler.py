"""Unit tests for LLMScheduler — priority queue + concurrency control."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from silicon_memory.llm.scheduler import LLMScheduler, Priority


class MockLLMProvider:
    """Mock provider with configurable latency and call tracking."""

    def __init__(self, latency: float = 0.0, fail: bool = False) -> None:
        self.latency = latency
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self._active = 0
        self._peak_active = 0
        self._lock = asyncio.Lock()

    async def generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float | None = None
    ) -> str:
        async with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
        try:
            self.calls.append({"method": "generate", "prompt": prompt})
            if self.latency:
                await asyncio.sleep(self.latency)
            if self.fail:
                raise RuntimeError("LLM provider error")
            return f"response:{prompt}"
        finally:
            async with self._lock:
                self._active -= 1

    async def generate_structured(
        self, prompt: str, schema: type, max_tokens: int | None = None
    ) -> Any:
        async with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
        try:
            self.calls.append({"method": "generate_structured", "prompt": prompt})
            if self.latency:
                await asyncio.sleep(self.latency)
            if self.fail:
                raise RuntimeError("LLM provider error")
            return {"result": prompt}
        finally:
            async with self._lock:
                self._active -= 1

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        async with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
        try:
            self.calls.append({"method": "complete", "prompt": prompt, "system": system})
            if self.latency:
                await asyncio.sleep(self.latency)
            if self.fail:
                raise RuntimeError("LLM provider error")
            return f"complete:{prompt}"
        finally:
            async with self._lock:
                self._active -= 1


# --- Test 1: Priority ordering ---


def test_priority_ordering():
    """HIGH < NORMAL < LOW (lower value = higher priority for PriorityQueue)."""
    assert Priority.HIGH < Priority.NORMAL < Priority.LOW
    assert Priority.HIGH.value == 0
    assert Priority.NORMAL.value == 1
    assert Priority.LOW.value == 2


# --- Test 2: Single generate passthrough ---


@pytest.mark.asyncio
async def test_single_generate():
    """Basic generate passthrough works."""
    provider = MockLLMProvider()
    scheduler = LLMScheduler(provider, max_concurrency=2)
    await scheduler.start()
    try:
        result = await scheduler.generate("hello")
        assert result == "response:hello"
        assert len(provider.calls) == 1
        assert provider.calls[0]["method"] == "generate"
    finally:
        await scheduler.shutdown()


# --- Test 3: Single generate_structured passthrough ---


@pytest.mark.asyncio
async def test_single_generate_structured():
    """Structured output passthrough works."""
    provider = MockLLMProvider()
    scheduler = LLMScheduler(provider, max_concurrency=2)
    await scheduler.start()
    try:
        result = await scheduler.generate_structured("prompt", dict)
        assert result == {"result": "prompt"}
        assert provider.calls[0]["method"] == "generate_structured"
    finally:
        await scheduler.shutdown()


# --- Test 4: Single complete passthrough ---


@pytest.mark.asyncio
async def test_single_complete():
    """Bridge method passthrough works."""
    provider = MockLLMProvider()
    scheduler = LLMScheduler(provider, max_concurrency=2)
    await scheduler.start()
    try:
        result = await scheduler.complete("world", system="sys")
        assert result == "complete:world"
        assert provider.calls[0]["method"] == "complete"
        assert provider.calls[0]["system"] == "sys"
    finally:
        await scheduler.shutdown()


# --- Test 5: Concurrency limit ---


@pytest.mark.asyncio
async def test_concurrency_limit():
    """Fire 5 requests with max_concurrency=2, verify max 2 in-flight."""
    provider = MockLLMProvider(latency=0.05)
    scheduler = LLMScheduler(provider, max_concurrency=2)
    await scheduler.start()
    try:
        tasks = [scheduler.generate(f"req-{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert provider._peak_active <= 2
    finally:
        await scheduler.shutdown()


# --- Test 6: Priority ordering execution ---


@pytest.mark.asyncio
async def test_priority_ordering_execution():
    """Queue LOW then HIGH while busy; HIGH completes before LOW."""
    completion_order: list[str] = []
    provider = MockLLMProvider(latency=0.05)
    scheduler = LLMScheduler(provider, max_concurrency=1)
    await scheduler.start()
    try:
        # Fill the single worker slot
        blocker = scheduler.generate("blocker", priority=Priority.HIGH)

        # Give the blocker time to start processing
        await asyncio.sleep(0.01)

        # Now queue LOW first, then HIGH — HIGH should execute before LOW
        async def track(label: str, priority: Priority) -> None:
            await scheduler.generate(label, priority=priority)
            completion_order.append(label)

        low_task = asyncio.create_task(track("low", Priority.LOW))
        high_task = asyncio.create_task(track("high", Priority.HIGH))

        await blocker
        await asyncio.gather(high_task, low_task)

        assert completion_order.index("high") < completion_order.index("low")
    finally:
        await scheduler.shutdown()


# --- Test 7: FIFO within same priority ---


@pytest.mark.asyncio
async def test_fifo_within_priority():
    """Same-priority requests complete in submission order."""
    completion_order: list[str] = []
    provider = MockLLMProvider(latency=0.02)
    scheduler = LLMScheduler(provider, max_concurrency=1)
    await scheduler.start()
    try:
        # Fill the worker slot
        blocker = scheduler.generate("blocker", priority=Priority.HIGH)
        await asyncio.sleep(0.01)

        # Queue 3 NORMAL-priority requests in known order
        async def track(label: str) -> None:
            await scheduler.generate(label, priority=Priority.NORMAL)
            completion_order.append(label)

        t1 = asyncio.create_task(track("first"))
        await asyncio.sleep(0.001)
        t2 = asyncio.create_task(track("second"))
        await asyncio.sleep(0.001)
        t3 = asyncio.create_task(track("third"))

        await blocker
        await asyncio.gather(t1, t2, t3)

        assert completion_order == ["first", "second", "third"]
    finally:
        await scheduler.shutdown()


# --- Test 8: Error propagation ---


@pytest.mark.asyncio
async def test_error_propagation():
    """Provider exception reaches caller."""
    provider = MockLLMProvider(fail=True)
    scheduler = LLMScheduler(provider, max_concurrency=2)
    await scheduler.start()
    try:
        with pytest.raises(RuntimeError, match="LLM provider error"):
            await scheduler.generate("will-fail")
    finally:
        await scheduler.shutdown()


# --- Test 9: Queue full rejects ---


@pytest.mark.asyncio
async def test_queue_full_rejects():
    """max_queue_size=2, submit many → error on overflow."""
    provider = MockLLMProvider(latency=0.1)
    scheduler = LLMScheduler(provider, max_concurrency=1, max_queue_size=2)
    await scheduler.start()
    try:
        # Fill the worker slot
        _ = asyncio.create_task(scheduler.generate("slot-filler"))
        await asyncio.sleep(0.01)

        # Fill the queue (2 slots)
        _ = asyncio.create_task(scheduler.generate("q1"))
        _ = asyncio.create_task(scheduler.generate("q2"))
        await asyncio.sleep(0.01)

        # Next request should be rejected
        with pytest.raises(RuntimeError, match="queue full"):
            await scheduler.generate("overflow")
    finally:
        await scheduler.shutdown()


# --- Test 10: Stats tracking ---


@pytest.mark.asyncio
async def test_stats():
    """active_count, queued_count, total_completed track correctly."""
    provider = MockLLMProvider(latency=0.05)
    scheduler = LLMScheduler(provider, max_concurrency=1)
    await scheduler.start()
    try:
        # Before any requests
        s = scheduler.stats()
        assert s["active"] == 0
        assert s["queued"] == 0
        assert s["total_completed"] == 0
        assert s["total_errors"] == 0

        # Run a request
        await scheduler.generate("req")
        s = scheduler.stats()
        assert s["total_completed"] == 1

        # Run a failing request
        provider.fail = True
        with pytest.raises(RuntimeError):
            await scheduler.generate("fail")
        provider.fail = False

        s = scheduler.stats()
        assert s["total_completed"] == 1
        assert s["total_errors"] == 1
    finally:
        await scheduler.shutdown()


# --- Test 11: Shutdown drains in-flight, cancels queued ---


@pytest.mark.asyncio
async def test_shutdown_drains():
    """In-flight complete, queued cancelled."""
    provider = MockLLMProvider(latency=0.05)
    scheduler = LLMScheduler(provider, max_concurrency=1)
    await scheduler.start()

    # Start an in-flight request
    inflight = asyncio.create_task(scheduler.generate("inflight"))
    await asyncio.sleep(0.01)

    # Queue additional requests
    queued1 = asyncio.create_task(scheduler.generate("queued1"))
    queued2 = asyncio.create_task(scheduler.generate("queued2"))
    await asyncio.sleep(0.01)

    # Shutdown should let in-flight finish but cancel queued
    await scheduler.shutdown(timeout=2.0)

    # In-flight should have completed
    assert inflight.done()
    result = await inflight
    assert result == "response:inflight"

    # Queued should be cancelled
    assert queued1.done()
    assert queued2.done()
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await queued1
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await queued2


# --- Test 12: Default priority is NORMAL ---


@pytest.mark.asyncio
async def test_default_priority_normal():
    """Omitting priority kwarg uses NORMAL."""
    provider = MockLLMProvider()
    scheduler = LLMScheduler(provider, max_concurrency=2)
    await scheduler.start()
    try:
        # Just verify the request goes through at default priority
        result = await scheduler.generate("default-prio")
        assert result == "response:default-prio"

        result = await scheduler.complete("default-prio")
        assert result == "complete:default-prio"

        result = await scheduler.generate_structured("default-prio", dict)
        assert result == {"result": "default-prio"}
    finally:
        await scheduler.shutdown()


# --- Test 13: Starvation warning ---


@pytest.mark.asyncio
async def test_starvation_warning(caplog):
    """LOW request waiting > max_wait_seconds logs warning."""
    provider = MockLLMProvider(latency=0.05)
    scheduler = LLMScheduler(
        provider, max_concurrency=1, max_wait_seconds=0.01  # very short threshold
    )
    await scheduler.start()
    try:
        # Fill the worker slot
        blocker = asyncio.create_task(scheduler.generate("blocker"))
        await asyncio.sleep(0.01)

        # Queue a LOW request — it will wait while blocker runs
        with caplog.at_level(logging.WARNING, logger="silicon_memory.llm.scheduler"):
            result = await scheduler.generate("slow-low", priority=Priority.LOW)

        await blocker
        assert result == "response:slow-low"
        assert any("waited" in r.message.lower() for r in caplog.records)
    finally:
        await scheduler.shutdown()
