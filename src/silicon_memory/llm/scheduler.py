"""LLM request scheduler — priority queue + concurrency control.

Wraps a SiliconLLMProvider to cap simultaneous in-flight LLM requests
and prioritize user-facing work over background batch operations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels. Lower value = higher priority (PriorityQueue convention)."""

    HIGH = 0  # User-facing, latency-sensitive (e.g. auto-classify)
    NORMAL = 1  # Interactive but tolerant (e.g. ingestion, decisions)
    LOW = 2  # Background/batch (e.g. entity bootstrap, snapshots)


@dataclass(order=True)
class _PendingRequest:
    """An enqueued LLM request awaiting execution."""

    priority: int
    sequence: int  # FIFO within same priority
    enqueued_at: float = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False, repr=False)
    coro_fn: Callable[[], Awaitable[Any]] = field(compare=False, repr=False)


class LLMScheduler:
    """Priority queue scheduler wrapping an LLM provider.

    N worker tasks (where N = max_concurrency) consume from an asyncio.PriorityQueue.
    Callers await a Future that resolves when their request completes.
    """

    def __init__(
        self,
        provider: Any,
        max_concurrency: int = 2,
        max_queue_size: int = 100,
        max_wait_seconds: float = 30.0,
    ) -> None:
        self._provider = provider
        self._max_concurrency = max_concurrency
        self._max_queue_size = max_queue_size
        self._max_wait_seconds = max_wait_seconds

        self._queue: asyncio.PriorityQueue[_PendingRequest] = asyncio.PriorityQueue()
        self._sequence = 0
        self._workers: list[asyncio.Task[None]] = []
        self._active_count = 0
        self._queued_count = 0
        self._total_completed = 0
        self._total_errors = 0
        self._total_wait_ms = 0.0
        self._running = False

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def queued_count(self) -> int:
        return self._queued_count

    def stats(self) -> dict[str, Any]:
        total = self._total_completed + self._total_errors
        avg_wait = self._total_wait_ms / total if total else 0.0
        return {
            "active": self._active_count,
            "queued": self._queued_count,
            "total_completed": self._total_completed,
            "total_errors": self._total_errors,
            "avg_wait_ms": round(avg_wait, 2),
        }

    async def start(self) -> None:
        """Spawn N worker tasks."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_concurrency):
            task = asyncio.create_task(self._worker_loop(i), name=f"llm-worker-{i}")
            self._workers.append(task)
        logger.info("LLM scheduler started (%d workers)", self._max_concurrency)

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Drain in-flight requests and cancel queued ones."""
        self._running = False

        # Cancel all queued (not yet started) requests
        cancelled = []
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                self._queued_count -= 1
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError("Scheduler shutting down")
                    )
                cancelled.append(req)
            except asyncio.QueueEmpty:
                break

        # Signal workers to exit by putting sentinel values
        for _ in self._workers:
            sentinel = _PendingRequest(
                priority=-1, sequence=-1, enqueued_at=0,
                future=asyncio.get_event_loop().create_future(),
                coro_fn=lambda: asyncio.sleep(0),
            )
            sentinel.future.cancel()
            await self._queue.put(sentinel)

        # Wait for workers to finish current work
        if self._workers:
            done, pending = await asyncio.wait(self._workers, timeout=timeout)
            for task in pending:
                task.cancel()

        self._workers.clear()
        logger.info("LLM scheduler shut down")

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop: pull from queue, execute, resolve future."""
        while self._running or not self._queue.empty():
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            # Sentinel check
            if req.priority == -1:
                break

            if req.future.done():
                self._queued_count -= 1
                continue

            self._queued_count -= 1
            self._active_count += 1

            wait_seconds = time.monotonic() - req.enqueued_at
            wait_ms = wait_seconds * 1000
            self._total_wait_ms += wait_ms

            if wait_seconds > self._max_wait_seconds:
                logger.warning(
                    "Request waited %.1fs (priority=%s, threshold=%.1fs)",
                    wait_seconds,
                    Priority(req.priority).name,
                    self._max_wait_seconds,
                )

            try:
                result = await req.coro_fn()
                if not req.future.done():
                    req.future.set_result(result)
                self._total_completed += 1
            except Exception as exc:
                if not req.future.done():
                    req.future.set_exception(exc)
                self._total_errors += 1
            finally:
                self._active_count -= 1

    async def _enqueue(
        self,
        coro_fn: Callable[[], Awaitable[Any]],
        priority: Priority,
    ) -> Any:
        """Enqueue a request and wait for its result."""
        if self._queued_count >= self._max_queue_size:
            raise RuntimeError(
                f"LLM scheduler queue full ({self._max_queue_size})"
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._sequence += 1

        req = _PendingRequest(
            priority=int(priority),
            sequence=self._sequence,
            enqueued_at=time.monotonic(),
            future=future,
            coro_fn=coro_fn,
        )

        await self._queue.put(req)
        self._queued_count += 1

        return await future

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
        *,
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """Schedule a generate request."""
        result: str = await self._enqueue(
            lambda: self._provider.generate(prompt, max_tokens, temperature),
            priority,
        )
        return result

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        max_tokens: int | None = None,
        *,
        priority: Priority = Priority.NORMAL,
    ) -> Any:
        """Schedule a generate_structured request."""
        return await self._enqueue(
            lambda: self._provider.generate_structured(prompt, schema, max_tokens),
            priority,
        )

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        *,
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """Schedule a complete request."""
        result: str = await self._enqueue(
            lambda: self._provider.complete(prompt, system, temperature, max_tokens),
            priority,
        )
        return result
