"""Background workers for reflection cycles (the 'dreaming' loop)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any

from silicon_memory.core.utils import utc_now
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.types import ReflectionConfig
from silicon_memory.server.config import ServerConfig
from silicon_memory.server.dependencies import MemoryPool

logger = logging.getLogger(__name__)


class ReflectionWorker:
    """Runs periodic reflection cycles for all active users/teams.

    This is what makes Silicon Memory a living system — it processes
    unprocessed experiences into beliefs, detects contradictions,
    and reviews decision assumptions in the background.

    When use_event_stream is enabled, reflection is triggered by
    SiliconDB events (via percolator rules) instead of a fixed timer.
    The timer is kept as a fallback when SSE disconnects.
    """

    def __init__(self, pool: MemoryPool, config: ServerConfig, llm: Any = None) -> None:
        self._pool = pool
        self._config = config
        self._llm = llm
        self._task: asyncio.Task[None] | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._running = False
        self._cycle_count = 0
        self._last_run: datetime | None = None
        self._event_stream_active = False

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def event_stream_active(self) -> bool:
        return self._event_stream_active

    async def start(self) -> None:
        """Start the background reflection loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the background reflection loop."""
        self._running = False
        if self._event_task:
            self._event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._event_task
            self._event_task = None
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        """Main loop — event-driven when available, timer fallback."""
        logger.info(
            "Reflection worker started (interval=%ds, max_experiences=%d)",
            self._config.reflect_interval,
            self._config.reflect_max_experiences,
        )

        # Try to start event-driven mode
        if self._config.use_event_stream:
            self._event_task = asyncio.create_task(self._event_loop())

        while self._running:
            try:
                await asyncio.sleep(self._config.reflect_interval)
                if not self._running:
                    break
                # Skip timer-triggered cycles when event stream is healthy
                if self._event_stream_active:
                    logger.debug("Skipping timer cycle — event stream is active")
                    continue
                await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in reflection cycle")
                await asyncio.sleep(60)

    async def _event_loop(self) -> None:
        """Event-driven reflection via SiliconDB SSE subscription."""
        while self._running:
            try:
                db = self._get_event_db()
                if db is None:
                    logger.debug("No SiliconDB client available for event stream")
                    await asyncio.sleep(30)
                    continue

                logger.info("Subscribing to reflection.trigger events")
                self._event_stream_active = True

                async for event in db.subscribe_events(
                    event_types=["reflection.trigger"],
                    from_latest=True,
                ):
                    if not self._running:
                        break
                    logger.info(
                        "Event-triggered reflection (seq=%s)",
                        event.get("sequence", "?"),
                    )
                    await self.run_cycle()

                    # Commit cursor for resumption after restart
                    seq = event.get("sequence")
                    if seq is not None:
                        try:
                            db.commit_checkpoint(
                                consumer_group="reflection",
                                sequence=seq,
                            )
                        except Exception:
                            logger.debug("Failed to commit reflection checkpoint")

            except asyncio.CancelledError:
                break
            except Exception:
                self._event_stream_active = False
                logger.warning(
                    "Event stream disconnected, falling back to timer "
                    "(reconnecting in 30s)",
                    exc_info=True,
                )
                await asyncio.sleep(30)

        self._event_stream_active = False

    def _get_event_db(self) -> Any:
        """Get a SiliconDB client from the first active instance."""
        instances = self._pool.active_instances()
        if not instances:
            return None
        backend = getattr(instances[0], "_backend", None)
        if backend is None:
            return None
        return getattr(backend, "_db", None)

    async def run_cycle(self) -> None:
        """Run one reflection cycle across all active memory instances."""
        instances = self._pool.active_instances()
        if not instances:
            logger.debug("No active memory instances, skipping reflection")
            return

        logger.info("Starting reflection cycle for %d instance(s)", len(instances))

        total_experiences = 0
        total_beliefs = 0
        total_contradictions = 0

        reflection_config = ReflectionConfig(
            max_experiences_per_batch=self._config.reflect_max_experiences,
            auto_commit_beliefs=self._config.reflect_auto_commit,
        )

        for memory in instances:
            try:
                engine = ReflectionEngine(
                    memory=memory,
                    llm=self._llm,
                    config=reflection_config,
                )
                result = await engine.reflect(auto_commit=self._config.reflect_auto_commit)

                total_experiences += result.experiences_processed
                total_beliefs += len(result.new_beliefs)
                total_contradictions += len(result.contradictions)

                if result.experiences_processed > 0:
                    logger.info(
                        "Reflected for %s/%s: %d experiences → %d beliefs, %d contradictions",
                        memory.user_context.tenant_id,
                        memory.user_context.user_id,
                        result.experiences_processed,
                        len(result.new_beliefs),
                        len(result.contradictions),
                    )
            except Exception:
                logger.exception(
                    "Reflection failed for %s/%s",
                    memory.user_context.tenant_id,
                    memory.user_context.user_id,
                )

        self._cycle_count += 1
        self._last_run = utc_now()

        logger.info(
            "Reflection cycle #%d complete: %d experiences, %d beliefs, %d contradictions",
            self._cycle_count,
            total_experiences,
            total_beliefs,
            total_contradictions,
        )
