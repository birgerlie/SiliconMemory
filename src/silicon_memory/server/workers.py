"""Background workers for reflection cycles (the 'dreaming' loop)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

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
    """

    def __init__(self, pool: MemoryPool, config: ServerConfig) -> None:
        self._pool = pool
        self._config = config
        self._task: asyncio.Task | None = None
        self._running = False
        self._cycle_count = 0
        self._last_run: datetime | None = None

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    async def start(self) -> None:
        """Start the background reflection loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the background reflection loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        """Main loop — run reflection cycles at the configured interval."""
        logger.info(
            "Reflection worker started (interval=%ds, max_experiences=%d)",
            self._config.reflect_interval,
            self._config.reflect_max_experiences,
        )
        while self._running:
            try:
                await asyncio.sleep(self._config.reflect_interval)
                if not self._running:
                    break
                await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in reflection cycle")
                # Continue running despite errors
                await asyncio.sleep(60)

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
                engine = ReflectionEngine(memory, reflection_config)
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
