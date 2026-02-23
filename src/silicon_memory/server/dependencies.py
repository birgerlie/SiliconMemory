"""Dependency injection: MemoryPool, UserContext resolution, server lifespan."""

from __future__ import annotations

import logging
from collections import OrderedDict

from fastapi import Depends, Header, Request

from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.llm.scheduler import LLMScheduler
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.security.types import UserContext
from silicon_memory.server.config import ServerConfig
from silicon_memory.server.errors import MemoryPoolError

logger = logging.getLogger(__name__)


class MemoryPool:
    """Manages SiliconMemory instances per user/tenant pair.

    Each unique (tenant_id, user_id) gets its own SiliconMemory instance
    sharing the same underlying database path and LLM provider.
    """

    def __init__(self, config: ServerConfig, llm: SiliconLLMProvider) -> None:
        self._config = config
        self._llm = llm
        self._instances: OrderedDict[tuple[str, str], SiliconMemory] = OrderedDict()

    def get(self, user_ctx: UserContext) -> SiliconMemory:
        """Get or create a SiliconMemory instance for the user context."""
        key = (user_ctx.tenant_id, user_ctx.user_id)
        if key in self._instances:
            self._instances.move_to_end(key)
            return self._instances[key]

        if len(self._instances) >= self._config.memory_pool_max_instances:
            evicted_key, evicted_memory = self._instances.popitem(last=False)
            try:
                evicted_memory.close()
            except Exception:
                logger.exception("Error closing evicted memory instance")
            logger.warning(
                "Evicted memory instance for %s/%s to enforce pool limit (%d)",
                evicted_key[0],
                evicted_key[1],
                self._config.memory_pool_max_instances,
            )

        if key not in self._instances:
            self._instances[key] = SiliconMemory(
                path=self._config.db_path,
                user_context=user_ctx,
                db_grpc_host=self._config.db_grpc_host,
                db_grpc_port=self._config.db_grpc_port,
                db_retry_attempts=self._config.db_retry_attempts,
                db_retry_base_ms=self._config.db_retry_base_ms,
                db_retry_max_ms=self._config.db_retry_max_ms,
                db_request_timeout_s=self._config.db_request_timeout_s,
                db_max_inflight_mutations=self._config.db_max_inflight_mutations,
                db_idempotency_ttl_s=self._config.db_idempotency_ttl_s,
                language=self._config.language,
                enable_graph=self._config.enable_graph,
                auto_embedder=self._config.auto_embedder,
                embedder_model=self._config.embedder_model,
                llm_provider=self._llm,
            )
            logger.info("Created memory instance for %s/%s", user_ctx.tenant_id, user_ctx.user_id)
        return self._instances[key]

    def active_instances(self) -> list[SiliconMemory]:
        """Return all active memory instances."""
        return list(self._instances.values())

    @property
    def llm_provider(self) -> SiliconLLMProvider:
        """Shared LLM provider used for memory instances."""
        return self._llm

    def close_all(self) -> None:
        """Close all memory instances."""
        for memory in self._instances.values():
            try:
                memory.close()
            except Exception:
                logger.exception("Error closing memory instance")
        self._instances.clear()


def resolve_user_context(
    x_user_id: str = Header(default="default"),
    x_tenant_id: str = Header(default="default"),
) -> UserContext:
    """Resolve UserContext from request headers."""
    return UserContext(user_id=x_user_id, tenant_id=x_tenant_id)


def get_pool(request: Request) -> MemoryPool:
    """Get the MemoryPool from app state."""
    pool: MemoryPool | None = request.app.state.pool
    if pool is None:
        raise MemoryPoolError("Memory pool not initialized")
    return pool


def get_llm(request: Request) -> SiliconLLMProvider:
    """Get the shared LLM provider from app state."""
    return request.app.state.llm


def get_scheduler(request: Request) -> LLMScheduler:
    """Get the shared LLM scheduler from app state."""
    return request.app.state.scheduler


def get_memory(
    pool: MemoryPool = Depends(get_pool),
    user_ctx: UserContext = Depends(resolve_user_context),
) -> SiliconMemory:
    """Get a SiliconMemory instance for the current request user."""
    return pool.get(user_ctx)
