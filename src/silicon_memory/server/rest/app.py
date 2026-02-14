"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.server.config import ServerConfig
from silicon_memory.server.dependencies import MemoryPool
from silicon_memory.server.errors import EXCEPTION_HANDLERS
from silicon_memory.server.rest.middleware import RequestLoggingMiddleware
from silicon_memory.server.rest.routers import (
    decisions,
    entities,
    health,
    ingestion,
    memory,
    reflect,
    security,
    working,
)

logger = logging.getLogger(__name__)


def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        app.state.config = config
        app.state.start_time = time.monotonic()
        app.state.llm = SiliconLLMProvider(config=config.llm)
        app.state.pool = MemoryPool(config, app.state.llm)
        app.state.reflection_count = 0
        app.state.last_reflection = None

        # Entity resolver
        from silicon_memory.entities import EntityCache, EntityResolver, RuleEngine

        app.state.entity_resolver = EntityResolver(
            cache=EntityCache(),
            rules=RuleEngine(),
        )
        logger.info("Entity resolver initialized")

        # Start background workers if full mode
        if config.mode == "full":
            from silicon_memory.server.workers import ReflectionWorker

            worker = ReflectionWorker(app.state.pool, config)
            app.state.worker = worker
            await worker.start()
            logger.info("Background reflection worker started (interval=%ds)", config.reflect_interval)

        logger.info("Silicon Memory server started (mode=%s)", config.mode)
        yield

        # Shutdown
        if hasattr(app.state, "worker"):
            await app.state.worker.stop()
            logger.info("Background reflection worker stopped")

        app.state.pool.close_all()
        logger.info("Silicon Memory server stopped")

    app = FastAPI(
        title="Silicon Memory",
        description="Living knowledge network for teams and organizations",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Exception handlers
    for exc_class, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_class, handler)

    # Routers
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix, tags=["health"])
    app.include_router(memory.router, prefix=prefix, tags=["memory"])
    app.include_router(working.router, prefix=prefix, tags=["working"])
    app.include_router(decisions.router, prefix=prefix, tags=["decisions"])
    app.include_router(ingestion.router, prefix=prefix, tags=["ingestion"])
    app.include_router(reflect.router, prefix=prefix, tags=["reflection"])
    app.include_router(security.router, prefix=prefix, tags=["security"])
    app.include_router(entities.router, prefix=prefix, tags=["entities"])

    return app
