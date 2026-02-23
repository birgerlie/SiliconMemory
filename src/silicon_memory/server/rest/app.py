"""FastAPI application factory."""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.llm.scheduler import LLMScheduler
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


def _as_float(value: object, default: float = 1.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _load_bootstrap_rules_json(path: Path) -> tuple[list[object], list[object]]:
    """Load detector/extractor rules from JSON.

    Supports the bootstrap schema used in the epstein workspace:
    {
      "detectors": [{"id","pattern","description",...}],
      "extractors": [{"id","pattern","entity_type","confidence",...}]
    }
    """
    import json

    from silicon_memory.entities.types import DetectorRule, ExtractorRule

    raw = json.loads(path.read_text(encoding="utf-8"))
    detectors: list[DetectorRule] = []
    extractors: list[ExtractorRule] = []

    for d in raw.get("detectors", []):
        rule_id = d.get("id")
        pattern = d.get("pattern")
        if not rule_id or not pattern:
            continue
        detectors.append(
            DetectorRule(
                id=str(rule_id),
                pattern=str(pattern),
                description=str(d.get("description") or rule_id),
            )
        )

    for e in raw.get("extractors", []):
        rule_id = e.get("id")
        pattern = e.get("pattern")
        entity_type = e.get("entity_type")
        if not rule_id or not pattern or not entity_type:
            continue
        extractors.append(
            ExtractorRule(
                id=str(rule_id),
                entity_type=str(entity_type),
                detector_ids=[str(x) for x in (e.get("detector_ids") or [])],
                pattern=str(pattern),
                normalize_template=str(e.get("normalize_template") or "{match}"),
                confidence=_as_float(e.get("confidence"), 1.0),
            )
        )

    return detectors, extractors


def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        app.state.config = config
        app.state.start_time = time.monotonic()
        app.state.llm = SiliconLLMProvider(config=config.llm)
        app.state.scheduler = LLMScheduler(
            app.state.llm,
            max_concurrency=config.llm.max_concurrency,
            max_queue_size=config.llm.max_queue_size,
            max_wait_seconds=config.llm.max_wait_seconds,
        )
        await app.state.scheduler.start()
        app.state.pool = MemoryPool(config, app.state.llm)
        app.state.reflection_count = 0
        app.state.last_reflection = None

        # Entity resolver with persistent rule store
        from silicon_memory.entities import EntityCache, EntityResolver, EntityRuleStore, RuleEngine

        entity_cache = EntityCache()
        entity_rules = RuleEngine()

        # Open persistent store (shares the same db_path as main DB)
        try:
            entity_store = EntityRuleStore(
                db_path=config.db_path,
                language=config.language,
                auto_embedder=config.auto_embedder,
                embedder_model=config.embedder_model,
            )
            # Load persisted rules and aliases into the in-memory engine/cache
            for d in entity_store.load_all_detectors():
                entity_rules.add_detector(d)
            for e in entity_store.load_all_extractors():
                entity_rules.add_extractor(e)
            for alias, canonical_id, entity_type in entity_store.load_all_aliases():
                entity_cache.store(alias, canonical_id, entity_type)
            logger.info(
                "Loaded persisted entity rules: %d detectors, %d extractors, %d aliases",
                len(entity_rules._detectors),
                len(entity_rules._extractors),
                entity_cache.size,
            )
        except Exception:
            logger.warning("Could not open entity rule store — rules will not persist", exc_info=True)
            entity_store = None

        app.state.entity_resolver = EntityResolver(
            cache=entity_cache,
            rules=entity_rules,
            store=entity_store,
            max_unresolved_queue=config.entity_unresolved_queue_max,
        )
        app.state.entity_rule_store = entity_store
        logger.info("Entity resolver initialized")

        # Optional one-shot bootstrap from rules JSON.
        bootstrap_path = config.entity_bootstrap_rules_json
        if bootstrap_path:
            try:
                detectors, extractors = _load_bootstrap_rules_json(bootstrap_path)
                existing_detectors = {d.id for d in app.state.entity_resolver.rules._detectors}
                existing_extractors = {e.id for e in app.state.entity_resolver.rules._extractors}
                added_detectors = 0
                added_extractors = 0
                invalid_patterns = 0

                for d in detectors:
                    if d.id in existing_detectors:
                        continue
                    try:
                        re.compile(d.pattern)
                        app.state.entity_resolver.add_detector(d)
                        existing_detectors.add(d.id)
                        added_detectors += 1
                    except re.error:
                        invalid_patterns += 1

                for e in extractors:
                    if e.id in existing_extractors:
                        continue
                    try:
                        re.compile(e.pattern)
                        app.state.entity_resolver.add_extractor(e)
                        existing_extractors.add(e.id)
                        added_extractors += 1
                    except re.error:
                        invalid_patterns += 1

                logger.info(
                    "Loaded bootstrap rules from %s: +%d detectors, +%d extractors, invalid=%d",
                    bootstrap_path,
                    added_detectors,
                    added_extractors,
                    invalid_patterns,
                )
            except Exception:
                logger.warning(
                    "Failed to load bootstrap rules JSON from %s", bootstrap_path, exc_info=True
                )

        # Enable SiliconDB event log and register percolator rules
        if config.use_event_stream:
            try:
                _first_instance = app.state.pool.active_instances()
                if _first_instance:
                    _db = _first_instance[0]._backend._db
                    _db.enable_event_log(capacity=100_000)
                    _db.create_event_rule(
                        name="reflection_trigger",
                        emit_event_type="reflection.trigger",
                        filter={"event_type": "ingest.batch.searchable"},
                        cooldown_ms=5000,
                        dedupe_window_ms=10000,
                    )
                    logger.info("SiliconDB event log enabled, percolator rules registered")
            except Exception:
                logger.debug("Event log/percolator setup skipped (no active instances or unsupported)", exc_info=True)

        # Start background workers if full mode
        if config.mode == "full":
            from silicon_memory.server.workers import ReflectionWorker

            worker = ReflectionWorker(app.state.pool, config, llm=app.state.scheduler)
            app.state.worker = worker
            await worker.start()
            logger.info("Background reflection worker started (interval=%ds)", config.reflect_interval)

        logger.info("Silicon Memory server started (mode=%s)", config.mode)
        yield

        # Shutdown
        if hasattr(app.state, "worker"):
            await app.state.worker.stop()
            logger.info("Background reflection worker stopped")

        await app.state.scheduler.shutdown()
        logger.info("LLM scheduler stopped")

        if getattr(app.state, "entity_rule_store", None):
            app.state.entity_rule_store.close()
            logger.info("Entity rule store closed")

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

    # Static files & SPA catch-all
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        index_html = static_dir / "index.html"

        @app.get("/", include_in_schema=False)
        async def spa_root() -> FileResponse:
            return FileResponse(str(index_html), media_type="text/html")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa_fallback(path: str) -> FileResponse:
            # Serve actual static files if they exist, otherwise SPA index
            candidate = static_dir / path
            if candidate.is_file() and static_dir in candidate.resolve().parents:
                return FileResponse(str(candidate))
            return FileResponse(str(index_html), media_type="text/html")

    return app
