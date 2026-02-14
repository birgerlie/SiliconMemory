# Silicon Memory

Cognitive memory system for LLM augmentation. Four memory types (semantic, episodic, procedural, working) with reflection, entity resolution, salience profiles, and graph queries. Backed by SiliconDB (Apple Silicon-native storage).

## Commands

```bash
# Run unit tests (no external deps)
.venv/bin/pytest tests/test_entity_resolver.py tests/test_salience.py tests/test_security.py -v

# Run e2e tests (requires SILICONDB_LIBRARY_PATH)
.venv/bin/pytest tests/test_e2e_cognitive.py -v

# Run entity resolver e2e (requires SiliconServe with model loaded)
.venv/bin/pytest tests/test_e2e_entity_resolver.py -v

# Run server e2e tests
.venv/bin/pytest tests/test_e2e_server.py -v

# Type check
.venv/bin/mypy src/silicon_memory

# Lint
.venv/bin/ruff check src/

# Start REST server (requires SiliconServe on :8000)
.venv/bin/silicon-memory-server --mode rest --port 8420 --llm-model qwen3-80b

# Start MCP server (stdio, for Claude Desktop)
.venv/bin/silicon-memory-server --mode mcp
```

## Project Structure

```
src/silicon_memory/
  core/types.py          # Belief, Experience, Procedure, Triplet, Source, RecallResult
  core/protocols.py      # SemanticMemory, EpisodicMemory, ProceduralMemory protocols
  core/exceptions.py     # MemoryError, BeliefConflictError, StorageError
  memory/silicondb_router.py  # SiliconMemory — main class, async context manager
  storage/silicondb_backend.py  # SiliconDBBackend — storage layer
  retrieval/salience.py  # SalienceProfile, PROFILES dict (debugging, architecture, planning)
  temporal/              # Clock, DecayConfig, TemporalValidator
  reflection/            # ReflectionEngine — experiences → patterns → beliefs
  graph/                 # GraphQuery, GraphQueryBuilder, EntityExplorer
  entities/              # EntityResolver — 3-pass regex + LLM bootstrap
  decision/              # DecisionBriefGenerator
  security/              # UserContext, PolicyEngine, ForgettingService, AuditLogger
  ingestion/             # 7 adapters: meeting, email, document, news, slack, teams, discord
  llm/                   # SiliconLLMProvider (OpenAI v1 compatible)
  clients/               # MemoryAugmentedClient (OpenAI, Anthropic wrappers)
  tools/                 # MemoryTool, QueryTool, DecisionTool (LLM-callable)
  snapshot/              # SnapshotService — context switch snapshots
  dspy/                  # DSPy integration modules
  server/
    config.py            # ServerConfig dataclass
    dependencies.py      # MemoryPool, get_memory, get_llm, resolve_user_context
    schemas.py           # Pydantic request/response models
    rest/app.py          # FastAPI factory (create_app)
    rest/routers/        # health, memory, working, decisions, ingestion, reflect, entities, security
    mcp/server.py        # FastMCP server with 10 tools
    workers.py           # ReflectionWorker (background)

tests/
  conftest.py            # Shared fixtures
  test_entity_resolver.py     # Unit: entity resolver (52 tests)
  test_salience.py            # Unit: salience profiles
  test_e2e_cognitive.py       # E2E: memory features (requires SiliconDB)
  test_e2e_entity_resolver.py # E2E: entity resolution (requires SiliconServe)
  test_e2e_server.py          # E2E: REST server
  test_ingestion.py           # Unit: ingestion adapters
  test_security.py            # Unit: security module

deps/silicondb/          # Git submodule — Swift storage engine with Python bindings
```

## Architecture

All operations are async. SiliconMemory is the main entry point (async context manager). Every operation requires a UserContext for multi-tenancy.

```
Client → SiliconMemory (router)
           ├── SiliconDBBackend (storage, embedding, graph)
           ├── ReflectionEngine (experiences → beliefs)
           ├── EntityResolver (3-pass: detect → extract → disambiguate)
           └── SalienceProfile (weighted retrieval)
```

### Memory Types

- **Belief** — Triplet (subject, predicate, object) + confidence + source + status (provisional → validated → contested → rejected)
- **Experience** — Event content + outcome + temporal context + causal parent chain
- **Procedure** — Name + trigger + steps + success/failure tracking
- **Working** — Key-value with TTL (default 300s)

### Key Patterns

- **Async context manager**: `async with SiliconMemory(path, user_context) as memory:`
- **Dependency injection**: FastAPI uses `get_memory(pool, user_ctx)`, user resolved from X-User-Id/X-Tenant-Id headers
- **Memory pool**: One SiliconMemory per (tenant_id, user_id) pair, cached in MemoryPool
- **Protocol-based adapters**: IngestionAdapter is `@runtime_checkable Protocol` with `source_type` property and `ingest()` method
- **External ID format**: `{tenant_id}/{user_id}/{type}-{uuid}` in SiliconDB
- **Salience weighting**: Retrieval blends vector + text + temporal + graph + entropy + confidence weights
- **Entity resolution**: Pass 1 broad regex detect, pass 2 precise regex extract, pass 3 embedding disambiguate (only if ambiguous). Rules generated offline by LLM from sample documents.

## REST API (19 endpoints, prefix /api/v1)

| Method | Path | Body | Response |
|--------|------|------|----------|
| POST | /store | `{content, type?, confidence?, tags?, subject?, predicate?, object?}` | `{id, type, stored}` |
| POST | /recall | `{query, max_facts?, salience_profile?}` | `{facts[], experiences[], procedures[], working_context}` |
| POST | /query | `{query, limit?, min_confidence?}` | `{beliefs[], count}` |
| GET | /memory/{type}/{id} | — | `{id, type, content, confidence}` |
| PUT | /working/{key} | `{value, ttl_seconds?}` | `{key, value}` |
| GET | /working | — | `{key: value, ...}` |
| DELETE | /working/{key} | — | `{deleted}` |
| POST | /decisions | `{title, description, assumptions?, alternatives?}` | `{id, title, status}` |
| POST | /decisions/search | `{query, limit?}` | `[{id, title, status}]` |
| POST | /ingest | `{source_type, content, metadata?}` | `{experiences_created, entities_resolved}` |
| POST | /reflect | `{max_experiences?, auto_commit?}` | `{experiences_processed, patterns_found, new_beliefs}` |
| POST | /entities/bootstrap | `{text}` | `{detectors_created, extractors_created, aliases_discovered}` |
| POST | /entities/register | `{alias, canonical_id, entity_type}` | `{alias, canonical_id, stored}` |
| POST | /entities/resolve | `{text}` | `{resolved[], unresolved[]}` |
| POST | /entities/learn | — | `{rules_created}` |
| GET | /entities/rules | — | `{detectors[], extractors[], total}` |
| POST | /forget | `{scope, entity_id?, session_id?, topics?, query?, reason?}` | `{deleted_count, scope, success}` |
| GET | /health | — | `{status, version, uptime_seconds}` |
| GET | /status | — | `{status, version, active_users, mode}` |

## MCP Tools (10 tools)

| Tool | Parameters | Purpose |
|------|-----------|---------|
| memory_recall | query, max_facts?, max_experiences?, min_confidence? | Recall across all memory types |
| memory_store | content, type?, confidence?, subject?, predicate?, object?, tags? | Store belief/experience/procedure (auto-classifies) |
| memory_get | type, id | Get specific memory by UUID |
| memory_query | query, limit?, min_confidence? | Semantic search over beliefs |
| what_do_you_know | query, min_confidence? | Knowledge proof with sources and contradictions |
| working_memory | action (get_all/get/set/delete), key?, value?, ttl_seconds? | Manage working memory |
| decision_store | title, description, assumptions?, tags? | Record decision |
| decision_recall | query, limit? | Search past decisions |
| context_switch | action (snapshot/resume/list), task_context? | Save/restore context |
| forget | scope, entity_id?, session_id?, topics?, query?, reason? | GDPR deletion |

## Config

- **ServerConfig**: host, port (8420), mode (full/rest/mcp), db_path, llm (LLMConfig), reflect_interval (1800s)
- **LLMConfig**: base_url (localhost:8000/v1), model (qwen3-4b), temperature (0.7), max_tokens (1024)
- **SiliconDB**: requires `SILICONDB_LIBRARY_PATH` env var pointing to compiled library
- **Python**: 3.12+, dependencies in pyproject.toml
- **Server entry point**: `silicon-memory-server` → `silicon_memory.server.__main__:main`

## Conventions

- All async, no sync wrappers
- Pydantic v2 for API schemas, dataclasses for internal types
- Ruff for linting (line-length 100, Python 3.12)
- MyPy strict mode
- pytest-asyncio with `asyncio_mode = "auto"`
- Test markers: `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.slow`
