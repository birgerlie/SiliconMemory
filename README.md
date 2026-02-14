# Silicon Memory

Multi-layered cognitive memory system for LLM augmentation with temporal awareness, belief tracking, and self-learning entity resolution.

## Overview

Silicon Memory provides a cognitive architecture for AI systems, implementing four types of memory:

- **Semantic Memory** - Facts and beliefs stored as triplets with confidence scores
- **Episodic Memory** - Experiences with temporal context and causal relationships
- **Procedural Memory** - How-to knowledge with step-by-step procedures
- **Working Memory** - Short-term context with TTL-based expiration

On top of the memory system:

- **Reflection Engine** - Processes experiences into patterns and beliefs ("dreaming")
- **Entity Resolution** - Self-learning entity detection with LLM-bootstrapped regex rules
- **Salience Profiles** - Context-aware retrieval weighting (debugging, architecture, planning)
- **REST + MCP Server** - HTTP API and Model Context Protocol for LLM clients

All storage is backed by [SiliconDB](https://github.com/birgerlie/silicondb), an Apple Silicon-native storage engine optimized for RAG workloads.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│     REST API (FastAPI)  │  MCP Server (stdio)  │  Python SDK     │
├──────────────────────────────────────────────────────────────────┤
│                    Server Layer                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │ Memory   │ │ Ingestion│ │Reflection│ │ Entity Resolution │   │
│  │ CRUD     │ │ Adapters │ │ Engine   │ │ (3-pass regex)    │   │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                     SiliconMemory Core                            │
│  ┌─────────┐ ┌─────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ │
│  │Semantic │ │Episodic │ │Procedural │ │Working  │ │ Graph   │ │
│  │(beliefs)│ │(events) │ │(how-to)   │ │(context)│ │(relations)│
│  └─────────┘ └─────────┘ └───────────┘ └─────────┘ └─────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                    LLM Provider (OpenAI v1 API)                  │
│          Auto-classification │ Entity Bootstrap │ Reflection     │
├──────────────────────────────────────────────────────────────────┤
│                     SiliconDB Storage Engine                     │
│   mmap + WAL │ Metal GPU │ Auto-embedding │ Graph │ Beliefs     │
└──────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install silicon-memory
```

With server dependencies:
```bash
pip install silicon-memory[server]
```

Requires SiliconDB to be installed and `SILICONDB_LIBRARY_PATH` environment variable set.

## Quick Start

### Python SDK

```python
from silicon_memory import SiliconMemory, RecallContext
from silicon_memory.core.types import Belief, Triplet, Source

async with SiliconMemory("/path/to/db") as memory:
    # Store a belief
    belief = Belief(
        id=uuid4(),
        triplet=Triplet("Python", "is", "programming language"),
        confidence=0.95,
        source=Source(id="docs", type="documentation", name="Python Docs"),
    )
    await memory.commit_belief(belief)

    # Recall relevant memories
    ctx = RecallContext(query="Python programming")
    response = await memory.recall(ctx)
```

### REST Server

```bash
# Start the server (requires SiliconServe on port 8000)
silicon-memory-server --mode rest --port 8420 --llm-model qwen3-80b

# Store a memory (auto-classified by LLM)
curl -X POST localhost:8420/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Python uses dynamic typing and garbage collection"}'

# Recall memories
curl -X POST localhost:8420/api/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "Python memory management"}'

# Bootstrap entity resolver from sample documents
curl -X POST localhost:8420/api/v1/entities/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"text": "Arbeidsmiljøloven (aml.) § 10-4 regulerer arbeidstid..."}'

# Resolve entities in new text
curl -X POST localhost:8420/api/v1/entities/resolve \
  -H "Content-Type: application/json" \
  -d '{"text": "I henhold til aml. § 14-9 om midlertidig ansettelse"}'
```

## Server Modes

| Mode | Description |
|------|-------------|
| `rest` | REST API only (FastAPI on configurable port) |
| `mcp` | MCP server only (stdio, for LLM clients like Claude) |
| `full` | REST + background reflection worker |

```bash
# REST mode
silicon-memory-server --mode rest --port 8420

# MCP mode (for Claude Desktop, etc.)
silicon-memory-server --mode mcp

# Full mode with background reflection every 5 minutes
silicon-memory-server --mode full --port 8420 --reflect-interval 300
```

## Entity Resolution

Self-learning entity detection using a three-pass architecture:

1. **Detect** - Broad regex patterns scan full text for candidates (microseconds)
2. **Extract** - Precise regex patterns per candidate determine type and normalize
3. **Disambiguate** - Context embedding comparison for ambiguous matches

Rules are generated offline by the LLM from sample documents — no LLM calls at runtime.

```bash
# Bootstrap: feed sample documents, LLM generates regex rules
curl -X POST localhost:8420/api/v1/entities/bootstrap \
  -d '{"text": "Full text of your laws, court decisions, regulations..."}'

# Register manual aliases
curl -X POST localhost:8420/api/v1/entities/register \
  -d '{"alias": "aml.", "canonical_id": "arbeidsmiljøloven", "entity_type": "law"}'

# Inspect generated rules
curl localhost:8420/api/v1/entities/rules

# Trigger incremental learning from unresolved entities
curl -X POST localhost:8420/api/v1/entities/learn
```

## Reflection Engine

Processes unprocessed experiences into patterns and beliefs ("dreaming"):

```bash
# Trigger on-demand reflection
curl -X POST localhost:8420/api/v1/reflect \
  -d '{"max_experiences": 100, "auto_commit": true}'
```

In `full` mode, reflection runs automatically on a configurable interval.

## Salience Profiles

Context-aware retrieval weighting for different use cases:

```python
# Recall with a salience profile
ctx = RecallContext(
    query="database connection timeout",
    salience_profile="debugging",  # or "architecture", "planning"
)
response = await memory.recall(ctx)
```

Built-in profiles: `debugging`, `architecture`, `planning`, `learning`, `reviewing`.

## API Reference

See [API.md](API.md) for the complete REST API specification.

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/store` | Store a memory (auto-classified or explicit type) |
| `POST` | `/api/v1/recall` | Recall memories with salience-aware retrieval |
| `POST` | `/api/v1/query` | Query beliefs by semantic search |
| `GET` | `/api/v1/memory/{type}/{id}` | Get a specific memory item |
| `PUT` | `/api/v1/working/{key}` | Set working memory entry |
| `GET` | `/api/v1/working` | Get all working memory entries |
| `DELETE` | `/api/v1/working/{key}` | Delete working memory entry |
| `POST` | `/api/v1/decisions` | Store a decision record |
| `POST` | `/api/v1/decisions/search` | Search decision records |
| `POST` | `/api/v1/ingest` | Ingest content via adapters (meeting, chat, email, document) |
| `POST` | `/api/v1/reflect` | Trigger on-demand reflection cycle |
| `POST` | `/api/v1/entities/bootstrap` | Bootstrap entity rules from sample text |
| `POST` | `/api/v1/entities/register` | Register an alias mapping |
| `POST` | `/api/v1/entities/resolve` | Resolve entities in text |
| `POST` | `/api/v1/entities/learn` | Generate rules from unresolved queue |
| `GET` | `/api/v1/entities/rules` | List all detection/extraction rules |
| `POST` | `/api/v1/forget` | Delete memories by scope |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/status` | Detailed server status |

## MCP Tools

When running in MCP mode, these tools are available to LLM clients:

| Tool | Description |
|------|-------------|
| `memory_store` | Store a belief, experience, or procedure (auto-classified) |
| `memory_recall` | Recall relevant memories for a query |
| `memory_query` | Query beliefs by semantic search |
| `working_memory_set` | Set a working memory entry |
| `working_memory_get` | Get a working memory entry |
| `memory_reflect` | Trigger a reflection cycle |
| `memory_forget` | Delete memories by scope |
| `memory_ingest` | Ingest content from various sources |
| `decision_store` | Store a decision record |
| `decision_search` | Search decision records |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,server]"

# Run unit tests (no external dependencies)
pytest tests/test_entity_resolver.py tests/test_salience.py -v

# Run e2e tests (requires SiliconDB)
SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e_cognitive.py -v

# Run server e2e tests (requires SiliconDB + SiliconServe)
pytest tests/test_e2e_server.py -v

# Run entity resolver e2e tests (requires SiliconServe with qwen3-80b)
pytest tests/test_e2e_entity_resolver.py -v

# Type checking
mypy src/silicon_memory

# Linting
ruff check src/
```

## License

MIT
