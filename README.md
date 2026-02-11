# Silicon Memory

Multi-layered memory system for LLM augmentation with temporal awareness and belief tracking.

## Overview

Silicon Memory provides a cognitive architecture for AI systems, implementing four types of memory:

- **Semantic Memory** - Facts and beliefs stored as triplets with confidence scores
- **Episodic Memory** - Experiences with temporal context and causal relationships
- **Procedural Memory** - How-to knowledge with step-by-step procedures
- **Working Memory** - Short-term context with TTL-based expiration

All storage is backed by [SiliconDB](https://github.com/birgerlie/silicondb), an Apple Silicon-native storage engine optimized for RAG workloads.

## Installation

```bash
pip install silicon-memory
```

Requires SiliconDB to be installed and `SILICONDB_LIBRARY_PATH` environment variable set.

## Quick Start

```python
from silicon_memory import SiliconMemory, RecallContext
from silicon_memory.core.types import Belief, Triplet, Source

# Initialize memory system
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

    # Ask "what do you know"
    proof = await memory.what_do_you_know("Python")
    print(proof.as_report())
```

## LLM Tool Integration

Silicon Memory provides tools for LLM function calling:

```python
from silicon_memory.tools import MemoryTool, QueryTool

# Create tools
memory_tool = MemoryTool(memory)
query_tool = QueryTool(memory)

# Get OpenAI function schema
schema = MemoryTool.get_openai_schema()

# Invoke tool actions
response = await memory_tool.invoke(
    "store_fact",
    subject="Django",
    predicate="is",
    object="web framework",
    confidence=0.9,
)
```

## Features

- **Temporal Awareness** - Confidence decay over time with configurable half-life
- **Belief Tracking** - Probabilistic confidence with Bayesian updates
- **Contradiction Detection** - Identifies conflicting beliefs
- **Knowledge Proofs** - "What do you know" queries with source citations
- **Graph Relationships** - Causal chains and entity relationships via SiliconDB

## Examples

See the `examples/` directory for complete working examples:

- **basic_usage.py** - Core memory operations: beliefs, experiences, procedures, working memory
- **llm_tools.py** - Using MemoryTool and QueryTool with LLM function calling
- **knowledge_base.py** - Building a knowledge base with contradiction detection
- **conversational_agent.py** - Building a memory-augmented conversational agent

Run examples:
```bash
SILICONDB_LIBRARY_PATH=/path/to/lib python examples/basic_usage.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run integration tests (requires SiliconDB)
SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_integration.py -v

# Run E2E tests (requires SiliconDB)
SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e.py -v

# Run all tests with real embeddings
USE_REAL_EMBEDDINGS=1 SILICONDB_LIBRARY_PATH=/path/to/lib pytest -v
```

## License

MIT
