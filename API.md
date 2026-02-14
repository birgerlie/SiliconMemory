# Silicon Memory REST API

Base URL: `http://localhost:8420/api/v1`

All endpoints accept and return JSON. Content-Type: `application/json`.

---

## Health & Status

### GET /health

Health check.

**Response** `200`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 123.4
}
```

### GET /status

Detailed server status including reflection metrics.

**Response** `200`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 123.4,
  "active_users": 2,
  "last_reflection": "2025-01-15T10:30:00Z",
  "reflection_count": 5,
  "mode": "full"
}
```

---

## Core Memory

### POST /store

Store a memory item. When `type` is `"auto"` (default), the LLM classifies the content as belief, experience, or procedure.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | string | *required* | The memory content |
| `type` | string | `"auto"` | `"belief"` \| `"experience"` \| `"procedure"` \| `"auto"` |
| `confidence` | float | `0.5` | Confidence score (0.0-1.0) |
| `tags` | string[] | `[]` | Tags for categorization |
| `metadata` | object | `{}` | Arbitrary metadata |
| `subject` | string? | `null` | Belief triplet subject |
| `predicate` | string? | `null` | Belief triplet predicate |
| `object` | string? | `null` | Belief triplet object |
| `outcome` | string? | `null` | Experience outcome |
| `session_id` | string? | `null` | Experience session ID |
| `name` | string? | `null` | Procedure name |
| `description` | string? | `null` | Procedure description |
| `trigger` | string? | `null` | Procedure trigger condition |
| `steps` | string[] | `[]` | Procedure steps |

```json
{
  "content": "Python uses dynamic typing and garbage collection",
  "type": "auto",
  "confidence": 0.9,
  "tags": ["python", "language-features"]
}
```

**Response** `200`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "belief",
  "stored": true
}
```

### POST /recall

Recall relevant memories across all memory types with salience-aware retrieval.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *required* | Search query |
| `max_facts` | int | `20` | Max semantic memory results |
| `max_experiences` | int | `10` | Max episodic memory results |
| `max_procedures` | int | `5` | Max procedural memory results |
| `min_confidence` | float | `0.3` | Minimum confidence threshold |
| `include_episodic` | bool | `true` | Include episodic memories |
| `include_procedural` | bool | `true` | Include procedural memories |
| `include_working` | bool | `true` | Include working memory context |
| `salience_profile` | string? | `null` | `"debugging"` \| `"architecture"` \| `"planning"` \| `"learning"` \| `"reviewing"` |

```json
{
  "query": "Python memory management",
  "salience_profile": "debugging",
  "max_facts": 10
}
```

**Response** `200`

```json
{
  "facts": [
    {
      "content": "Python uses reference counting and cyclic garbage collector",
      "confidence": 0.92,
      "memory_type": "belief",
      "relevance_score": 0.87,
      "belief_id": "550e8400-..."
    }
  ],
  "experiences": [
    {
      "content": "Debugged a memory leak caused by circular references in event handlers",
      "confidence": 0.85,
      "memory_type": "experience",
      "relevance_score": 0.73,
      "belief_id": null
    }
  ],
  "procedures": [
    {
      "content": "How to profile Python memory usage with tracemalloc",
      "confidence": 0.90,
      "memory_type": "procedure",
      "relevance_score": 0.65,
      "belief_id": null
    }
  ],
  "working_context": {
    "current_task": "optimize data pipeline"
  },
  "total_items": 3,
  "query": "Python memory management"
}
```

### POST /query

Query beliefs by semantic search with confidence filtering.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *required* | Semantic search query |
| `limit` | int | `10` | Max results |
| `min_confidence` | float | `0.0` | Minimum confidence |

```json
{
  "query": "programming languages",
  "limit": 5,
  "min_confidence": 0.5
}
```

**Response** `200`

```json
{
  "beliefs": [
    {
      "id": "550e8400-...",
      "content": "Python is a dynamically typed programming language",
      "confidence": 0.95,
      "status": "active",
      "tags": ["python"],
      "subject": "Python",
      "predicate": "is",
      "object": "programming language"
    }
  ],
  "query": "programming languages",
  "count": 1
}
```

### GET /memory/{memory_type}/{memory_id}

Get a specific memory item by type and ID.

**Path Parameters**

| Parameter | Description |
|-----------|-------------|
| `memory_type` | `"belief"` \| `"experience"` \| `"procedure"` \| `"decision"` |
| `memory_id` | UUID of the memory item |

**Response** `200`

```json
{
  "id": "550e8400-...",
  "type": "belief",
  "content": "Python uses dynamic typing",
  "confidence": 0.95,
  "metadata": {}
}
```

---

## Working Memory

Short-term context with TTL-based expiration.

### GET /working

Get all working memory entries.

**Response** `200`

```json
{
  "current_task": "optimize data pipeline",
  "debug_context": {"file": "main.py", "line": 42}
}
```

### PUT /working/{key}

Set or update a working memory entry.

**Path Parameters**

| Parameter | Description |
|-----------|-------------|
| `key` | Context key name |

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `value` | any | *required* | The value to store |
| `ttl_seconds` | int | `300` | Time-to-live in seconds |

```json
{
  "value": {"file": "main.py", "line": 42},
  "ttl_seconds": 600
}
```

**Response** `200`

```json
{
  "key": "debug_context",
  "value": {"file": "main.py", "line": 42}
}
```

### DELETE /working/{key}

Delete a working memory entry.

**Path Parameters**

| Parameter | Description |
|-----------|-------------|
| `key` | Context key to delete |

**Response** `200`

```json
{
  "deleted": true
}
```

---

## Decisions

### POST /decisions

Store a decision record with assumptions and alternatives.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | string | *required* | Decision title |
| `description` | string | *required* | Decision description |
| `assumptions` | AssumptionInput[] | `[]` | Beliefs the decision depends on |
| `alternatives` | AlternativeInput[] | `[]` | Considered alternatives |
| `tags` | string[] | `[]` | Tags |
| `metadata` | object | `{}` | Arbitrary metadata |

**AssumptionInput**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `belief_id` | string | *required* | UUID of the supporting belief |
| `description` | string | *required* | How this belief supports the decision |
| `confidence_at_decision` | float | *required* | Belief confidence when decision was made |
| `is_critical` | bool | `false` | If true, decision should be revisited if this belief changes |

**AlternativeInput**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | string | *required* | Alternative title |
| `description` | string | *required* | Alternative description |
| `rejection_reason` | string | `""` | Why this was not chosen |

```json
{
  "title": "Use PostgreSQL for user data",
  "description": "Selected PostgreSQL over MongoDB for structured user data",
  "assumptions": [
    {
      "belief_id": "550e8400-...",
      "description": "User data has well-defined schema",
      "confidence_at_decision": 0.9,
      "is_critical": true
    }
  ],
  "alternatives": [
    {
      "title": "MongoDB",
      "description": "Document store for flexible schema",
      "rejection_reason": "Schema is well-defined, relational queries needed"
    }
  ],
  "tags": ["database", "architecture"]
}
```

**Response** `200`

```json
{
  "id": "550e8400-...",
  "title": "Use PostgreSQL for user data",
  "description": "Selected PostgreSQL over MongoDB for structured user data",
  "status": "active",
  "decided_at": "2025-01-15T10:30:00Z",
  "tags": ["database", "architecture"]
}
```

### POST /decisions/search

Search decision records by semantic query.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *required* | Search query |
| `limit` | int | `10` | Max results |
| `min_confidence` | float | `0.0` | Minimum confidence |

```json
{
  "query": "database choice",
  "limit": 5
}
```

**Response** `200`

```json
[
  {
    "id": "550e8400-...",
    "title": "Use PostgreSQL for user data",
    "description": "Selected PostgreSQL over MongoDB",
    "status": "active",
    "decided_at": "2025-01-15T10:30:00Z",
    "outcome": null,
    "tags": ["database"]
  }
]
```

---

## Ingestion

### POST /ingest

Ingest content from various sources. The appropriate adapter is selected based on `source_type`.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_type` | string | *required* | `"meeting"` \| `"chat"` \| `"email"` \| `"document"` \| `"news"` |
| `content` | string | *required* | Raw content to ingest |
| `metadata` | object | `{}` | Source metadata (participants, subject, etc.) |

```json
{
  "source_type": "meeting",
  "content": "Meeting notes: Decided to migrate from REST to GraphQL...",
  "metadata": {
    "participants": ["alice", "bob"],
    "date": "2025-01-15"
  }
}
```

**Response** `200`

```json
{
  "experiences_created": 3,
  "entities_resolved": 5,
  "decisions_detected": 1,
  "action_items_detected": 2,
  "errors": [],
  "source_type": "meeting"
}
```

---

## Reflection

### POST /reflect

Trigger an on-demand reflection cycle. Processes unprocessed experiences into patterns and beliefs.

In `full` server mode, reflection also runs automatically on a configurable interval.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_experiences` | int | `100` | Max experiences to process per cycle |
| `auto_commit` | bool | `true` | Automatically commit generated beliefs |

```json
{
  "max_experiences": 50,
  "auto_commit": true
}
```

**Response** `200`

```json
{
  "experiences_processed": 42,
  "patterns_found": 8,
  "new_beliefs": 5,
  "updated_beliefs": 2,
  "contradictions": 1,
  "summary": "Processed 42 experiences. Found 8 patterns including repeated database timeout issues."
}
```

---

## Entity Resolution

Self-learning entity detection with LLM-bootstrapped regex rules. Three-pass architecture: detect (broad regex) -> extract (precise regex) -> disambiguate (context embedding).

### POST /entities/bootstrap

Bootstrap entity rules from a sample document using LLM. Feed representative documents to generate detection and extraction rules.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Sample document text to learn patterns from |

```json
{
  "text": "Arbeidsmiljoloven (aml.) regulerer arbeidsforhold. I henhold til aml. § 10-4..."
}
```

**Response** `200`

```json
{
  "detectors_created": 6,
  "extractors_created": 6,
  "aliases_discovered": 2
}
```

### POST /entities/register

Manually register an alias to canonical ID mapping.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alias` | string | *required* | Short form or alias |
| `canonical_id` | string | *required* | Canonical identifier |
| `entity_type` | string | *required* | Entity type (e.g. `"law"`, `"person"`, `"project"`) |

```json
{
  "alias": "aml.",
  "canonical_id": "arbeidsmiljoloven",
  "entity_type": "law"
}
```

**Response** `200`

```json
{
  "alias": "aml.",
  "canonical_id": "arbeidsmiljoloven",
  "stored": true
}
```

### POST /entities/resolve

Resolve entity references in text using loaded rules and cache.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to resolve entities in |

```json
{
  "text": "I henhold til aml. § 14-9 om midlertidig ansettelse"
}
```

**Response** `200`

```json
{
  "resolved": [
    {
      "text": "aml. § 14-9",
      "canonical_id": "aml. § 14-9",
      "entity_type": "law_section",
      "confidence": 1.0,
      "span": [18, 29],
      "rule_id": "ext_law_section"
    }
  ],
  "unresolved": []
}
```

### POST /entities/learn

Generate rules from accumulated unresolved entities using LLM.

**Request** None

**Response** `200`

```json
{
  "rules_created": 3
}
```

### GET /entities/rules

List all detector and extractor rules.

**Response** `200`

```json
{
  "detectors": [
    {
      "id": "det_law_ref",
      "rule_type": "detector",
      "pattern": "§\\s*\\d+[-\\d]*",
      "entity_type": null,
      "description": "Law section references",
      "confidence": null
    }
  ],
  "extractors": [
    {
      "id": "ext_law_section",
      "rule_type": "extractor",
      "pattern": "(?:aml|strl)\\.?\\s*§\\s*(\\d+-\\d+)",
      "entity_type": "law_section",
      "description": null,
      "confidence": 0.95
    }
  ],
  "total": 2
}
```

---

## Security & GDPR

### POST /forget

GDPR-compliant forgetting. Delete memories by scope.

**Request**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scope` | string | *required* | `"entity"` \| `"session"` \| `"topic"` \| `"query"` \| `"all"` |
| `entity_id` | string? | `null` | Required when scope is `"entity"` |
| `session_id` | string? | `null` | Required when scope is `"session"` |
| `topics` | string[] | `[]` | Required when scope is `"topic"` |
| `query` | string? | `null` | Required when scope is `"query"` |
| `reason` | string? | `null` | Audit trail reason |

```json
{
  "scope": "entity",
  "entity_id": "user-12345",
  "reason": "GDPR deletion request"
}
```

**Response** `200`

```json
{
  "deleted_count": 15,
  "scope": "entity",
  "success": true
}
```

---

## Error Responses

All endpoints return standard error responses:

**`400` Bad Request**

```json
{
  "error": "ValidationError",
  "detail": "Field 'query' is required"
}
```

**`404` Not Found**

```json
{
  "error": "NotFound",
  "detail": "Memory item not found"
}
```

**`503` Service Unavailable**

```json
{
  "error": "ServiceUnavailable",
  "detail": "Entity resolver not initialized"
}
```
