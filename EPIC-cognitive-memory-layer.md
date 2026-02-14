# EPIC: Cognitive Memory Layer for Corporate Decision Support

## Epic Summary

Build a persistent, self-correcting memory layer for LLM applications that accumulates knowledge across sessions, tracks beliefs with uncertainty, detects contradictions, and provides evidence-based decision support for teams. This requires foundational changes to SiliconDB (edge embeddings, temporal/graph-aware scoring, belief snapshots) and application-layer features in silicon-memory (decision records, synthesis, passive ingestion).

## Business Context

### Problem Statement

Current LLM applications have no durable memory. Every conversation starts from zero. In corporate settings this means:

- **Institutional knowledge lives in people's heads.** When someone leaves, decisions, context, and rationale are lost.
- **Decisions get re-litigated.** Teams revisit the same debates because no one recorded why a choice was made or what assumptions informed it.
- **Context is scattered.** Knowledge lives across Slack, Docs, Jira, meeting notes, and email with no unified, queryable picture.
- **Assumptions are never revisited.** Decisions are made on beliefs that may have since been contradicted, but no system tracks this.

### Solution

A two-phase cognitive memory system:

1. **Session-based learning** — During conversations, extract concrete facts (triples), preferences, and context. Store as beliefs with confidence scores.
2. **Background consolidation ("dreaming")** — Offline process that deduplicates, verifies, detects contradictions, strengthens confirmed beliefs, and decays stale knowledge.

Built on SiliconDB's existing vector/text/graph hybrid search with Monte Carlo belief system, extended with edge embeddings for context-aware retrieval and temporal/graph-aware scoring for salience.

### Target Users

- Engineering teams needing persistent project context across LLM sessions
- Product/business teams needing decision support with evidence tracking
- Organizations needing auditable institutional knowledge management

### Success Metrics

- Retrieval precision: >80% of recalled beliefs rated as relevant by users in decision support queries
- Knowledge accuracy: <5% of established beliefs flagged as incorrect after dreaming consolidation
- Contradiction detection: >90% of synthetically introduced contradictions detected within one consolidation cycle
- Adoption: system used in >50% of team LLM sessions after onboarding

---

## Architecture Overview

```
                    Silicon-Memory (Application Layer)
┌─────────────────────────────────────────────────────────────┐
│  Decision Records  │  Decision Synthesis  │  Passive Ingest │
│  Context Snapshots │  Salience Retrieval  │  News/External  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    SiliconDB (Storage Layer)
┌──────────────────────────────┴──────────────────────────────┐
│  Edge Embeddings  │  Temporal Scoring  │  Belief Snapshots  │
│  Graph Proximity  │  Custom Scoring    │  Existing features │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Chain

```
SDB-1 (Temporal Scoring) ──────────────────────────┐
SDB-2 (Custom Scoring) ───────────────────────────┐│
SDB-3 (Graph Proximity) ─────────────────────────┐││
SDB-4 (Belief Snapshots) ───────────────────┐    │││
SDB-5 (Edge Embeddings) ──────────────────┐ │    │││
                                          │ │    │││
SM-1 (Decision Records) ─── needs SDB-4 ─┘ │    │││
SM-2 (Decision Synthesis) ── needs SM-1 ────┘    │││
SM-3 (Salience Retrieval) ── needs SDB-1,2,3 ───┘││
SM-4 (Context Snapshots) ── independent ──────────┘│
SM-5 (Passive Ingestion) ── independent ───────────┘
SM-6 (News Integration) ─── needs SM-5
```

---

## WORKSTREAM 1: SiliconDB Foundation

### SDB-1: Temporal Ranking Signal

#### Context (Why)

Currently, SiliconDB's hybrid search treats all documents equally regardless of age. A belief from a year ago and one from yesterday get the same ranking weight if they both match the query. For corporate memory, recency matters — recent decisions, fresh market data, and current project context should rank higher than stale information. Without this, retrieval quality degrades as the knowledge base grows because old, potentially outdated beliefs compete equally with current ones.

#### Goal (What)

Add a configurable temporal decay function as a continuous ranking signal in the hybrid search pipeline, so that more recent documents score higher than older ones, with the decay rate controllable per query.

#### SMART Objective

Implement a temporal decay scoring function integrated into hybrid search within SiliconDB, measurable by: (a) search results for identical queries return documents ordered with a recency bias when temporal weight > 0, (b) decay curve is configurable per query with at least exponential and linear options, (c) zero performance regression on existing benchmarks when temporal weight is 0.

#### Implementation Notes (How)

The temporal score for a document is computed at search time:

```swift
// Exponential decay
temporalScore = exp(-lambda * (now - document.updatedAt))
// where lambda = ln(2) / halfLife

// Linear decay
temporalScore = max(0, 1.0 - (now - document.updatedAt) / maxAge)
```

This score is integrated into the existing RRF fusion or the new custom scoring (SDB-2) as an additional signal. The score is computed lazily — only for documents that pass the initial candidate retrieval phase.

Key files to modify:
- `SiliconDB+Search.swift` — Add temporal scoring to the search pipeline
- `SearchWeights` — Extend with temporal weight and decay configuration
- `QueryExecutor` — Compute temporal scores during fusion

#### Acceptance Criteria

- [ ] `SearchWeights` extended with `temporalWeight: Float` (default 0.0 for backward compatibility) and `temporalDecay: TemporalDecayConfig`
- [ ] `TemporalDecayConfig` supports at minimum: `.exponential(halfLife: TimeInterval)` and `.linear(maxAge: TimeInterval)`
- [ ] When `temporalWeight > 0`, search results are influenced by document recency — verified by test: two documents with identical text/embeddings but different `updatedAt` return in recency order when temporal weight is dominant
- [ ] When `temporalWeight == 0`, search behaviour is identical to current implementation (no regression)
- [ ] Temporal scoring adds <1ms overhead per search (measured on 100K document corpus)
- [ ] Unit tests covering: exponential decay curve correctness, linear decay curve correctness, zero-weight passthrough, edge cases (future timestamps, zero age)
- [ ] Integration test: hybrid search with vector + text + temporal weights returns expected ranking
- [ ] Python bindings expose temporal configuration in `search()` call

---

### SDB-2: Custom Scoring / Extended Search Configuration

#### Context (Why)

The current hybrid search uses fixed Reciprocal Rank Fusion (RRF) with per-modality weights (vector, text, graph). The cognitive memory layer needs to combine additional signals — temporal recency (SDB-1), graph proximity (SDB-3), belief confidence, entropy — in ways that differ per use case. Decision support queries should weight confidence and evidence count highly. Context recall should weight recency and graph proximity. Knowledge exploration should weight entropy (finding uncertain beliefs). A single fixed scoring formula cannot serve all these needs.

#### Goal (What)

Replace or augment the current `SearchWeights` with a declarative, extensible scoring configuration that supports multiple signals with per-query weights, enabling silicon-memory to compose different retrieval strategies without SiliconDB code changes.

#### SMART Objective

Implement an extensible `SearchScoring` configuration in SiliconDB that supports at minimum 5 scoring signals (vector similarity, text relevance, temporal recency, belief confidence, graph proximity) with per-query weights, measurable by: (a) all existing search tests pass with equivalent configuration, (b) new scoring signals can be added without modifying the core search loop, (c) scoring configuration is expressible from both Swift and Python APIs.

#### Implementation Notes (How)

```swift
public struct SearchScoring: Sendable {
    /// Core retrieval signals (existing)
    public var vectorWeight: Float = 0.7
    public var textWeight: Float = 0.3

    /// Extended signals (new)
    public var temporalWeight: Float = 0.0
    public var temporalDecay: TemporalDecayConfig = .exponential(halfLife: 7 * 86400)

    public var confidenceWeight: Float = 0.0  // Belief probability as boost
    public var entropyWeight: Float = 0.0     // High entropy = uncertain = interesting
    public var entropyDirection: EntropyDirection = .preferLow  // or .preferHigh

    public var graphProximityWeight: Float = 0.0
    public var graphContextNodes: [GlobalDocumentId] = []  // Anchor nodes for proximity

    /// Fusion method
    public var fusion: FusionMethod = .rrf(k: 60)  // or .weighted
}
```

The `QueryExecutor` computes each signal's score for candidate documents, normalises them to [0, 1], and combines via the chosen fusion method. RRF remains the default for backward compatibility; weighted linear combination is the new option for fine-grained control.

#### Acceptance Criteria

- [ ] `SearchScoring` struct defined with all signal weights and configurations
- [ ] Backward compatible: `SearchWeights(vector: 0.7, text: 0.3)` produces identical results to current implementation
- [ ] At minimum supports: vector similarity, text relevance, temporal recency, belief confidence, graph proximity signals
- [ ] Two fusion methods available: `.rrf(k:)` (existing) and `.weighted` (new linear combination)
- [ ] Each signal's score is normalised to [0, 1] before fusion
- [ ] Zero-weighted signals are skipped entirely (no computation cost)
- [ ] Python bindings expose full `SearchScoring` configuration
- [ ] Integration test: same query with different scoring configs produces different result rankings
- [ ] Performance: <2ms overhead per search for scoring computation on 100K corpus (excluding graph proximity, which has its own budget)
- [ ] Documentation: inline doc comments on each field explaining what it does and valid ranges

---

### SDB-3: Graph Proximity Scoring

#### Context (Why)

When a user is working in a specific context (e.g., "project Alpha" or "customer onboarding"), beliefs and facts that are closer in the knowledge graph to that context are more relevant than distant ones. Currently, graph search is a separate modality — you provide seed nodes and the graph expands the candidate set. But graph distance isn't used as a continuous ranking signal. A belief that is 1 hop from the current context should rank higher than one that is 3 hops away, even if both are semantically similar to the query.

#### Goal (What)

Implement graph proximity as a continuous scoring signal in hybrid search, using Personalized PageRank (PPR) from context anchor nodes to score candidate documents by their graph distance to the current working context.

#### SMART Objective

Implement Personalized PageRank-based graph proximity scoring integrated into the hybrid search pipeline, measurable by: (a) documents closer in the graph to specified context nodes receive higher proximity scores, (b) PPR computation completes within 50ms for graphs up to 100K nodes, (c) proximity scores are available as a signal in `SearchScoring` (SDB-2).

#### Implementation Notes (How)

Personalized PageRank computes a relevance score for every node relative to a set of "seed" nodes. The algorithm is iterative:

```
For each iteration:
    For each node v:
        ppr[v] = dampingFactor * sum(ppr[u] / degree(u) for u in neighbors(v))
        if v in contextNodes: ppr[v] += (1 - dampingFactor) / |contextNodes|
```

This runs on the existing CSR graph structure. The Metal GPU implementation of PageRank already exists — it needs to be extended to support personalization (non-uniform teleportation vector).

The resulting PPR scores are used as the `graphProximityScore` for each candidate document in the search scoring pipeline. Scores are computed once per query (amortized across all candidates) and cached for the duration of the search.

Key files:
- `GraphStore.swift` — Add personalizedPageRank method
- `Metal/graph_kernels.metal` — Extend PageRank kernel with teleportation vector
- `QueryExecutor` — Integrate PPR scores into scoring pipeline

#### Acceptance Criteria

- [ ] `GraphStore` exposes `personalizedPageRank(contextNodes: [GlobalDocumentId], dampingFactor: Float, iterations: Int) -> [GlobalDocumentId: Float]`
- [ ] PPR scores are higher for nodes closer to context nodes — verified by test with known graph topology
- [ ] Metal GPU implementation for PPR (extends existing PageRank kernel with teleportation vector)
- [ ] CPU fallback when Metal is unavailable
- [ ] PPR computation completes in <50ms for 100K node graph with 10 iterations
- [ ] PPR scores integrated into `SearchScoring.graphProximityWeight` signal
- [ ] When `graphProximityWeight == 0` or `graphContextNodes` is empty, no PPR computation occurs
- [ ] Unit tests: known graph topology produces expected PPR distribution
- [ ] Integration test: search with graph proximity context returns closer documents ranked higher than distant ones with similar semantic scores
- [ ] Python bindings expose `graph_context_nodes` parameter in search

---

### SDB-4: Belief Snapshots / Point-in-Time Queries

#### Context (Why)

For decision records (SM-1), the system needs to capture "what did we believe at the time this decision was made?" Currently, beliefs are mutable — when confidence updates, the previous value is overwritten. If a decision was made when belief confidence was 0.8 and it's now 0.3, that's critical information for evaluating whether the decision's assumptions still hold. Without belief history, the system can only show current state, not the state that actually informed the decision.

The WAL already contains the full history of every belief change, but this history isn't queryable.

#### Goal (What)

Add a belief history mechanism that records every belief state change with a timestamp, enabling point-in-time queries ("what was the confidence of belief X on date Y?") and belief snapshots ("capture current state of these beliefs for a decision record").

#### SMART Objective

Implement belief history tracking and point-in-time queries in SiliconDB, measurable by: (a) every belief probability change is recorded with a timestamp, (b) `getBeliefsAsOf(date)` returns the correct belief state for any past date, (c) `snapshotBeliefs(ids)` returns an immutable copy of current belief states, (d) storage overhead is bounded — history entries are compact and purgeable.

#### Implementation Notes (How)

Two approaches, choose one:

**Option A: Dedicated History Store (Recommended)**

A lightweight append-only store for belief state changes:

```swift
struct BeliefHistoryEntry: Sendable {
    let beliefId: GlobalDocumentId
    let probability: Float
    let observations: Int
    let confirmations: Int
    let contradictions: Int
    let timestamp: UInt64  // microseconds since epoch
}
// ~32 bytes per entry
```

Stored in a memory-mapped, append-only file. Indexed by beliefId for fast lookups. Typical usage: a belief changes 10-50 times over its lifetime, so 100K beliefs = 1-5M history entries = 32-160MB.

**Option B: WAL Replay**

Reconstruct history from existing WAL entries. Cheaper to implement but expensive to query (requires scanning WAL). Only viable if queries are rare.

Recommend Option A for query performance.

Key files:
- New: `BeliefHistory.swift` — Append-only history store
- `SiliconDB+Beliefs.swift` — Hook into belief update path to record history
- New: `SiliconDB+BeliefQueries.swift` — Point-in-time query API

#### Acceptance Criteria

- [ ] Every call to `updateProbabilities()`, `propagate()`, or direct belief confidence update records a history entry with timestamp
- [ ] `getBeliefHistory(id: GlobalDocumentId) -> [BeliefHistoryEntry]` returns full history for a belief, ordered by timestamp
- [ ] `getBeliefAsOf(id: GlobalDocumentId, date: Date) -> BeliefHistoryEntry?` returns the belief state at a specific point in time (latest entry before the given date)
- [ ] `snapshotBeliefs(ids: [GlobalDocumentId]) -> BeliefSnapshot` returns an immutable copy of current states with a snapshot timestamp and ID
- [ ] `getSnapshot(id: SnapshotId) -> BeliefSnapshot` retrieves a previously created snapshot
- [ ] History entries are compact: <40 bytes per entry
- [ ] History store is append-only and crash-safe (fsync after write or WAL-protected)
- [ ] History can be purged: `purgeHistory(olderThan: Date)` removes entries before a date
- [ ] Storage overhead measured and documented for: 1K, 10K, 100K beliefs with typical update frequencies
- [ ] Python bindings expose: `get_belief_history()`, `get_belief_as_of()`, `snapshot_beliefs()`, `get_snapshot()`
- [ ] Unit tests: create belief, update 5 times, query at each historical point, verify correct values returned
- [ ] Integration test: snapshot beliefs, update them, verify snapshot still returns original values

---

### SDB-5: Edge Embeddings — FERDIG

> **Status: COMPLETE.** Implementert med separat edge-HNSW-indeks (syntetisk tenant 0xED6E), EdgeMetadataStore med ReadWriteLock, WAL-persistering (opcode 15), og Python-bindinger via GraphMixin (`add_edge_with_description`, `search_edges`). Bruker samme embedding-dimensjon og Metal distance-kerneler som dokument-HNSW — ingen separat konfigurasjon nødvendig.

#### Context (Why) — Original

Currently, graph edges are typed pointers with a weight (29 bytes). They carry no semantic content. This means you cannot search for relationships by meaning — only by type. In a corporate memory system, context scoping is critical: "uses Redis" in project Alpha is a different relationship than "evaluated Redis" in project Beta, but both have `type: .custom("uses")` pointing at the same Redis node.

Edge embeddings solve this by making relationships searchable by semantic similarity. The edge description "uses Redis for session caching, chosen for sub-ms latency" becomes a vector that can be compared to queries like "caching performance issues." This enables analogical retrieval (finding similar experiences across different projects) and context scoping through embedding geometry rather than brittle metadata rules.

SiliconDB already has the infrastructure to make this cheap: the on-device Metal GPU embedding pipeline (`LingerBatchEmbedder` + `EmbeddingWorker`) can embed edge descriptions alongside document text with minimal overhead, and the unified memory architecture means no data copying between CPU and GPU.

#### Goal (What)

Extend SiliconDB edges to support optional text descriptions that are auto-embedded via the existing GPU pipeline, stored in a dedicated edge HNSW index, and searchable via semantic similarity.

#### SMART Objective

Implement edge embeddings in SiliconDB, measurable by: (a) edges can be created with text descriptions that are auto-embedded, (b) `searchEdges()` returns edges ranked by semantic similarity to a query, (c) edge embeddings use the existing `EmbeddingWorker` pipeline with no separate infrastructure, (d) edge embedding adds <20% overhead to the `addEdge` path for edges with descriptions.

#### Implementation Notes (How)

**Storage Architecture:**

The CSR remains unchanged for fast graph traversal. Edge embeddings are stored separately:

```
CSR (existing):        Fast traversal, adjacency queries
EdgeMetadataStore:     edge_id → description text + metadata
EdgeHNSW:              Separate HNSW index for edge embeddings
```

Edge IDs link all three. The CSR is never bloated by embedding data.

**Embedding Pipeline:**

When `addEdge` is called with a description:
1. Edge is added to CSR immediately (existing path, unchanged)
2. Description is stored in EdgeMetadataStore
3. Description is submitted to `EmbeddingWorker` (same pipeline as document embeddings)
4. Worker batches it with other pending embeddings
5. Result inserted into EdgeHNSW (separate index from document HNSW)
6. Edge is immediately searchable via Hot Vector Store while waiting for HNSW insertion

**Embedding Dimension:**

Consider supporting a smaller embedding dimension for edges (e.g., 384 vs 768 for documents). Edge descriptions are typically shorter and carry less information. This halves storage and speeds up search. The EdgeHNSW would be configured independently from the document HNSW.

**Edge Description Construction:**

For best embedding quality, concatenate context into the description:

```
embed("{source_text} | {edge_description} | {target_text}")
// e.g., "birger | uses for session caching | Redis"
```

This gives the embedding model enough context to differentiate similar relationships.

Key files:
- New: `EdgeMetadataStore.swift` — Description + metadata storage for edges
- New: `EdgeHNSW` — Separate HNSW instance for edge embeddings
- `GraphStore.swift` — Extended `addEdge` signature with optional description
- `EmbeddingWorker.swift` — Accept edge embedding requests (same pipeline, different target index)
- `SiliconDB+Search.swift` — New `searchEdges()` method
- `SiliconDB+Graph.swift` — Updated `addEdge` API

#### Acceptance Criteria

- [ ] `addEdge()` accepts optional `description: String` parameter
- [ ] When description is provided, it is auto-embedded via the existing `EmbeddingWorker` pipeline
- [ ] Edge embeddings stored in a separate HNSW index (not mixed with document embeddings)
- [ ] `searchEdges(query: String, embedding: [Float]?, k: Int, filter: EdgeFilter?) -> [EdgeSearchResult]` returns edges ranked by semantic similarity
- [ ] `EdgeSearchResult` contains: source, target, edge type, weight, description, similarity score
- [ ] Edges without descriptions are not in the edge HNSW (backward compatible — existing edges unaffected)
- [ ] Edge embedding uses the same `LingerBatchEmbedder` batching as document embeddings
- [ ] Hot Vector Store pattern applied: edge embeddings are immediately searchable via brute-force while waiting for HNSW insertion
- [ ] Edge HNSW dimension is independently configurable (can differ from document HNSW)
- [ ] Performance: `addEdge` with description adds <5ms overhead beyond the embedding computation itself
- [ ] Performance: `searchEdges` with k=10 completes in <20ms on 100K edge index
- [ ] Storage: edge metadata store is memory-mapped and crash-safe
- [ ] WAL: edge descriptions are included in WAL entries for crash recovery
- [ ] Python bindings expose: `add_edge(description=...)`, `search_edges(query, k)`
- [ ] Unit tests: add edges with descriptions, search by similarity, verify ranking
- [ ] Integration test: two edges with same type but different descriptions are distinguishable by semantic search
- [ ] Integration test: edge search + document search return complementary results for the same query

---

## WORKSTREAM 2: Silicon-Memory Application Layer

### SM-1: Decision Records

#### Context (Why)

The core value proposition for corporate teams is not just remembering facts — it's tracking why decisions were made, what was believed at the time, what alternatives were considered, and whether the underlying assumptions still hold. Currently, silicon-memory stores beliefs and experiences independently with no explicit decision object linking them. A team lead cannot ask "why did we choose Kubernetes?" and get back the beliefs, alternatives, and assumptions that informed that choice.

This is the foundation of the decision support use case. Without decision records, the system is a memory store. With them, it becomes a decision intelligence tool.

#### Goal (What)

Implement a `Decision` memory type in silicon-memory that captures the decision, the beliefs that informed it, alternatives considered, assumptions made, and tracks over time whether assumptions held.

#### SMART Objective

Implement decision records in silicon-memory with full lifecycle tracking, measurable by: (a) decisions can be stored with linked beliefs, alternatives, and assumptions, (b) the system can report which decision assumptions have since changed, (c) decisions are retrievable by semantic query with evidence chains, (d) the dreaming phase automatically flags decisions based on revised assumptions.

#### Implementation Notes (How)

**Core Data Model:**

```python
@dataclass
class Decision:
    id: UUID
    title: str                          # "Use Kubernetes for deployment"
    description: str                    # Full context and rationale
    decided_at: datetime
    decided_by: str                     # User or team
    session_id: str                     # Session where decision was made

    # Linked knowledge (at time of decision)
    belief_snapshot_id: str             # SiliconDB belief snapshot reference
    assumptions: list[Assumption]       # Beliefs critical to this decision
    alternatives: list[Alternative]     # Options that were considered

    # Tracking
    status: DecisionStatus              # ACTIVE, REVISIT_SUGGESTED, REVISED, SUPERSEDED
    outcome: str | None                 # What actually happened
    outcome_recorded_at: datetime | None
    revision_of: UUID | None            # If this revises a previous decision

    # Storage
    node_type: str = "decision"

@dataclass
class Assumption:
    belief_id: UUID                     # Reference to the belief
    description: str                    # "We expect 10x traffic growth"
    confidence_at_decision: float       # Confidence when decision was made
    is_critical: bool                   # Would the decision change if this is wrong?

@dataclass
class Alternative:
    title: str                          # "Use ECS instead"
    description: str                    # Why it was considered
    rejection_reason: str               # Why it was rejected
    beliefs_supporting: list[UUID]      # Evidence for this option
    beliefs_against: list[UUID]         # Evidence against this option
```

**Dreaming Integration:**

During consolidation, the reflection engine checks active decisions:

```
For each ACTIVE decision:
    For each critical assumption:
        current_confidence = get_current_belief_confidence(assumption.belief_id)
        if |current_confidence - assumption.confidence_at_decision| > threshold:
            flag decision for review
            set decision.status = REVISIT_SUGGESTED
```

**Storage:**

Decisions stored as SiliconDB documents with `node_type: "decision"`. The belief snapshot (SDB-4) captures the belief state at decision time. Graph edges connect the decision to its assumptions, alternatives, and the entities involved.

Key files:
- New: `src/silicon_memory/core/decision.py` — Decision, Assumption, Alternative dataclasses
- `src/silicon_memory/memory/silicondb_router.py` — Add decision storage/retrieval methods
- `src/silicon_memory/storage/silicondb_backend.py` — Decision CRUD operations
- `src/silicon_memory/reflection/engine.py` — Add decision review to consolidation cycle
- `src/silicon_memory/tools/memory_tool.py` — Add STORE_DECISION, RECALL_DECISIONS actions

#### Acceptance Criteria

- [ ] `Decision` dataclass defined with all fields: title, description, assumptions, alternatives, belief snapshot reference, status, outcome
- [ ] `SiliconMemory.commit_decision(decision)` stores the decision with a SiliconDB belief snapshot of all linked beliefs
- [ ] `SiliconMemory.recall_decisions(query, min_confidence)` returns decisions by semantic similarity
- [ ] `SiliconMemory.get_decision(id)` returns full decision with current vs. original belief confidences for each assumption
- [ ] `SiliconMemory.record_outcome(decision_id, outcome)` updates the decision with what actually happened
- [ ] `SiliconMemory.revise_decision(decision_id, new_decision)` creates a new decision linked to the original, sets original status to SUPERSEDED
- [ ] Reflection engine checks active decisions during consolidation: if any critical assumption's confidence has changed by >0.2 from the snapshot, decision status is set to REVISIT_SUGGESTED
- [ ] `MemoryTool` extended with `STORE_DECISION` and `RECALL_DECISIONS` actions for LLM function calling
- [ ] Graph edges created: decision → assumption beliefs, decision → alternative beliefs, decision → involved entities
- [ ] Decision recall returns evidence chains: decision → assumptions → current belief state → supporting/contradicting evidence
- [ ] Unit tests: full decision lifecycle (create, query, record outcome, revise)
- [ ] Unit tests: reflection flags decisions with changed assumptions
- [ ] Integration test: create decision with 3 assumptions, update one assumption's confidence significantly, run reflection, verify decision flagged for review

---

### SM-2: Decision Support Synthesis

#### Context (Why)

Raw beliefs and decision records are necessary but not sufficient. When a team lead asks "should we migrate to microservices?", they don't want a list of 20 beliefs — they want a synthesised brief that weighs the evidence, identifies risks, highlights uncertainties, and presents options with tradeoffs. This is the highest-value feature for corporate adoption: transforming structured knowledge into actionable decision support.

This builds on SM-1 (decision records) to include past similar decisions and their outcomes as evidence.

#### Goal (What)

Implement a decision brief generator that takes a question or decision to make, retrieves relevant beliefs/experiences/past decisions, and uses an LLM to synthesise an evidence-based brief with options, risks, and confidence levels.

#### SMART Objective

Implement decision support synthesis in silicon-memory, measurable by: (a) given a question, the system produces a structured brief with options, evidence, risks, and confidence, (b) every claim in the brief is linked to specific beliefs with confidence scores, (c) past similar decisions and their outcomes are included when available, (d) the brief explicitly identifies key uncertainties (high-entropy beliefs).

#### Implementation Notes (How)

**Retrieval Phase:**

```python
async def gather_decision_context(question: str) -> DecisionContext:
    # 1. Recall relevant beliefs
    beliefs = await memory.recall(RecallContext(
        query=question, max_facts=30, min_confidence=0.3
    ))

    # 2. Find past similar decisions
    past_decisions = await memory.recall_decisions(question, k=5)

    # 3. Find relevant experiences
    experiences = await memory.recall(RecallContext(
        query=question, max_experiences=10
    ))

    # 4. Identify high-uncertainty beliefs (entropy > threshold)
    uncertainties = [b for b in beliefs if b.entropy > 0.7]

    # 5. Detect contradictions among retrieved beliefs
    contradictions = await memory.find_contradictions_among(beliefs)

    return DecisionContext(beliefs, past_decisions, experiences,
                          uncertainties, contradictions)
```

**Synthesis Phase:**

An LLM call with structured output:

```python
@dataclass
class DecisionBrief:
    question: str
    summary: str                        # 2-3 sentence overview
    options: list[Option]               # Possible courses of action
    key_beliefs: list[EvidencedClaim]   # Most relevant facts with confidence
    risks: list[Risk]                   # Identified risks
    uncertainties: list[Uncertainty]    # High-entropy beliefs that matter
    past_precedents: list[Precedent]   # Similar past decisions + outcomes
    recommendation: str | None          # Optional recommendation
    confidence_in_recommendation: float

@dataclass
class EvidencedClaim:
    claim: str
    belief_id: UUID
    confidence: float
    evidence_count: int
    source_description: str

@dataclass
class Option:
    title: str
    description: str
    supporting_evidence: list[EvidencedClaim]
    opposing_evidence: list[EvidencedClaim]
    risks: list[str]
    estimated_confidence: float         # How confident are we this will work
```

Key files:
- New: `src/silicon_memory/decision/synthesis.py` — DecisionBrief generator
- New: `src/silicon_memory/decision/types.py` — DecisionBrief, Option, EvidencedClaim dataclasses
- New: `src/silicon_memory/tools/decision_tool.py` — LLM tool for decision support
- `src/silicon_memory/clients/` — LLM client used for synthesis

#### Acceptance Criteria

- [ ] `DecisionBrief` dataclass defined with: summary, options, evidenced claims, risks, uncertainties, precedents
- [ ] `SiliconMemory.generate_decision_brief(question, llm_provider)` returns a structured `DecisionBrief`
- [ ] Every claim in the brief includes: belief_id, confidence score, evidence count — no unsourced claims
- [ ] Brief includes past similar decisions and their outcomes when available (from SM-1)
- [ ] Brief explicitly lists high-uncertainty beliefs relevant to the decision (entropy > configurable threshold)
- [ ] Brief identifies contradictions among relevant beliefs
- [ ] Each option lists supporting and opposing evidence with confidence scores
- [ ] `DecisionTool` available for LLM function calling: agent can generate briefs on demand
- [ ] Brief generation completes in <30 seconds (including retrieval + LLM synthesis)
- [ ] Brief is human-readable: structured markdown output suitable for presentation in a meeting
- [ ] Unit tests: mock beliefs and decisions produce expected brief structure
- [ ] Integration test: ingest 20 beliefs about a topic, generate brief, verify all claims reference real beliefs
- [ ] Integration test: ingest contradicting beliefs, generate brief, verify contradictions are surfaced

---

### SM-3: Salience-Weighted Retrieval

#### Context (Why)

The current `RecallContext` uses `min_confidence` as a hard filter and relies on SiliconDB's RRF for ranking. As the knowledge base grows, this produces too many marginally relevant results. The brain's salience network solves this by combining multiple signals — novelty, emotional weight, recency, relevance to current context — into a single priority score. Silicon-memory needs the same: a composite salience score that combines semantic similarity, belief confidence, temporal recency, graph proximity to current context, and usage history.

This depends on SDB-1 (temporal scoring), SDB-2 (custom scoring), and SDB-3 (graph proximity) being available in SiliconDB.

#### Goal (What)

Implement a salience-weighted retrieval system in silicon-memory that uses SiliconDB's extended scoring to produce high-precision, context-aware results for decision support queries.

#### SMART Objective

Implement salience-weighted retrieval in silicon-memory using SiliconDB's custom scoring infrastructure, measurable by: (a) retrieval precision improves by >20% over current RRF baseline on a test set of 50 queries against 10K+ beliefs, (b) retrieval is context-aware — results differ when the same query is issued in different project contexts, (c) salience profiles are configurable per use case.

#### Implementation Notes (How)

**Salience Profiles:**

```python
@dataclass
class SalienceProfile:
    """Pre-configured scoring weights for different use cases."""
    vector_weight: float
    text_weight: float
    temporal_weight: float
    temporal_half_life_days: float
    confidence_weight: float
    graph_proximity_weight: float
    entropy_weight: float
    entropy_direction: str  # "prefer_low" or "prefer_high"

PROFILES = {
    "decision_support": SalienceProfile(
        vector_weight=0.3, text_weight=0.1, temporal_weight=0.15,
        temporal_half_life_days=30, confidence_weight=0.25,
        graph_proximity_weight=0.15, entropy_weight=0.05,
        entropy_direction="prefer_low"
    ),
    "exploration": SalienceProfile(
        vector_weight=0.4, text_weight=0.2, temporal_weight=0.05,
        temporal_half_life_days=365, confidence_weight=0.1,
        graph_proximity_weight=0.1, entropy_weight=0.15,
        entropy_direction="prefer_high"  # Find uncertain things
    ),
    "context_recall": SalienceProfile(
        vector_weight=0.2, text_weight=0.1, temporal_weight=0.3,
        temporal_half_life_days=7, confidence_weight=0.1,
        graph_proximity_weight=0.25, entropy_weight=0.05,
        entropy_direction="prefer_low"
    ),
}
```

**Context Detection:**

The system automatically determines context nodes for graph proximity:
1. Extract entities from the current query
2. Resolve to canonical entities via EntityResolver
3. Add current session's working memory entities
4. Pass as `graph_context_nodes` to SiliconDB search

Key files:
- New: `src/silicon_memory/retrieval/salience.py` — SalienceProfile and profile definitions
- `src/silicon_memory/memory/silicondb_router.py` — Update `recall()` to use salience profiles
- `src/silicon_memory/entities/resolver.py` — Extract context entities for graph proximity

#### Acceptance Criteria

- [ ] `SalienceProfile` dataclass defined with weights for all scoring signals
- [ ] At least 3 pre-configured profiles: `decision_support`, `exploration`, `context_recall`
- [ ] `RecallContext` extended with optional `salience_profile` parameter
- [ ] When a salience profile is provided, `recall()` translates it to SiliconDB `SearchScoring` configuration
- [ ] Context entities are automatically extracted from query and working memory for graph proximity scoring
- [ ] Custom profiles can be created by users (not limited to presets)
- [ ] Retrieval precision improvement measured: create test set of 50 queries with human-judged relevance labels, measure precision@5 with default RRF vs. salience-weighted. Target: >20% improvement
- [ ] Context sensitivity verified: same query with different `graph_context_nodes` returns different result rankings
- [ ] Backward compatible: `recall()` without salience profile behaves identically to current implementation
- [ ] Unit tests: each profile produces different SearchScoring configurations
- [ ] Integration test: salience-weighted recall on 1K+ beliefs returns measurably better precision than default

---

### SM-4: Context Switch Snapshots

#### Context (Why)

When a user switches between tasks or projects across sessions, the LLM loses all context about where they left off. The brain handles this with hippocampal snapshots — a compressed bookmark of the current state that enables rapid resumption. Silicon-memory has experiences and working memory, but no explicit suspend/resume mechanism. A developer working on project Alpha who switches to project Beta for a week should be able to resume Alpha with a brief like: "Last time you were debugging the auth module. You'd identified the token expiry bug and your next step was to add refresh logic."

#### Goal (What)

Implement an automatic context snapshot mechanism that captures the current working state when a session ends or a task switch is detected, and injects it into the LLM context when that task is resumed.

#### SMART Objective

Implement context switch snapshots in silicon-memory, measurable by: (a) session end or explicit task switch triggers a snapshot capturing working memory, recent episode summary, and next steps, (b) session resume retrieves the most recent snapshot for the detected task context, (c) snapshot injection into LLM context enables continuation of the previous conversation thread without manual recap.

#### Implementation Notes (How)

**Snapshot Creation:**

At session end or explicit task switch:

```python
@dataclass
class ContextSnapshot:
    id: UUID
    task_context: str               # "project-alpha/auth-module"
    summary: str                    # LLM-generated: "You were debugging..."
    working_memory: dict[str, Any]  # Copy of current working memory
    recent_experiences: list[UUID]  # Last N experience IDs in this context
    next_steps: list[str]           # Extracted from conversation
    open_questions: list[str]       # Unresolved questions
    created_at: datetime
    session_id: str
    node_type: str = "snapshot"
```

Generation: an LLM call with recent conversation + working memory as input, structured output of summary + next steps + open questions.

**Snapshot Retrieval:**

On session start, detect task context (from explicit user indication or entity matching from first message), retrieve most recent snapshot for that context, inject into system prompt or first LLM context.

Key files:
- New: `src/silicon_memory/core/snapshot.py` — ContextSnapshot dataclass
- `src/silicon_memory/memory/silicondb_router.py` — Add snapshot creation/retrieval methods
- `src/silicon_memory/conversation/store.py` — Hook snapshot creation into session end
- `src/silicon_memory/tools/memory_tool.py` — Add SWITCH_CONTEXT, RESUME_CONTEXT actions

#### Acceptance Criteria

- [ ] `ContextSnapshot` dataclass defined with: task_context, summary, working_memory, recent_experiences, next_steps, open_questions
- [ ] `SiliconMemory.create_snapshot(task_context, llm_provider)` generates a snapshot from current working memory and recent conversation
- [ ] `SiliconMemory.get_latest_snapshot(task_context)` retrieves the most recent snapshot for a given task context
- [ ] Snapshot summary is LLM-generated: concise, includes where the user was, what they were doing, and what's next
- [ ] Snapshot stored as SiliconDB document with `node_type: "snapshot"`, searchable by task_context metadata
- [ ] `MemoryTool` extended with `SWITCH_CONTEXT` (creates snapshot + clears working memory) and `RESUME_CONTEXT` (retrieves snapshot + loads into working memory) actions
- [ ] Snapshot retrieval is fast: <100ms (it's a metadata-filtered query, not a semantic search)
- [ ] Old snapshots for the same task context are retained (not overwritten) for history, but only the latest is used for resume
- [ ] Unit tests: create snapshot, switch context, resume, verify snapshot content matches
- [ ] Integration test: simulate multi-session workflow — session 1 on task A, session 2 on task B, session 3 resume task A — verify correct snapshot is retrieved
- [ ] Integration test with LLM: snapshot summary is coherent and captures key working state

---

### SM-5: Passive Ingestion Adapters

#### Context (Why)

For corporate adoption, the system cannot rely on users manually storing facts and experiences. Knowledge needs to flow in from existing tools — meeting transcripts, Slack conversations, documents, and news feeds. Each source maps naturally to the experience → reflection → belief pipeline: raw input becomes experiences, the dreaming phase extracts beliefs. The first adapter should target the highest-value, lowest-friction source.

Meeting transcripts are the best starting point: they contain decisions, action items, context, and rationale in a structured temporal format. They're already being generated by tools like Otter.ai, Whisper, and built-in meeting transcription.

#### Goal (What)

Implement an ingestion adapter framework with an initial meeting transcript adapter that converts transcript text into experiences, extracts entities and decisions, and feeds them into the existing reflection pipeline.

#### SMART Objective

Implement a passive ingestion adapter framework with a meeting transcript adapter, measurable by: (a) meeting transcripts are ingested as structured experiences with speaker attribution, (b) entities mentioned in transcripts are resolved against the entity registry, (c) action items and decisions are extracted and stored, (d) the framework is extensible for additional source types (Slack, documents) without core changes.

#### Implementation Notes (How)

**Adapter Framework:**

```python
class IngestionAdapter(Protocol):
    """Base protocol for all ingestion adapters."""

    @property
    def source_type(self) -> str: ...

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: SiliconMemory,
        llm_provider: LLMProvider | None = None
    ) -> IngestionResult: ...

@dataclass
class IngestionResult:
    experiences_created: int
    entities_resolved: int
    decisions_detected: int
    errors: list[str]
```

**Meeting Transcript Adapter:**

```python
class MeetingTranscriptAdapter(IngestionAdapter):
    """
    Ingests meeting transcripts as experiences.

    Expected input: plain text or structured format with:
    - Speaker labels (optional)
    - Timestamps (optional)
    - Raw transcript text

    Output:
    - One experience per topic/segment
    - Entities resolved from speaker names and mentioned terms
    - Action items extracted as procedural memories
    - Decisions extracted as decision records (SM-1)
    """
```

Processing pipeline:
1. Parse transcript into segments (by topic shift or time blocks)
2. For each segment, create an experience with speaker attribution and timestamp
3. Run entity resolution on mentioned names, companies, technologies
4. Use LLM to extract action items → store as procedural memories
5. Use LLM to extract decisions → store as decision records (SM-1)
6. Link all via graph edges: meeting → segments → entities → decisions

Key files:
- New: `src/silicon_memory/ingestion/adapter.py` — IngestionAdapter protocol, IngestionResult
- New: `src/silicon_memory/ingestion/meeting.py` — MeetingTranscriptAdapter
- New: `src/silicon_memory/ingestion/` — Package for future adapters

#### Acceptance Criteria

- [ ] `IngestionAdapter` protocol defined with `source_type` property and `ingest()` method
- [ ] `IngestionResult` captures: experiences created, entities resolved, decisions detected, errors
- [ ] `MeetingTranscriptAdapter` accepts plain text transcript input
- [ ] Transcript is segmented into topic blocks (LLM-based or heuristic)
- [ ] Each segment stored as an experience with metadata: speaker(s), timestamp range, meeting_id
- [ ] Speaker names resolved against entity registry (creates new entities if unknown)
- [ ] Action items extracted and stored as procedural memories with owner attribution
- [ ] Decisions extracted and stored as decision records (SM-1) when detected
- [ ] Graph edges created: meeting_experience → participants, meeting_experience → decisions, meeting_experience → action_items
- [ ] Adapter framework is extensible: new adapters can be added by implementing the protocol without modifying existing code
- [ ] `SiliconMemory.ingest_from(adapter, content, metadata)` provides a unified entry point
- [ ] Unit tests: mock transcript produces expected experiences, entities, and decisions
- [ ] Integration test: ingest a real meeting transcript, run reflection, verify beliefs extracted from meeting content
- [ ] Error handling: malformed input produces partial results + error list, not a crash

---

### SM-6: News / External Knowledge Integration

#### Context (Why)

Corporate decision support benefits from combining internal knowledge (meetings, decisions, project context) with external intelligence (industry news, market trends, competitor actions). The two-source model — internal beliefs validated or contradicted by external evidence — provides richer decision support than either alone. News articles are also an excellent corpus for stress-testing the entire pipeline at scale: extraction, entity resolution, contradiction detection, belief consolidation, and temporal tracking.

This depends on SM-5 (adapter framework) for the ingestion infrastructure.

#### Goal (What)

Implement a news ingestion adapter that processes articles into beliefs and entities, with a source distinction between internal and external knowledge, enabling cross-referencing queries.

#### SMART Objective

Implement news/external knowledge integration in silicon-memory, measurable by: (a) news articles are ingested as external experiences and processed into beliefs with source attribution, (b) internal and external beliefs are distinguished and queryable separately or together, (c) cross-referencing identifies where internal beliefs are supported or contradicted by external evidence, (d) the system handles 1000+ articles without retrieval quality degradation.

#### Implementation Notes (How)

**Source Type Extension:**

```python
class SourceType(str, Enum):
    INTERNAL = "internal"      # Conversations, meetings, decisions
    EXTERNAL = "external"      # News, industry reports, public data

# Added to belief metadata
@dataclass
class BeliefSource:
    source_type: SourceType
    source_name: str           # "Reuters", "team-standup", etc.
    source_url: str | None     # For external sources
    credibility: float         # 0.0-1.0, configurable per source
```

**News Adapter:**

```python
class NewsArticleAdapter(IngestionAdapter):
    """
    Ingests news articles as external experiences.

    Processing:
    1. Extract title, date, source, content
    2. Create experience with source_type=EXTERNAL
    3. Extract entities (companies, people, technologies)
    4. Extract key claims as belief candidates
    5. Tag with source credibility
    """
```

**Cross-Referencing:**

A new recall mode that retrieves both internal and external beliefs and highlights agreement/disagreement:

```python
@dataclass
class CrossReferenceResult:
    query: str
    internal_beliefs: list[RecallResult]
    external_beliefs: list[RecallResult]
    agreements: list[tuple[RecallResult, RecallResult]]    # Internal + external that agree
    contradictions: list[tuple[RecallResult, RecallResult]] # Internal + external that disagree
    external_only: list[RecallResult]  # External knowledge not reflected internally
```

Key files:
- New: `src/silicon_memory/ingestion/news.py` — NewsArticleAdapter
- New: `src/silicon_memory/core/source.py` — SourceType, BeliefSource
- `src/silicon_memory/memory/silicondb_router.py` — Add cross-reference recall
- `src/silicon_memory/storage/silicondb_backend.py` — Source type filtering in queries

#### Acceptance Criteria

- [ ] `SourceType` enum with INTERNAL and EXTERNAL values
- [ ] `BeliefSource` metadata attached to beliefs, including source_type, source_name, credibility
- [ ] `NewsArticleAdapter` implements `IngestionAdapter` protocol
- [ ] Articles ingested as experiences with `source_type: EXTERNAL` and source URL
- [ ] Entity extraction from articles resolves against existing entity registry
- [ ] Key claims extracted as belief candidates with source credibility weighting
- [ ] `RecallContext` extended with optional `source_type` filter: recall internal only, external only, or both
- [ ] `SiliconMemory.cross_reference(query)` returns `CrossReferenceResult` with agreements, contradictions, and external-only findings
- [ ] Source credibility weights the belief confidence: low-credibility source → lower initial confidence
- [ ] Reflection engine handles external beliefs: can strengthen internal beliefs confirmed by external sources, flag contradictions
- [ ] Scale test: ingest 1000 articles, verify recall precision doesn't degrade vs. 100 articles (measured on fixed test query set)
- [ ] Unit tests: news adapter produces expected beliefs with source attribution
- [ ] Integration test: ingest internal belief + contradicting news article, run reflection, verify contradiction detected
- [ ] Integration test: cross-reference query returns correctly categorised agreements and contradictions

---

## Task Summary

| ID | Task | Depends On | Workstream |
|----|------|-----------|------------|
| SDB-1 | Temporal Ranking Signal | None | SiliconDB |
| SDB-2 | Custom Scoring Configuration | SDB-1 | SiliconDB |
| SDB-3 | Graph Proximity Scoring | SDB-2 | SiliconDB |
| SDB-4 | Belief Snapshots | None | SiliconDB |
| SDB-5 | Edge Embeddings | None | SiliconDB | **COMPLETE** — separat edge-HNSW, EdgeMetadataStore, WAL opcode 15, Python-bindinger |
| SM-1 | Decision Records | SDB-4 | Silicon-Memory |
| SM-2 | Decision Support Synthesis | SM-1 | Silicon-Memory |
| SM-3 | Salience-Weighted Retrieval | SDB-1, SDB-2, SDB-3 | Silicon-Memory |
| SM-4 | Context Switch Snapshots | None | Silicon-Memory |
| SM-5 | Passive Ingestion Adapters | None | Silicon-Memory |
| SM-6 | News / External Knowledge | SM-5 | Silicon-Memory |

## Suggested Execution Order

**Phase 1 — Foundation (parallel tracks)**
- Track A: SDB-1 → SDB-2 (scoring pipeline)
- Track B: SDB-4 (belief snapshots, independent)
- Track C: SM-4 (context snapshots, independent)
- Track D: SM-5 (ingestion adapters, independent)

**Phase 2 — Core Decision Support**
- SM-1 (decision records, needs SDB-4)
- SDB-3 (graph proximity, needs SDB-2)
- SM-6 (news integration, needs SM-5)

**Phase 3 — Advanced Retrieval**
- SM-3 (salience retrieval, needs SDB-1 + SDB-2 + SDB-3)
- SM-2 (decision synthesis, needs SM-1)

**Phase 4 — Edge Intelligence (COMPLETE)**
- SDB-5 (edge embeddings) — ✅ Implementert med separat HNSW, WAL, Python-bindinger. Bruker samme embedding-dimensjon som dokumenter.
