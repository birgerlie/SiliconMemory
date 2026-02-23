# Intelligence Layer Gap List

This tracks remaining gaps while preserving the EPIC intelligence architecture.

## Keep (Already Present)

- Decision records lifecycle APIs in router/backend (`commit_decision`, `recall_decisions`, `get_decision`, `record_outcome`, `revise_decision`)
- Decision synthesis pipeline (`DecisionBriefGenerator`, `DecisionTool`)
- Reflection and dreaming pipeline (incremental reflect, inference, hypotheses, consolidation hooks)
- Context snapshots (`create_snapshot`, `get_latest_snapshot`, list)
- Salience profiles and weighted recall plumbing
- Ingestion adapters (meeting, news, email, chat/Slack/Teams/Discord)
- Cross-reference API (`cross_reference`) with internal/external split
- MemoryTool actions for decision + context switching flows

## Closed In This Pass

1. `SiliconMemory.generate_decision_brief(...)` API missing
- Added first-class method on router API that uses `DecisionBriefGenerator`.

2. Decision drift flag is not persisted as decision status
- Added backend `update_decision_status(...)`.
- Reflection drift review now persists `REVISIT_SUGGESTED` with reason.

3. `RecallContext.source_type` is defined but not applied in recall pipeline
- Added source-type filtering in router recall (`internal` / `external` / all).

4. `get_decision(...)` does not enrich with assumption drift view
- Added decision enrichment with per-assumption:
  - `confidence_at_decision`
  - `current_confidence`
  - `delta` / `abs_delta`
  - `drift_threshold_exceeded` (critical + >0.2)
- Adds `metadata.assumption_drift` and `metadata.needs_revisit`.

## Gaps To Close

1. Async remote data-plane hardening is incomplete at app layer
- Current gRPC usage still has many synchronous calls without explicit retry, backpressure, or idempotency guarantees.
- Required:
  - idempotency keys for mutation operations,
  - retry policy with bounded backoff and error classification,
  - write/read consistency contract documentation and enforcement points,
  - concurrency/backpressure guardrails for reflection/extraction workloads.

2. Adaptive orchestration for reflect/dream cadence and focus is missing
- Current system is mostly manual/fixed cadence.
- Need an adaptive scheduler that uses runtime signals to decide:
  - when to run reflect/dream,
  - what to prioritize (temporal, contradictions, bridge discovery, maintenance),
  - when to back off under model/runtime stress.
- Epic: `EPIC-adaptive-reflection-orchestrator.md`
- Runtime guardrails issue for heavy dream/hypothesis phases:
  - https://github.com/birgerlie/SiliconMemory/issues/18

### Progress This Pass

- Added backend retry wrapper with bounded exponential backoff + jitter (`_run_db`).
- Added transient error classification (`_is_retryable_exception`).
- Added mutation backpressure semaphore (`max_inflight_mutations`).
- Added idempotency key generation + TTL dedupe cache for writes.
- Applied reliability controls to key mutation paths:
  - `commit_belief`
  - `record_experience`
  - `commit_procedure`
  - `mark_experiences_processed`
  - `mark_experiences_extracted`
  - `mark_beliefs_reflection_processed`
  - `set_working` / `delete_working` / cleanup deletes
  - `commit_decision`
  - `record_decision_outcome`
  - `update_decision_status`
  - supersede write in `revise_decision`
- Added focused tests:
  - `tests/test_backend_reliability.py` (retry + idempotency)
- SiliconDB integration note:
  - `query_triples(min_probability=...)` mismatch fixed upstream in `deps/silicondb` update (`dd5b07c`, issue #171)
  - new upstream blocker: generated Python gRPC stubs import `silicondb_pb2` as top-level module
  - tracked at: https://github.com/birgerlie/SiliconDB/issues/172
  - retrieval/indexing consistency gap for search/filter visibility tracked at:
    https://github.com/birgerlie/SiliconDB/issues/176
  - developer-mode full reset (stop serving + wipe index + restart) tracked at:
    https://github.com/birgerlie/SiliconDB/issues/177
  - local temporary shim applied in vendored dependency until upstream fix lands

## Recommended Closure Order

1. Add async data-plane reliability controls (retry/idempotency/backpressure).
