# Async Data Plane Contract

## Scope

This contract defines reliability behavior between `silicon-memory` and the remote SiliconDB gRPC server.

## Consistency Model

- Write path: at-least-once transport with client-side idempotency suppression.
- Read-after-write: expected eventually consistent across independent requests; same-process writes are immediately visible through local flow.
- Decision status transitions (`ACTIVE -> REVISIT_SUGGESTED`) are persisted as explicit metadata updates.

## Retry Policy

- Transient failures are retried with bounded exponential backoff + jitter.
- Defaults:
  - `retry_attempts = 3`
  - `retry_base_ms = 100`
  - `retry_max_ms = 2000`
  - `request_timeout_s = 10.0`
- Retryable classes include timeout/unavailable/resource-exhausted style transport errors.

## Idempotency Policy

- Critical write operations use deterministic idempotency keys derived from operation payload.
- Keys are retained in-process for `idempotency_ttl_s` (default `600s`).
- Duplicate keys inside TTL are suppressed.

## Backpressure Policy

- Write operations are rate-limited by a per-backend semaphore.
- Default max concurrent writes: `max_inflight_mutations = 32`.
- Reflection/extraction workloads share the same write budget to avoid runaway write bursts.

## Operational Controls (CLI)

- `--db-retry-attempts`
- `--db-retry-base-ms`
- `--db-retry-max-ms`
- `--db-request-timeout-s`
- `--db-max-inflight-mutations`
- `--db-idempotency-ttl-s`

## Known Limits

- Idempotency cache is process-local (not distributed).
- Read-path retries are only partially applied; high-volume reads may still execute direct calls.
- Full consistency semantics depend on SiliconDB server behavior.

