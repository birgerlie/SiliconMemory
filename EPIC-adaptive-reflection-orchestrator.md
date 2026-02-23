# EPIC: Adaptive Reflection Orchestrator (4B Meta-Controller)

## Epic Summary

Add a lightweight orchestration layer that decides when to run `reflect` and `dream`, and what each run should focus on (`temporal`, `contradictions`, `bridge discovery`, `entity cleanup`) using cheap runtime signals plus a local 4B model as a bounded policy advisor.

The 4B model is not used for truth extraction or belief storage decisions. It is used only for scheduling and prioritization recommendations under strict guardrails.

## Problem

Current operation is mostly manual and fixed-interval:

- Reflection/dream timing is not adaptive to workload conditions.
- Expensive phases can run when they have low expected information gain.
- Temporal and contradiction-focused work can be delayed or under-prioritized.
- Local model/runtime instability can cause slowdowns when the system should back off.

## Goals

1. Decide run cadence adaptively from live metrics.
2. Route each run to a clear focus profile.
3. Keep cost/latency bounded with deterministic guardrails.
4. Improve temporal progression yield and contradiction closure rate.

## Non-Goals

- Replacing existing extraction/reflection logic.
- Letting LLM freeform-control commits or safety decisions.
- Introducing external cloud dependency for scheduling decisions.

## Proposed Design

### Control Inputs

Per-cycle features collected from backend and recent run journals:

- `new_extracted_count`
- `new_triplet_count`
- `new_occurred_on_count`
- `timeline_progresses_count`
- `contradiction_candidates_count`
- `uncertain_beliefs_count`
- `avg_confidence_recent`
- `last_reflect_minutes`
- `last_dream_minutes`
- `llm_error_rate_recent`
- `model_rate_limit_events`
- `worker_health_signal`

### Deterministic Guardrails (Hard Rules)

- Force `reflect` when:
  - `new_extracted_count >= 25`, or
  - `last_reflect_minutes >= 30`.
- Force `dream` when:
  - `new_extracted_count >= 120`, or
  - `last_dream_minutes >= 180`.
- Force `temporal` focus when:
  - `new_occurred_on_count >= 10`, or
  - `timeline_progresses_count / max(new_occurred_on_count, 1) < 0.15`.
- Backoff when:
  - `llm_error_rate_recent` high, or
  - rate-limit/worker-health signals exceed threshold.

### 4B Policy Advisor (Soft Rules)

A local 4B model receives the feature vector and returns bounded JSON:

```json
{
  "run_reflect_now": true,
  "run_dream_now": false,
  "focus": ["temporal", "contradictions"],
  "priority": "high",
  "reason": "temporal coverage low vs new occurred_on events"
}
```

Output is validated against schema and constrained by hard guardrails.

### Execution Profiles

- `temporal`: prioritize temporal hypothesis generation/validation.
- `contradictions`: increase contradiction scan/validation budget.
- `bridge_discovery`: favor structural discovery operators.
- `maintenance`: low-cost cleanup and confidence hygiene.

## Implementation Workstreams

### W1: Metrics Snapshot Builder

- Add a `scheduler_metrics` collector from DB and latest run journals.
- Output one normalized feature payload per scheduling tick.

### W2: Policy Engine

- Implement deterministic rules engine.
- Add optional 4B advisor stage with strict JSON schema validation.
- Merge deterministic + advisor outputs into final `RunPlan`.

### W3: Orchestrator Runner

- New orchestrator entrypoint (`scripts/run_orchestrator_tick.py`).
- Executes `reflect`/`dream` with selected focus profile.
- Writes plan + outcomes to metrics journal.

### W4: Safety + Backoff

- Add rate-limit and worker-health aware throttling.
- Force fallback to deterministic-only mode when advisor unavailable.

### W5: Evaluation

- Compare fixed schedule vs adaptive schedule on:
  - temporal progression yield,
  - contradiction closure,
  - wall-clock and token spend,
  - run stability (errors/timeouts).

## Acceptance Criteria

- `RunPlan` JSON schema enforced; invalid advisor output is rejected safely.
- Deterministic guardrails always take precedence over advisor suggestions.
- Orchestrator can run in:
  - `deterministic_only`,
  - `deterministic_plus_4b`.
- Temporal KPI improvement on test corpus:
  - `timeline_progresses_from_to / occurred_on` improves by >= 25% vs fixed baseline.
- Stability KPI:
  - no increase in failed cycles relative to fixed schedule.
- Cost KPI:
  - no increase in average cost per useful hypothesis.

## Rollout Plan

1. Shadow mode: generate plans, do not execute.
2. Execute in deterministic-only mode.
3. Enable 4B advisor for focus selection only.
4. Expand to cadence recommendation after stability gates pass.

## Risks

- Overfitting schedule to noisy short-term metrics.
- Advisor drift producing low-value focus choices.
- Runtime model instability causing orchestration delays.

## Mitigations

- Keep hard deterministic floor rules.
- Add plan audit trail and post-run scoring.
- Use cooldown windows and confidence thresholds.
- Auto-fallback to deterministic mode on advisor failures.

## Deliverables

- `scheduler_policy.py` (policy + guardrails)
- `scheduler_metrics.py` (feature extraction)
- `run_orchestrator_tick.py` (execution driver)
- `export/metrics/orchestrator/*.json` (plan/outcome logs)
- tests for policy decisions and fallback behavior

