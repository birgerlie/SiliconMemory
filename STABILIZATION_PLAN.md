# Stabilization Plan

Goal: make ingest -> extract -> reflect -> dream deterministic and repeatable without doc-specific tuning.

## Gate 1: Episodic Retrieval Consistency (Current Priority)

Success criteria:
- Newly ingested experiences are visible through:
  - direct `get_experience(id)`,
  - `get_recent_experiences(...)`,
  - `get_unprocessed_experiences(...)`,
  - `get_unextracted_experiences(...)`,
  - `count_extraction_progress()`.
- No false-zero state after ingest in the same run.

Implementation status:
- Added app-layer fallback in `silicondb_backend`:
  - broad multi-probe experience search,
  - direct-id hydration fallback via indexed external IDs.
- `count_extraction_progress()` now enforces tenant/user access filtering
  (prevents cross-tenant overcount in shared DB).
- Experience flag updates now merge metadata (processed/extracted no longer
  overwrite each other).
- Batch harness now performs true ingest before extraction and waits for
  visibility (`record_experience` + `wait_for_ingest_visibility`).
- Added app-layer batch ingest barrier API on `SiliconMemory`:
  - `ingest_experiences_batch(..., wait_for_visibility=True, ...)`
  - returns explicit batch state (`SUCCEEDED|PARTIAL|FAILED|TIMEOUT`)
    and accepted IDs for deterministic downstream flow wiring.
- Ingestion adapters now use a shared persistence helper with safe fallback:
  - `document/email/meeting/chat/news` call `persist_experiences(...)`
  - uses batch API when available, falls back to per-item `record_experience`
    for mocked/minimal memory implementations.
- Upstream DB feature request opened for deterministic index/search visibility:
  - https://github.com/birgerlie/SiliconDB/issues/176

Validation:
- `tests/test_backend_reliability.py` includes fallback coverage.
- Added regression coverage for:
  - access-scoped extraction progress counting,
  - processed/extracted metadata merge preservation.
  - processed/extracted order-independence (`processed->extracted` and vice versa).
  - inaccessible search-result filtering in extraction progress.
- Added script-level smoke coverage for CI-fast full-cycle logic:
  - `tests/test_script_smoke.py`
  - validates batch ingest/extract/reflect retrieval-sanity wiring
  - validates benchmark dream retrieval-sanity wiring
- Run 5-doc cycle and confirm reflection processes experiences (>0) after ingest.
  - Latest: `export/metrics/stabilization/stabilization_extract_reflect_5docs_local_20260219_continue_v8.json`
    shows `direct_experience_visibility=5/5`, `extraction_progress.extracted=5`,
    `unprocessed_experiences_count=0`, `unextracted_experiences_count=0`.

## Gate 2: Temporal Normalization and Propagation

Success criteria:
- Timeline patterns produce stable `occurred_on` triplets when canonical date is available.
- Relative/ambiguous expressions are not anchored as stable long-term dates.
- Temporal metadata is attached consistently for beliefs when safe.

Implementation status:
- Timeline candidate generation now derives canonical date from event text/description when context date is missing.
- Relative date fallback is guarded in temporal context builder.
- Temporal tests extended in `tests/test_reflection_temporal_enrichment.py`.

## Gate 3: Procedural + Working Memory Coverage

Success criteria:
- Non-zero procedural and working entries in full-cycle runs where documents contain actionable/process content.
- Retrieval APIs return these types consistently.

Validation:
- Add acceptance checks in batch run reports:
  - counts per memory type,
  - retrieval sanity checks after ingest and after reflect.
- Done in scripts:
  - `scripts/run_batch_extract_reflect.py`
  - `scripts/benchmark_reflection.py`
  - both now emit `retrieval_sanity` with episodic/procedural/working signals.
- Latest dream check:
  - `export/metrics/stabilization/stabilization_reflect_dream_5docs_local_20260219_continue_v8b.json`
  - `questions_generated=5`
  - `procedures_created=4`
  - `retrieval_sanity.procedures_count=8`
  - `retrieval_sanity.open_questions_count=5`
  - question generator retries heuristic fallback under transient visibility lag.

## Gate 4: Full-Cycle Repeatability

Success criteria:
- Two consecutive runs on same 5-doc set produce stable ranges for:
  - extracted patterns,
  - committed beliefs,
  - reflected experiences,
  - generated hypotheses.
- No regressions when cache mode is `off`.

Validation protocol:
1. Clean DB.
2. Ingest 5 docs.
3. Extract + reflect.
4. Dream.
5. Repeat from clean DB and compare metrics deltas.

Implementation status:
- Added repeatability harness:
  - `scripts/eval_repeatability.py`
  - runs two isolated A/B cycles with unique tenant/user IDs,
    then computes gate checks + metric deltas.
  - forwards extraction `llm_temperature` (default 0.0 in repeatability mode)
    to reduce run-to-run extraction variance.
- Added tests for repeatability evaluation + script smoke wiring:
  - `tests/test_eval_repeatability.py`
  - `tests/test_script_smoke.py`
- Added stricter optional gate checks:
  - `--require-open-questions-min`
  - `--require-procedures-min`

Latest strict run:
- `export/metrics/stabilization/repeatability_report_20260219_continue_strict_v2.json`
- settings: `require_open_questions_min=1`, `require_procedures_min=1`
- result: `overall_passed=true`, `gates_passed=true`, `metrics_passed=true`

## Gate 5: Verifiable Knowledge and Provenance

Goal:
- Shift from extraction volume to canonical, testable knowledge with strict provenance.

Strategy document:
- `CLAIM_STRATEGY.md`

Success criteria:
- Canonical claims are separated from raw observations.
- Every canonical claim has complete provenance contract fields.
- Ingest flow classifies each observation as one of:
  - `MERGE`
  - `CONTRADICTION`
  - `RELATED`
  - `NEW`
- Trivial observations are retained in raw layer but blocked from canonical promotion.
- Reflection/dream execution is trigger-driven by epistemic signals.

Validation:
- Track quality gates from strategy:
  - `provenance_completeness_rate`
  - `merge_precision`
  - `contradiction_precision`
  - `non_trivial_claim_rate`
  - `testable_hypothesis_rate`
  - `insight_yield_per_doc`
- Batch harness now emits additional strategy metrics:
  - `temporal_implication_count`
  - `avg_candidates_per_observation`
  - `llm_escalation_rate`
  - `claim_strategy.gates` snapshot for threshold checks
