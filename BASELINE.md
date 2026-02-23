# Baseline Snapshot

Date: 2026-02-18

## Verified in this pass

- Reflection module sync from `/Users/birger/code/epstein/SiliconMemory` to `/Users/birger/code/SiliconMemory`.
- Compatibility gap closure for reflection-required backend/router APIs.
- Decision intelligence upgrades:
  - persisted `REVISIT_SUGGESTED` status transitions,
  - enriched `get_decision(...)` with assumption drift metadata.
- Async reliability controls added for key write paths:
  - retry/backoff,
  - idempotency suppression,
  - write backpressure.
- Server CLI/config now expose reliability controls.

## Test Results

- `tests/test_decision.py`: 13 passed
- `tests/test_backend_reliability.py`: 3 passed
- Combined run: 16 passed
- CLI/config smoke: reliability flags parse and map into `ServerConfig`
- Gold-set NER eval script added and run:
  - script: `scripts/eval_ner_against_gold.py`
  - output: `export/metrics/ner_eval_report.images.json`
  - current run reported micro F1 `1.0` (likely circular because gold was built from the same result corpus)
- Non-circular NER baseline run:
  - prediction script: `scripts/predict_ner_from_gold.py`
  - predictions: `export/metrics/ner_predictions.from_rules_heuristic.jsonl`
  - eval output: `export/metrics/ner_eval_report.from_rules_heuristic.json`
  - micro: precision `0.1526`, recall `0.1667`, F1 `0.1593`
- Reflection benchmark added and run:
  - script: `scripts/benchmark_reflection.py`
  - output: `export/metrics/reflection_benchmark.baseline.json`
  - 5-cycle no-workload baseline: warm cycle ~0.64s, steady-state ~0.054-0.076s per reflect call
- Reflection optimization pass benchmark:
  - output: `export/metrics/reflection_benchmark.after_cooldown.json`
  - warm cycle ~0.57s, steady no-work cycles ~0.0003-0.0004s
  - improvement source: empty extracted-belief scan cooldown (`empty_extracted_cooldown_s`)
- Post-SiliconDB-pull reflection benchmark:
  - output: `export/metrics/reflection_benchmark.post_db_pull.json`
  - cycle 1 ~0.0565s, cycles 2-3 ~0.0004s (no-workload path remains fast)

## Pending for full baseline parity

- One live reflection cycle against production-like data (requires runtime stack + seed data).
- One live decision brief generation in deployed server path.
- Recall smoke with `source_type=external` using real external-ingested beliefs.
- Gold-set named-entity scoring run from `/Users/birger/code/epstein/SiliconMemory/results/IMAGES*/*.json`.
- Non-circular NER eval run against fresh predictions (not the source corpus used to build gold).

## Runtime Note

- Pulled `deps/silicondb` to `dd5b07c` (includes upstream #171 `min_probability` fix).
- New upstream blocker discovered: Python gRPC stub import regression (`silicondb_pb2_grpc.py` top-level import), tracked at:
  - https://github.com/birgerlie/SiliconDB/issues/172
- Local temporary shim applied in vendored dependency to keep SiliconMemory runtime unblocked.
- Python `grpcio` is required for `SiliconDBClient` gRPC transport.
