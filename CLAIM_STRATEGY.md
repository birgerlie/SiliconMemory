# Claim Strategy: Verifiable Knowledge With Provenance

Goal: move from "fact storage" to verifiable, testable knowledge with explicit provenance and uncertainty handling.

## 1) Problem Framing

Current weakness:
- We store many extracted triples, but observations and canonical knowledge are mixed.
- Duplicate/semi-duplicate claims inflate apparent evidence.
- Reflection/dream can overfit to generic abstractions ("systematically", "multiple (n)").
- Provenance exists in metadata, but is not yet a strict gate for canonical belief quality.

Consequence:
- High volume, low epistemic gain.
- Contradictions and temporal implications are under-exploited.
- Hypotheses are too often non-testable.

## 2) Target State

For each canonical claim, the system must answer:
1. What is claimed?
2. Why do we believe it?
3. Which observations support or contradict it?
4. What is uncertain?
5. How can the claim be tested or falsified?

Output should produce value beyond a simple summary:
- cross-document links,
- contradiction surfaces,
- timeline implications,
- testable hypotheses.

## 3) Core Design Shift

Separate storage concerns:
- `raw_observations`: all extracted observations are retained.
- `canonical_claims`: curated belief layer used for reasoning and recall.

Canonical unit:
- `canonical_claim_id` is the primary identity for knowledge.
- New observations do not automatically create new canonical claims.
- New observations must be classified against existing claims.

Dual representation (kept in sync):
- Graph triple representation (reasoning, contradiction, temporal traversal).
- Surface retrieval representation (BM25 + embedding for semantic recall).

## 4) Ingest Decision Pipeline

### Step A: Candidate Recall (high recall, low cost)

For each new observation, gather candidate claims from:
- exact `triplet_key`,
- subject/predicate and entity neighborhood,
- BM25 on claim surface text,
- embedding search on claim surface text,
- graph-neighbor expansion.

Keep recall aggressive (do not optimize for precision here).

### Step B: Precision Judge (epistemic classification)

Classify relation to best candidate:
- `MERGE`: same canonical claim.
- `CONTRADICTION`: incompatible with a canonical claim.
- `RELATED`: contextually linked but distinct claim.
- `NEW`: genuinely new canonical claim.

Policy:
- rules first for deterministic fast paths,
- LLM for gray-zone precision decisions.

### Step C: Canonical Update

- `MERGE`: update evidence/provenance and confidence, do not clone claim.
- `CONTRADICTION`: attach contradiction relation with provenance for both sides.
- `RELATED`: add relation edge, no merge.
- `NEW`: create new canonical claim only if anti-triviality checks pass.

## 5) Provenance Contract (required fields)

Canonical claim evidence must include:
- `source_doc_id`
- `evidence_span` (or equivalent source pointer)
- `extraction_run_id`
- `extractor_model` and `extractor_version`
- `extracted_at`
- `source_type`
- `confidence_basis` (rule/LLM/hybrid)

Missing required provenance means:
- retain observation in `raw_observations`,
- do not promote to canonical claim.

## 6) Anti-Triviality Policy (strategy-level)

Promotion from observation -> canonical claim requires at least one strong non-trivial signal:
- new relation with specific semantic content,
- numeric or bounded constraint,
- explicit temporal anchor (canonical date/range),
- contradiction resolution potential,
- cross-document synthesis not present in any single source.

Generic statements without specific informational gain remain observations only.

## 7) Reflection and Dream Strategy

Reflection and dream should be trigger-driven, not cadence-driven.

Trigger examples:
- contradiction delta above threshold,
- temporal inconsistency delta above threshold,
- uncertainty concentration increase,
- novelty/info-gain increase in recent ingest window.

No trigger:
- skip heavy dream/hypothesis generation.

## 8) Model Routing Strategy

Use model tiers by task:
- cheap/local model: extraction normalization, candidate classification pre-checks.
- stronger model: only for gray-zone precision judgments and final hypothesis refinement.

Escalate model only when:
- low margin between top candidate classes,
- contradiction risk high,
- temporal implications ambiguous.

## 9) Metrics and Gates

Primary quality metrics:
- `non_trivial_claim_rate`
- `provenance_completeness_rate`
- `merge_precision`
- `contradiction_precision`
- `temporal_implication_count`
- `testable_hypothesis_rate`
- `insight_yield_per_doc`

Operational metrics:
- `avg_candidates_per_observation`
- `llm_escalation_rate`
- `token_cost_per_doc`
- end-to-end ingest latency

Suggested promotion gates:
- provenance completeness >= 0.98
- merge precision >= 0.90
- contradiction precision >= 0.85
- testable hypothesis rate >= 0.70

## 10) Rollout Plan

Phase 1: Data Model and Contracts
- Introduce canonical claim identity and provenance required fields.
- Add observation/canonical separation and migration adapter.

Phase 2: Decision Pipeline
- Implement candidate recall + precision judge + canonical update actions.
- Add dedupe collapse in retrieval by `canonical_claim_id`.

Phase 3: Anti-Triviality Enforcement
- Activate promotion gate from observations to canonical claims.
- Track blocked trivial claims for audit.

Phase 4: Triggered Reflection/Dream
- Replace fixed cadence with signal-driven orchestration.
- Add budget constraints and model routing thresholds.

Phase 5: Evaluation and Tightening
- Run 5-doc and 20-doc benchmark cycles.
- Tune thresholds by measured precision and insight gain.

## 11) Immediate Implementation Tasks

1. Define canonical claim schema and relation labels in code types.
2. Add provenance validator and rejection path to observation-only storage.
3. Add candidate recall API returning scored merge candidates.
4. Add precision judge interface with strict structured output.
5. Add canonical update transaction (`MERGE`, `CONTRADICTION`, `RELATED`, `NEW`).
6. Add retrieval collapse by `canonical_claim_id`.
7. Add metric emission and dashboard report for strategy gates.
