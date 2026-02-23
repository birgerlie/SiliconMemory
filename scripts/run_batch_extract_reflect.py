#!/usr/bin/env python3
"""Extract from a batch of docs, commit richer triplets, then run reflection."""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from anthropic import AsyncAnthropic

from silicon_memory.core.types import Belief, Experience, Source, SourceType, Triplet
from silicon_memory.core.utils import utc_now
from silicon_memory.entities import EntityCache, EntityResolver, RuleEngine
from silicon_memory.entities.date_normalizer import normalize_date
from silicon_memory.entities.types import DetectorRule, ExtractorRule
from silicon_memory.llm.config import LLMConfig
from silicon_memory.llm.dev import CachedLLM, MockLLM
from silicon_memory.llm.model_policy import is_weak_extraction_model_name
from silicon_memory.llm.preflight import ensure_local_chat_model_ready
from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.llm_extractor import LLMPatternExtractor
from silicon_memory.reflection.types import PatternType, ReflectionConfig
from silicon_memory.security.types import UserContext


def _log(message: str) -> None:
    print(f"[run_batch_extract_reflect] {message}", flush=True)


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-latest") -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        req: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            req["system"] = system
        msg = await self._client.messages.create(**req)
        return "".join(
            getattr(block, "text", "")
            for block in msg.content
            if getattr(block, "text", None)
        )

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Compatibility method expected by ingestion adapters."""
        return await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        max_tokens: int | None = None,
    ) -> Any:
        schema_dict = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else {"type": "object"}
        )
        sys_prompt = (
            "Return only valid JSON matching this schema. "
            "No markdown or extra text.\n"
            + json.dumps(schema_dict)
        )
        raw = await self.generate(
            sys_prompt + "\n\n" + prompt,
            max_tokens=max_tokens or 3072,
            temperature=0.2,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            ).strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(match.group(0) if match else raw)
        return schema.model_validate(parsed) if hasattr(schema, "model_validate") else parsed


def _load_rules(path: Path) -> tuple[list[DetectorRule], list[ExtractorRule]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    detectors: list[DetectorRule] = []
    extractors: list[ExtractorRule] = []
    for d in data.get("detectors", []):
        if d.get("id") and d.get("pattern"):
            detectors.append(
                DetectorRule(
                    id=d["id"],
                    pattern=d["pattern"],
                    description=d.get("description") or d["id"],
                )
            )
    for e in data.get("extractors", []):
        if e.get("id") and e.get("pattern") and e.get("entity_type"):
            extractors.append(
                ExtractorRule(
                    id=e["id"],
                    entity_type=e["entity_type"],
                    detector_ids=e.get("detector_ids", []),
                    pattern=e["pattern"],
                    normalize_template=e.get("normalize_template", "{match}"),
                    confidence=float(e.get("confidence", 1.0) or 1.0),
                )
            )
    return detectors, extractors


def _load_docs(doc_glob: str, max_docs: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted(glob.glob(doc_glob))[:max_docs]:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        text = (obj.get("full_text") or "").strip()
        if len(text) < 80:
            continue
        out.append((Path(path).stem, text))
    return out


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = utc_now()

    isolate_run = bool(getattr(args, "isolate_run", False))
    skip_retrieval_sanity = bool(getattr(args, "skip_retrieval_sanity", False))
    allow_weak_extraction_model = bool(
        getattr(args, "allow_weak_extraction_model", False),
    )
    llm_preflight_timeout_s = float(getattr(args, "llm_preflight_timeout_s", 8.0))

    if args.llm_mode == "local" and is_weak_extraction_model_name(args.llm_model):
        if not allow_weak_extraction_model:
            raise RuntimeError(
                "Extraction policy: qwen3-4b-class models are disabled for extraction. "
                "Use qwen3-30b or qwen3-80b, or pass --allow-weak-extraction-model "
                "to override for diagnostics only.",
            )
        _log(
            "Extraction policy override enabled for weak model "
            f"({args.llm_model})",
        )

    if args.llm_mode == "local":
        _log(
            "Local LLM preflight "
            f"(model={args.llm_model}, url={args.llm_url}, timeout={llm_preflight_timeout_s}s)"
        )
        await ensure_local_chat_model_ready(
            base_url=args.llm_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            timeout_s=llm_preflight_timeout_s,
        )

    _log("Loading documents")
    docs = _load_docs(args.docs_glob, args.max_docs)
    if not docs:
        raise RuntimeError("No usable docs found")
    _log(f"Loaded {len(docs)} documents")

    effective_tenant_id = args.tenant_id
    effective_user_id = args.user_id
    if isolate_run:
        isolate_suffix = uuid4().hex[:10]
        effective_tenant_id = f"{args.tenant_id}-{isolate_suffix}"
        effective_user_id = f"{args.user_id}-{isolate_suffix}"
        _log(
            "Run isolation enabled "
            f"(tenant_id={effective_tenant_id}, user_id={effective_user_id})"
        )

    user = UserContext(user_id=effective_user_id, tenant_id=effective_tenant_id)
    memory = SiliconMemory(
        path=Path(args.db_path),
        user_context=user,
        db_grpc_host=args.db_grpc_host,
        db_grpc_port=args.db_grpc_port,
    )

    rules = RuleEngine()
    resolver = EntityResolver(cache=EntityCache(), rules=rules)
    d_rules, e_rules = _load_rules(Path(args.rules_json))
    for d in d_rules:
        rules.add_detector(d)
    for e in e_rules:
        rules.add_extractor(e)

    if args.llm_mode == "mock":
        llm: Any = MockLLM()
    elif args.llm_mode == "local":
        llm = SiliconLLMProvider(
            LLMConfig(
                base_url=args.llm_url,
                model=args.llm_model,
                api_key=args.llm_api_key,
            )
        )
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and args.llm_cache_mode != "replay_only":
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing in environment "
                "(required unless --llm-cache-mode replay_only)",
            )
        llm = AnthropicLLM(
            api_key=api_key or "not-needed",
            model=args.llm_model,
        )

    if args.llm_cache_mode != "off":
        llm = CachedLLM(
            llm=llm,
            cache_dir=args.llm_cache_dir,
            mode=args.llm_cache_mode,
        )

    exps = [
        Experience(
            id=uuid4(),
            content=text,
            context={"document_id": doc_id, "title": doc_id},
            processed=False,
        )
        for doc_id, text in docs
    ]
    ingest_visibility_ok: bool | None = None
    ingested_experience_ids: list[str] = []
    ingest_batch_state: str | None = None
    ingest_batch_failed_count = 0
    ingest_batch_errors: list[str] = []
    if args.record_experiences:
        _log(
            "Ingesting experiences "
            f"(count={len(exps)}, wait_for_visibility=True, timeout={args.ingest_visibility_timeout_s}s)"
        )
        ingest_batch = await memory.ingest_experiences_batch(
            exps,
            wait_for_visibility=True,
            timeout_s=args.ingest_visibility_timeout_s,
            poll_interval_s=args.ingest_visibility_poll_s,
        )
        ingest_visibility_ok = ingest_batch.visibility_ok
        ingest_batch_state = ingest_batch.state
        ingest_batch_failed_count = ingest_batch.receipt.failed_count
        ingest_batch_errors = list(ingest_batch.receipt.errors)
        ingested_experience_ids = list(ingest_batch.receipt.accepted_experience_ids)
        _log(
            "Ingest completed "
            f"(state={ingest_batch_state}, accepted={len(ingested_experience_ids)}, failed={ingest_batch_failed_count})"
        )

    reflection_config = ReflectionConfig(
        llm_temperature=args.llm_temperature,
        extraction_max_chars=args.extraction_max_chars,
        extraction_max_items=args.extraction_max_items,
        extraction_max_tokens=args.extraction_max_tokens,
        allow_weak_extraction_model=allow_weak_extraction_model,
        empty_extracted_cooldown_s=0.0,
        dream_enable_hypothesis_generation=not args.disable_hypothesis_generation,
        dream_enable_procedure_detection=args.dream_enable_procedure_detection,
        dream_enable_question_generation=args.dream_enable_question_generation,
        dream_enable_entity_consolidation=args.dream_enable_entity_consolidation,
        dream_enable_predicate_consolidation=args.dream_enable_predicate_consolidation,
        dream_enable_hypothesis_validation=not args.disable_hypothesis_validation,
        dream_max_hypothesis_validations=args.dream_max_hypothesis_validations,
        dream_graph_call_timeout_s=float(getattr(args, "dream_graph_call_timeout_s", 10.0)),
    )

    extractor = LLMPatternExtractor(
        memory=memory,
        llm=llm,
        config=reflection_config,
        cache_dir=args.extraction_cache_dir,
        resolver=resolver,
    )
    _log("Starting extraction")
    patterns = await extractor.extract_patterns_flat(exps)
    _log(f"Extraction completed (patterns={len(patterns)})")
    extraction_diagnostics: dict[str, Any] = {}
    diag_fn = getattr(extractor, "extraction_diagnostics", None)
    if callable(diag_fn):
        try:
            extraction_diagnostics = dict(diag_fn())
        except Exception:
            extraction_diagnostics = {}
    if int(extraction_diagnostics.get("limit_hit_chunks_total", 0) or 0) > 0:
        _log(
            "Extraction cap hit in one or more chunks; consider raising "
            "--extraction-max-items / --extraction-max-tokens."
        )
    by_type = Counter(p.type.value for p in patterns)
    rel_dir = Counter(
        (p.context or {}).get("relationship_direction")
        for p in patterns
        if p.type == PatternType.RELATIONSHIP
    )
    committed_predicates = Counter()

    committed_triplets = 0
    committed_docs = 0
    committed_external_ids: list[str] = []
    extraction_run_id = str(uuid4())
    extracted_at = utc_now().isoformat()
    total_patterns = len(patterns)
    for idx, p in enumerate(patterns, start=1):
        context = dict(p.context or {})
        context["pattern_type"] = p.type.value
        context.setdefault("source_type", "document")
        context.setdefault("extraction_run_id", extraction_run_id)
        context.setdefault("extractor_model", args.llm_model if args.llm_mode in {"anthropic", "local"} else "mock")
        context.setdefault("extractor_version", "run_batch_extract_reflect_v1")
        context.setdefault("extracted_at", extracted_at)
        context.setdefault(
            "confidence_basis",
            "llm" if args.llm_mode in {"anthropic", "local"} else "rule",
        )
        context.setdefault("evidence_span", str(p.id))
        triplet_subject = p.subject
        triplet_predicate = p.predicate
        triplet_object = p.object

        is_triplet = (
            p.type in {PatternType.FACT, PatternType.RELATIONSHIP}
            and bool(triplet_subject and triplet_predicate and triplet_object)
        )
        if p.type == PatternType.TIMELINE_EVENT and p.subject:
            date_meta = (p.context or {}).get("date_normalized") or {}
            canonical_date = str(date_meta.get("canonical") or "").strip()
            if not canonical_date:
                source_doc = (p.context or {}).get("source") or {}
                reference_datetime = ""
                if isinstance(source_doc, dict):
                    reference_datetime = str(source_doc.get("occurred_at") or "")
                context_raw_date = str((p.context or {}).get("date") or "").strip()
                for raw_date in (context_raw_date, p.object, p.description):
                    if not raw_date:
                        continue
                    normalized = normalize_date(
                        raw_date,
                        reference_datetime=reference_datetime or None,
                    )
                    if normalized and not normalized.ambiguous and not normalized.relative:
                        canonical_date = normalized.canonical
                        context["date_normalized"] = normalized.__dict__
                        break
            if canonical_date:
                is_triplet = True
                triplet_subject = p.subject
                triplet_predicate = "occurred_on"
                triplet_object = canonical_date
                context["event_text"] = p.object or p.description

        belief = Belief(
            id=uuid4(),
            content=p.description,
            triplet=(
                Triplet(triplet_subject, triplet_predicate, triplet_object)
                if is_triplet and triplet_subject and triplet_predicate and triplet_object
                else None
            ),
            confidence=p.confidence,
            source=Source(
                id="batch_extract_reflect",
                type=SourceType.REFLECTION,
                reliability=0.7,
                metadata=context,
            ),
            tags={"extracted", "run_batch"},
            metadata=context,
        )
        await memory.commit_belief(belief)
        committed_external_ids.append(
            f"{effective_tenant_id}/{effective_user_id}/belief-{belief.id}"
        )
        if is_triplet:
            committed_triplets += 1
            if triplet_predicate:
                committed_predicates[triplet_predicate] += 1
        else:
            committed_docs += 1
        if idx == total_patterns or idx % 25 == 0:
            _log(
                "Belief commit progress "
                f"({idx}/{total_patterns}, triplets={committed_triplets}, docs={committed_docs})"
            )

    if args.record_experiences and exps:
        _log(f"Marking experiences extracted (count={len(exps)})")
        await memory.mark_experiences_extracted([exp.id for exp in exps])

    engine = ReflectionEngine(
        memory=memory,
        llm=llm,
        config=reflection_config,
        resolver=resolver,
        extraction_cache_dir=args.extraction_cache_dir,
    )
    _log(f"Running reflection (limit={args.reflect_limit})")
    reflection = await engine.reflect(max_experiences=args.reflect_limit, auto_commit=True)
    _log(
        "Reflection completed "
        f"(experiences_processed={reflection.experiences_processed}, updated_beliefs={len(reflection.updated_beliefs)})"
    )
    if args.record_experiences and exps and reflection.experiences_processed > 0:
        _log(f"Marking experiences processed (count={len(exps)})")
        await memory.mark_experiences_processed([exp.id for exp in exps])

    committed_external_id_set = set(committed_external_ids)
    triple_metadata_by_external_id: dict[str, dict[str, Any]] = {}
    try:
        triple_scan = memory._backend._query_triples(k=max(5000, len(committed_external_ids) * 6))
    except Exception:
        triple_scan = []
    for triple in triple_scan:
        ext_id = memory._backend._rget(triple, "external_id", "")
        if not ext_id or ext_id not in committed_external_id_set:
            continue
        metadata = memory._backend._rget(triple, "metadata") or {}
        if isinstance(metadata, dict):
            triple_metadata_by_external_id[ext_id] = dict(metadata)

    def _load_committed_metadata(external_id: str) -> dict[str, Any]:
        doc = memory._backend._db.get(external_id)
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict) and metadata:
                return metadata
        return triple_metadata_by_external_id.get(external_id, {})

    reflected_count = 0
    for ext_id in committed_external_ids:
        metadata = _load_committed_metadata(ext_id)
        if metadata.get("reflection_processed"):
            reflected_count += 1

    claim_relation_counts: Counter[str] = Counter()
    unknown_reason_counts: Counter[str] = Counter()
    valid_claim_labels = {"MERGE", "CONTRADICTION", "RELATED", "NEW"}
    canonical_claim_ids: set[str] = set()
    provenance_complete = 0
    canonical_promoted = 0
    non_trivial = 0
    unknown_observations = 0
    claim_candidate_total = 0
    llm_escalations = 0
    temporal_implication_count = 0
    hypothesis_total = 0
    testable_hypotheses = 0

    def _normalize_tags(raw: Any) -> set[str]:
        if isinstance(raw, (list, tuple, set)):
            return {str(item).strip().lower() for item in raw if str(item).strip()}
        if isinstance(raw, str):
            text = raw.strip()
            if text:
                return {text.lower()}
        return set()

    for ext_id in committed_external_ids:
        metadata = _load_committed_metadata(ext_id)
        raw_relation = str(metadata.get("claim_relation_label") or "").strip().upper()
        relation = raw_relation if raw_relation in valid_claim_labels else "NEW"
        unknown_flag = bool(metadata.get("claim_relation_unknown", False))
        if raw_relation not in valid_claim_labels:
            unknown_flag = True
        if unknown_flag:
            unknown_observations += 1
            unknown_reason = str(
                metadata.get("claim_relation_unknown_reason")
                or ("missing_label" if not raw_relation else f"invalid_input_label:{raw_relation}"),
            ).strip() or "unknown"
            unknown_reason_counts[unknown_reason] += 1
        claim_relation_counts[relation] += 1
        canonical_claim_id = str(
            metadata.get("canonical_claim_id")
            or metadata.get("belief_id")
            or "",
        ).strip()
        if canonical_claim_id:
            canonical_claim_ids.add(canonical_claim_id)
        if bool(metadata.get("provenance_complete", False)):
            provenance_complete += 1
        if bool(metadata.get("canonical_promotion_allowed", True)):
            canonical_promoted += 1
        try:
            if float(metadata.get("anti_triviality_score", 0.0) or 0.0) > 0.0:
                non_trivial += 1
        except Exception:
            pass
        try:
            claim_candidate_total += max(0, int(metadata.get("claim_candidate_count", 0) or 0))
        except Exception:
            pass
        if str(metadata.get("claim_decision_via") or "").strip().lower() == "llm":
            llm_escalations += 1

        tags = _normalize_tags(metadata.get("tags"))
        source_id = str(metadata.get("source_id") or "").strip().lower()
        is_hypothesis = "hypothesis" in tags or source_id == "hypothesis_generation"
        if is_hypothesis:
            hypothesis_total += 1
            testable = any(
                bool(metadata.get(key))
                for key in (
                    "hypothesis_testable",
                    "testable",
                    "falsifiable",
                    "validation_query",
                    "test_question",
                    "disconfirming_evidence_needed",
                )
            )
            if testable:
                testable_hypotheses += 1

        triplet = metadata.get("triplet")
        triplet_predicate = ""
        if isinstance(triplet, dict):
            triplet_predicate = str(triplet.get("predicate") or "").strip().lower()
        if (
            triplet_predicate == "occurred_on"
            or str(metadata.get("observed_at") or "").strip()
            or str(metadata.get("valid_from") or "").strip()
            or str(metadata.get("valid_until") or "").strip()
        ):
            temporal_implication_count += 1

    total_claim_observations = max(1, len(committed_external_ids))
    provenance_completeness_rate = round(provenance_complete / total_claim_observations, 4)
    non_trivial_claim_rate = round(non_trivial / total_claim_observations, 4)
    merge_decision_rate = round(claim_relation_counts.get("MERGE", 0) / total_claim_observations, 4)
    contradiction_decision_rate = round(
        claim_relation_counts.get("CONTRADICTION", 0) / total_claim_observations,
        4,
    )
    avg_candidates_per_observation = round(claim_candidate_total / total_claim_observations, 4)
    llm_escalation_rate = round(llm_escalations / total_claim_observations, 4)
    testable_hypothesis_rate = (
        round(testable_hypotheses / hypothesis_total, 4)
        if hypothesis_total > 0
        else None
    )
    claim_strategy = {
        "observations_total": len(committed_external_ids),
        "canonical_claim_count": len(canonical_claim_ids),
        "claim_relation_counts": dict(claim_relation_counts),
        "unknown_observations": unknown_observations,
        "unknown_reason_counts": dict(unknown_reason_counts),
        "provenance_completeness_rate": provenance_completeness_rate,
        "non_trivial_claim_rate": non_trivial_claim_rate,
        "insight_yield_per_doc": round(canonical_promoted / max(1, len(docs)), 4),
        "merge_decision_rate": merge_decision_rate,
        "contradiction_decision_rate": contradiction_decision_rate,
        "temporal_implication_count": temporal_implication_count,
        "testable_hypothesis_rate": testable_hypothesis_rate,
        "avg_candidates_per_observation": avg_candidates_per_observation,
        "llm_escalation_rate": llm_escalation_rate,
        # True precision requires labelled gold comparisons.
        "merge_precision": None,
        "contradiction_precision": None,
        "gates": {
            "provenance_completeness_gte_0_98": provenance_completeness_rate >= 0.98,
            "merge_precision_gte_0_90": None,
            "contradiction_precision_gte_0_85": None,
            "testable_hypothesis_rate_gte_0_70": (
                testable_hypothesis_rate is not None and testable_hypothesis_rate >= 0.70
            ),
        },
    }

    retrieval_sanity: dict[str, Any] = {}
    retrieval_errors: dict[str, str] = {}
    retrieval_timeout_s = float(getattr(args, "retrieval_sanity_timeout_s", 10.0))

    async def _probe_async(name: str, awaitable: Any, default: Any) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=retrieval_timeout_s)
        except Exception as exc:
            retrieval_errors[name] = str(exc)
            return default

    if skip_retrieval_sanity:
        _log("Skipping retrieval sanity checks (--skip-retrieval-sanity)")
        retrieval_sanity = {"skipped": True}
    else:
        _log("Running retrieval sanity checks")
        retrieval_sanity["timeout_s"] = retrieval_timeout_s
        backlog_limit = max(500, len(docs) * 50)
        retrieval_sanity["extraction_progress"] = await _probe_async(
            "count_extraction_progress",
            memory.count_extraction_progress(),
            {},
        )

        recent = await _probe_async(
            "get_recent_experiences",
            memory.get_recent_experiences(hours=24 * 365, limit=max(200, len(docs) * 10)),
            [],
        )
        retrieval_sanity["recent_experiences_count"] = len(recent)

        direct_visible = 0
        for exp in exps:
            loaded = await _probe_async(
                f"get_experience:{exp.id}",
                memory.get_experience(exp.id),
                None,
            )
            if loaded is not None:
                direct_visible += 1
        retrieval_sanity["direct_experience_visibility"] = {
            "expected": len(exps),
            "visible": direct_visible,
        }

        unprocessed = await _probe_async(
            "get_unprocessed_experiences",
            memory._backend.get_unprocessed_experiences(limit=backlog_limit),
            [],
        )
        retrieval_sanity["unprocessed_experiences_count"] = len(unprocessed)

        unextracted = await _probe_async(
            "get_unextracted_experiences",
            memory.get_unextracted_experiences(limit=backlog_limit),
            [],
        )
        retrieval_sanity["unextracted_experiences_count"] = len(unextracted)

        try:
            backend = memory._backend
            if hasattr(backend, "_search_by_type"):
                seen_ids: set[str] = set()
                procedure_queries = ("Step", "playbook", "workflow", "procedure", "runbook")
                for probe in procedure_queries:
                    raw = await _probe_async(
                        f"_search_by_type:{probe}",
                        asyncio.to_thread(
                            partial(
                                backend._search_by_type,
                                query=probe,
                                node_types={backend.NODE_TYPE_PROCEDURE},
                                target=400,
                            )
                        ),
                        [],
                    )
                    for item in raw:
                        ext_id = backend._rget(item, "external_id", "")
                        if not ext_id or ext_id in seen_ids:
                            continue
                        if not backend._can_access(
                            (backend._rget(item, "metadata") or {}),
                            _external_id=ext_id,
                        ):
                            continue
                        seen_ids.add(ext_id)
                retrieval_sanity["procedures_count"] = len(seen_ids)
            else:
                procedures = await _probe_async(
                    "find_applicable_procedures",
                    memory.find_applicable_procedures("all procedures", limit=500),
                    [],
                )
                retrieval_sanity["procedures_count"] = len(procedures)
        except Exception:
            retrieval_sanity["procedures_count"] = 0

        working = await _probe_async(
            "get_all_context",
            memory.get_all_context(),
            {},
        )
        retrieval_sanity["working_keys_count"] = len(working)
        retrieval_sanity["open_questions_count"] = sum(
            1 for key in working if key.startswith("open_question_")
        )
        retrieval_sanity["errors"] = retrieval_errors

    memory.close()
    run_finished = utc_now()
    duration_s = (run_finished - run_started).total_seconds()
    _log(f"Run completed in {duration_s:.2f}s")

    return {
        "docs_used": [d[0] for d in docs],
        "doc_count": len(docs),
        "record_experiences": args.record_experiences,
        "ingest_visibility_ok": ingest_visibility_ok,
        "ingest_batch_state": ingest_batch_state,
        "ingest_batch_failed_count": ingest_batch_failed_count,
        "ingest_batch_errors": ingest_batch_errors,
        "ingested_experience_count": len(ingested_experience_ids),
        "llm_mode": args.llm_mode,
        "llm_cache_mode": args.llm_cache_mode,
        "llm_cache_dir": args.llm_cache_dir,
        "llm_preflight_timeout_s": llm_preflight_timeout_s,
        "extraction_limits": {
            "max_chars": args.extraction_max_chars,
            "max_items": args.extraction_max_items,
            "max_tokens": args.extraction_max_tokens,
        },
        "extraction_diagnostics": extraction_diagnostics,
        "llm_model": args.llm_model if args.llm_mode in {"anthropic", "local"} else "mock",
        "effective_tenant_id": effective_tenant_id,
        "effective_user_id": effective_user_id,
        "patterns_total": len(patterns),
        "patterns_by_type": dict(by_type),
        "relationship_direction_counts": dict(rel_dir),
        "committed_triplet_beliefs": committed_triplets,
        "committed_triplet_predicates": dict(committed_predicates),
        "committed_document_beliefs": committed_docs,
        "reflection_processed_belief_docs": reflected_count,
        "claim_strategy": claim_strategy,
        "reflection": {
            "experiences_processed": reflection.experiences_processed,
            "updated_beliefs": len(reflection.updated_beliefs),
            "timings": reflection.timings,
        },
        "retrieval_sanity": retrieval_sanity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-glob", default="/Users/birger/code/epstein/epstein-docs/results/IMAGES*/DOJ-OGR-*.json")
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--db-path", default="silicon_memory.db")
    parser.add_argument("--db-grpc-host", default="127.0.0.1")
    parser.add_argument("--db-grpc-port", type=int, default=8643)
    parser.add_argument("--user-id", default="batch-triplet-user")
    parser.add_argument("--tenant-id", default="batch-triplet-tenant")
    parser.add_argument(
        "--isolate-run",
        action="store_true",
        help="Append a unique suffix to tenant/user IDs for deterministic isolated runs.",
    )
    parser.add_argument("--rules-json", default="/Users/birger/code/epstein/SiliconMemory/bootstrap_rules.json")
    parser.add_argument("--record-experiences", action="store_true", default=True)
    parser.add_argument("--no-record-experiences", action="store_false", dest="record_experiences")
    parser.add_argument(
        "--skip-retrieval-sanity",
        action="store_true",
        help="Skip retrieval sanity checks to speed up local iteration.",
    )
    parser.add_argument("--retrieval-sanity-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--allow-weak-extraction-model",
        action="store_true",
        help="Allow qwen3-4b-class local extraction models (diagnostics only).",
    )
    parser.add_argument("--ingest-visibility-timeout-s", type=float, default=5.0)
    parser.add_argument("--ingest-visibility-poll-s", type=float, default=0.1)
    parser.add_argument("--llm-model", default="claude-3-5-haiku-latest")
    parser.add_argument("--llm-temperature", type=float, default=0.3)
    parser.add_argument("--llm-mode", choices=["anthropic", "local", "mock"], default="anthropic")
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-api-key", default="not-needed")
    parser.add_argument("--llm-preflight-timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--llm-cache-mode",
        choices=["off", "record_replay", "replay_only"],
        default="record_replay",
    )
    parser.add_argument("--llm-cache-dir", default="export/metrics/llm_cache")
    parser.add_argument("--extraction-max-chars", type=int, default=36000)
    parser.add_argument("--extraction-max-items", type=int, default=12)
    parser.add_argument("--extraction-max-tokens", type=int, default=2500)
    parser.add_argument("--reflect-limit", type=int, default=300)
    parser.add_argument("--disable-hypothesis-generation", action="store_true")
    parser.add_argument("--dream-enable-procedure-detection", action="store_true")
    parser.add_argument("--dream-enable-question-generation", action="store_true")
    parser.add_argument("--dream-enable-entity-consolidation", action="store_true")
    parser.add_argument("--dream-enable-predicate-consolidation", action="store_true")
    parser.add_argument("--disable-hypothesis-validation", action="store_true")
    parser.add_argument("--dream-max-hypothesis-validations", type=int, default=50)
    parser.add_argument("--dream-graph-call-timeout-s", type=float, default=10.0)
    parser.add_argument("--extraction-cache-dir", default="export/metrics/run_batch_extraction_cache")
    parser.add_argument("--output", default="export/metrics/run_batch_extract_reflect.json")
    args = parser.parse_args()

    out = asyncio.run(_run(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
