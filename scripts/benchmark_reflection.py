#!/usr/bin/env python3
"""Run reflection benchmark cycles and emit timing JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from silicon_memory.llm.config import LLMConfig
from silicon_memory.llm.dev import CachedLLM, MockLLM
from silicon_memory.llm.model_policy import is_weak_extraction_model_name
from silicon_memory.llm.preflight import ensure_local_chat_model_ready
from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.security.types import UserContext


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    user = UserContext(user_id=args.user_id, tenant_id=args.tenant_id)
    allow_weak_extraction_model = bool(
        getattr(args, "allow_weak_extraction_model", False),
    )
    llm_preflight_timeout_s = float(getattr(args, "llm_preflight_timeout_s", 8.0))
    if (
        args.llm_mode == "local"
        and is_weak_extraction_model_name(args.llm_model)
        and not allow_weak_extraction_model
    ):
        raise RuntimeError(
            "Extraction policy: qwen3-4b-class models are disabled for extraction. "
            "Use qwen3-30b or qwen3-80b, or pass --allow-weak-extraction-model "
            "to override for diagnostics only.",
        )

    if args.llm_mode == "local":
        await ensure_local_chat_model_ready(
            base_url=args.llm_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            timeout_s=llm_preflight_timeout_s,
        )

    if args.llm_mode == "mock":
        llm: Any = MockLLM()
    else:
        llm = SiliconLLMProvider(
            LLMConfig(
                base_url=args.llm_url,
                model=args.llm_model,
                api_key=args.llm_api_key,
            )
        )

    if args.llm_cache_mode != "off":
        llm = CachedLLM(
            llm=llm,
            cache_dir=args.llm_cache_dir,
            mode=args.llm_cache_mode,
        )
    memory = SiliconMemory(
        path=Path(args.db_path),
        user_context=user,
        db_grpc_host=args.db_grpc_host,
        db_grpc_port=args.db_grpc_port,
        db_retry_attempts=args.db_retry_attempts,
        db_request_timeout_s=args.db_request_timeout_s,
    )
    from silicon_memory.reflection.types import ReflectionConfig
    config = ReflectionConfig(
        allow_weak_extraction_model=allow_weak_extraction_model,
        dream_enable_hypothesis_generation=not args.disable_hypothesis_generation,
        dream_enable_procedure_detection=args.dream_enable_procedure_detection,
        dream_enable_question_generation=args.dream_enable_question_generation,
        dream_enable_entity_consolidation=args.dream_enable_entity_consolidation,
        dream_enable_predicate_consolidation=args.dream_enable_predicate_consolidation,
        dream_enable_hypothesis_validation=not args.disable_hypothesis_validation,
        dream_max_hypothesis_validations=args.dream_max_hypothesis_validations,
        dream_graph_call_timeout_s=float(getattr(args, "dream_graph_call_timeout_s", 10.0)),
    )
    engine = ReflectionEngine(
        memory=memory,
        llm=llm,
        config=config,
        extraction_cache_dir=args.extraction_cache_dir,
    )

    runs = []
    retrieval_sanity: dict[str, Any] = {}
    retrieval_errors: dict[str, str] = {}
    retrieval_timeout_s = float(getattr(args, "retrieval_sanity_timeout_s", 10.0))
    dream_timeout_s = float(getattr(args, "dream_timeout_s", 180.0))

    async def _probe_async(name: str, awaitable: Any, default: Any) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=retrieval_timeout_s)
        except Exception as exc:
            retrieval_errors[name] = str(exc)
            return default

    try:
        for i in range(args.cycles):
            t0 = time.monotonic()
            result = await engine.reflect(max_experiences=args.max_experiences, auto_commit=True)
            elapsed = time.monotonic() - t0
            runs.append(
                {
                    "cycle": i + 1,
                    "elapsed_s": round(elapsed, 4),
                    "experiences_processed": result.experiences_processed,
                    "new_beliefs": len(result.new_beliefs),
                    "updated_beliefs": len(result.updated_beliefs),
                    "timings": result.timings,
                }
            )

        dream_stats = None
        if args.include_dream:
            t0 = time.monotonic()
            try:
                dream_stats = await asyncio.wait_for(
                    engine.dream(),
                    timeout=dream_timeout_s,
                )
            except Exception as exc:
                dream_stats = {"error": str(exc), "timed_out": True}
            dream_stats["elapsed_s"] = round(time.monotonic() - t0, 4)

        retrieval_sanity["extraction_progress"] = await _probe_async(
            "count_extraction_progress",
            memory.count_extraction_progress(),
            {},
        )
        recent_limit = min(500, max(100, int(args.max_experiences) * 2))
        backlog_limit = min(2000, max(500, int(args.max_experiences) * 5))
        recent = await _probe_async(
            "get_recent_experiences",
            memory.get_recent_experiences(hours=24 * 365, limit=recent_limit),
            [],
        )
        retrieval_sanity["recent_experiences_count"] = len(recent)
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
                procedures = await memory.find_applicable_procedures("all procedures", limit=500)
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

        try:
            if hasattr(memory._backend, "get_beliefs_by_tag"):
                hypotheses = await _probe_async(
                    "get_beliefs_by_tag:hypothesis",
                    memory._backend.get_beliefs_by_tag(
                        "hypothesis",
                        limit=500,
                        min_confidence=0.0,
                    ),
                    [],
                )
                retrieval_sanity["hypothesis_beliefs_count"] = len(hypotheses)
            else:
                retrieval_sanity["hypothesis_beliefs_count"] = 0
        except Exception:
            retrieval_sanity["hypothesis_beliefs_count"] = 0
        retrieval_sanity["timeout_s"] = retrieval_timeout_s
        retrieval_sanity["errors"] = retrieval_errors
    finally:
        memory.close()

    return {
        "config": {
            "cycles": args.cycles,
            "max_experiences": args.max_experiences,
            "include_dream": args.include_dream,
            "dream_timeout_s": dream_timeout_s,
            "db_grpc_host": args.db_grpc_host,
            "db_grpc_port": args.db_grpc_port,
            "llm_mode": args.llm_mode,
            "llm_model": args.llm_model if args.llm_mode == "local" else "mock",
            "allow_weak_extraction_model": allow_weak_extraction_model,
            "llm_cache_mode": args.llm_cache_mode,
            "llm_cache_dir": args.llm_cache_dir,
            "llm_preflight_timeout_s": llm_preflight_timeout_s,
            "retrieval_sanity_timeout_s": retrieval_timeout_s,
            "dream_enable_procedure_detection": args.dream_enable_procedure_detection,
            "dream_enable_question_generation": args.dream_enable_question_generation,
            "dream_enable_entity_consolidation": args.dream_enable_entity_consolidation,
            "dream_enable_predicate_consolidation": args.dream_enable_predicate_consolidation,
            "dream_enable_hypothesis_generation": not args.disable_hypothesis_generation,
            "dream_enable_hypothesis_validation": not args.disable_hypothesis_validation,
            "dream_max_hypothesis_validations": args.dream_max_hypothesis_validations,
            "dream_graph_call_timeout_s": float(
                getattr(args, "dream_graph_call_timeout_s", 10.0),
            ),
        },
        "runs": runs,
        "dream": dream_stats,
        "retrieval_sanity": retrieval_sanity,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default="silicon_memory.db")
    p.add_argument("--db-grpc-host", default="127.0.0.1")
    p.add_argument("--db-grpc-port", type=int, default=8643)
    p.add_argument("--db-retry-attempts", type=int, default=2)
    p.add_argument("--db-request-timeout-s", type=float, default=10.0)
    p.add_argument("--user-id", default="benchmark-user")
    p.add_argument("--tenant-id", default="benchmark-tenant")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--max-experiences", type=int, default=100)
    p.add_argument("--include-dream", action="store_true")
    p.add_argument("--dream-timeout-s", type=float, default=180.0)
    p.add_argument("--dream-enable-procedure-detection", action="store_true")
    p.add_argument("--dream-enable-question-generation", action="store_true")
    p.add_argument("--dream-enable-entity-consolidation", action="store_true")
    p.add_argument("--dream-enable-predicate-consolidation", action="store_true")
    p.add_argument("--disable-hypothesis-generation", action="store_true")
    p.add_argument("--disable-hypothesis-validation", action="store_true")
    p.add_argument("--dream-max-hypothesis-validations", type=int, default=50)
    p.add_argument("--dream-graph-call-timeout-s", type=float, default=10.0)
    p.add_argument("--extraction-cache-dir", default=None)
    p.add_argument("--llm-mode", choices=["local", "mock"], default="local")
    p.add_argument("--llm-url", default="http://localhost:8000/v1")
    p.add_argument("--llm-model", default="qwen3-30b")
    p.add_argument("--llm-api-key", default="not-needed")
    p.add_argument("--llm-preflight-timeout-s", type=float, default=8.0)
    p.add_argument(
        "--allow-weak-extraction-model",
        action="store_true",
        help="Allow qwen3-4b-class local extraction models (diagnostics only).",
    )
    p.add_argument(
        "--llm-cache-mode",
        choices=["off", "record_replay", "replay_only"],
        default="record_replay",
    )
    p.add_argument("--llm-cache-dir", default="export/metrics/llm_cache/benchmark")
    p.add_argument("--retrieval-sanity-timeout-s", type=float, default=10.0)
    p.add_argument("--output", default="export/metrics/reflection_benchmark.json")
    args = p.parse_args()

    out = asyncio.run(_run(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
