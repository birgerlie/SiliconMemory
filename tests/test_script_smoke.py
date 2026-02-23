"""Smoke tests for batch/benchmark scripts without live DB/LLM."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID, uuid4

import pytest

from silicon_memory.core.types import Belief, Experience
from silicon_memory.reflection.types import Pattern, PatternType


def _load_script_module(name: str, rel_path: str) -> ModuleType:
    """Load a script file as a module for direct function testing."""
    if "anthropic" not in sys.modules:
        anthropic_stub = ModuleType("anthropic")

        class _AsyncAnthropic:  # pragma: no cover - import-time fallback only
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D401
                pass

        anthropic_stub.AsyncAnthropic = _AsyncAnthropic
        sys.modules["anthropic"] = anthropic_stub

    root = Path(__file__).resolve().parents[1]
    script_path = root / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDB:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def get(self, external_id: str) -> dict | None:
        return self.docs.get(external_id)


class _FakeBackend:
    NODE_TYPE_PROCEDURE = "procedure"

    def __init__(self, memory: "_FakeMemory") -> None:
        self._memory = memory
        self._db = _FakeDB()

    @staticmethod
    def _rget(obj: object, key: str, default: object = None) -> object:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _can_access(metadata: dict, _external_id: str = "") -> bool:  # noqa: ARG004
        return isinstance(metadata, dict)

    def _search_by_type(
        self,
        query: str,
        node_types: set[str],  # noqa: ARG002
        target: int,  # noqa: ARG002
    ) -> list[dict]:
        if query == "Step" and self._memory._procedures:  # noqa: SLF001
            return [
                {
                    "external_id": "t/u/procedure-1",
                    "metadata": {"owner_id": "u", "tenant_id": "t", "privacy_level": "private"},
                },
            ]
        return []

    async def get_unprocessed_experiences(self, limit: int = 10000) -> list[Experience]:
        out = [exp for exp in self._memory._experiences.values() if not exp.processed]
        return out[:limit]

    async def get_beliefs_by_tag(self, tag: str, limit: int = 500, min_confidence: float = 0.0) -> list[Belief]:  # noqa: ARG002
        return []


class _FakeMemory:
    def __init__(self, path: Path, user_context, **kwargs) -> None:  # noqa: ANN001, ARG002
        self._path = path
        self._user_context = user_context
        self._backend = _FakeBackend(self)
        self._experiences: dict[UUID, Experience] = {}
        self._working: dict[str, object] = {}
        self._procedures: list[str] = []

    def _external_id(self, entity_type: str, entity_id: UUID) -> str:
        return f"{self._user_context.tenant_id}/{self._user_context.user_id}/{entity_type}-{entity_id}"

    async def record_experience(self, experience: Experience) -> None:
        self._experiences[experience.id] = Experience(
            id=experience.id,
            content=experience.content,
            context=experience.context,
            processed=False,
        )

    async def wait_for_ingest_visibility(
        self,
        experience_ids: list[str],  # noqa: ARG002
        timeout_s: float = 5.0,  # noqa: ARG002
        poll_interval_s: float = 0.1,  # noqa: ARG002
    ) -> bool:
        return True

    async def ingest_experiences_batch(
        self,
        experiences: list[Experience],
        *,
        wait_for_visibility: bool = True,  # noqa: ARG002
        timeout_s: float = 5.0,  # noqa: ARG002
        poll_interval_s: float = 0.1,  # noqa: ARG002
    ) -> SimpleNamespace:
        ids: list[str] = []
        for exp in experiences:
            await self.record_experience(exp)
            ids.append(str(exp.id))
        return SimpleNamespace(
            visibility_ok=True,
            state="SUCCEEDED",
            receipt=SimpleNamespace(
                failed_count=0,
                errors=[],
                accepted_experience_ids=ids,
            ),
        )

    async def commit_belief(self, belief: Belief) -> None:
        ext_id = self._external_id("belief", belief.id)
        self._backend._db.docs[ext_id] = {"metadata": {"reflection_processed": False}}

    async def mark_experiences_extracted(self, experience_ids: list[UUID]) -> None:
        for eid in experience_ids:
            exp = self._experiences.get(eid)
            if exp:
                # Experience dataclass has no explicit extracted field; track in context.
                ctx = dict(exp.context or {})
                ctx["extracted"] = True
                exp.context = ctx

    async def mark_experiences_processed(self, experience_ids: list[UUID]) -> None:
        for eid in experience_ids:
            exp = self._experiences.get(eid)
            if exp:
                exp.processed = True

    async def count_extraction_progress(self) -> dict[str, int]:
        extracted = 0
        for exp in self._experiences.values():
            if bool((exp.context or {}).get("extracted")):
                extracted += 1
        total = len(self._experiences)
        return {"extracted": extracted, "unextracted": total - extracted, "total": total}

    async def get_recent_experiences(self, hours: int = 24, limit: int = 100) -> list[Experience]:  # noqa: ARG002
        return list(self._experiences.values())[:limit]

    async def get_experience(self, experience_id: UUID) -> Experience | None:
        return self._experiences.get(experience_id)

    async def get_unextracted_experiences(self, limit: int = 10000) -> list[Experience]:
        out = [exp for exp in self._experiences.values() if not bool((exp.context or {}).get("extracted"))]
        return out[:limit]

    async def get_all_context(self) -> dict[str, object]:
        return dict(self._working)

    async def set_context(self, key: str, value: object, ttl_seconds: int = 300) -> None:  # noqa: ARG002
        self._working[key] = value

    async def find_applicable_procedures(self, context: str, limit: int = 5) -> list[object]:  # noqa: ARG002
        return [SimpleNamespace(name=p) for p in self._procedures[:limit]]

    async def query_beliefs(self, query: str, limit: int = 10, min_confidence: float = 0.0) -> list[Belief]:  # noqa: ARG002
        return []

    def close(self) -> None:
        return


class _FakeExtractor:
    def __init__(self, memory: _FakeMemory, llm, config, cache_dir, resolver) -> None:  # noqa: ANN001, ARG002
        self._memory = memory

    async def extract_patterns_flat(self, exps: list[Experience]) -> list[Pattern]:
        if not exps:
            return []
        first = exps[0]
        return [
            Pattern(
                type=PatternType.FACT,
                description="Fact from doc",
                subject="A",
                predicate="relates_to",
                object="B",
                confidence=0.7,
                context={"source": {"document_id": str((first.context or {}).get("document_id", "d1"))}},
            ),
            Pattern(
                type=PatternType.TIMELINE_EVENT,
                description="Event dated",
                subject="Case 1",
                object="2024-03-12",
                confidence=0.7,
                context={"date": "2024-03-12"},
            ),
        ]


class _FakeEngine:
    def __init__(self, memory: _FakeMemory, llm, config, resolver=None, extraction_cache_dir=None, **kwargs) -> None:  # noqa: ANN001, ARG002
        self._memory = memory

    async def reflect(self, max_experiences: int, auto_commit: bool) -> SimpleNamespace:  # noqa: ARG002
        # Mark all belief docs as reflection processed.
        for doc in self._memory._backend._db.docs.values():
            meta = doc.get("metadata", {})
            meta["reflection_processed"] = True
            doc["metadata"] = meta
        return SimpleNamespace(experiences_processed=2, updated_beliefs=[], timings={"total": 0.1})

    async def dream(self) -> dict:
        self._memory._procedures.append("p1")
        await self._memory.set_context("open_question_x", {"question": "Q?"})
        return {"questions_generated": 1, "procedures_created": 1}


@pytest.mark.asyncio
async def test_run_batch_extract_reflect_smoke(monkeypatch) -> None:
    module = _load_script_module(
        "run_batch_extract_reflect_smoke_module",
        "scripts/run_batch_extract_reflect.py",
    )

    monkeypatch.setattr(module, "SiliconMemory", _FakeMemory)
    monkeypatch.setattr(module, "LLMPatternExtractor", _FakeExtractor)
    monkeypatch.setattr(module, "ReflectionEngine", _FakeEngine)
    monkeypatch.setattr(module, "_load_rules", lambda path: ([], []))
    monkeypatch.setattr(
        module,
        "_load_docs",
        lambda doc_glob, max_docs: [("doc1", "x" * 120), ("doc2", "y" * 120)][:max_docs],
    )

    args = argparse.Namespace(
        docs_glob="unused",
        max_docs=2,
        db_path="silicon_memory.db",
        db_grpc_host="127.0.0.1",
        db_grpc_port=8643,
        user_id="u",
        tenant_id="t",
        rules_json="unused.json",
        llm_model="qwen3-4b",
        llm_mode="mock",
        llm_url="http://localhost:8000/v1",
        llm_api_key="not-needed",
        llm_temperature=0.0,
        llm_cache_mode="off",
        llm_cache_dir="unused",
        extraction_max_chars=1000,
        extraction_max_items=6,
        extraction_max_tokens=500,
        reflect_limit=20,
        disable_hypothesis_generation=True,
        dream_enable_procedure_detection=False,
        dream_enable_question_generation=False,
        dream_enable_entity_consolidation=False,
        dream_enable_predicate_consolidation=False,
        disable_hypothesis_validation=True,
        dream_max_hypothesis_validations=0,
        extraction_cache_dir="",
        output="unused.json",
        record_experiences=True,
        ingest_visibility_timeout_s=1.0,
        ingest_visibility_poll_s=0.01,
    )

    out = await module._run(args)
    assert out["ingested_experience_count"] == 2
    assert out["ingest_visibility_ok"] is True
    assert out["ingest_batch_state"] == "SUCCEEDED"
    assert out["ingest_batch_failed_count"] == 0
    assert out["retrieval_sanity"]["direct_experience_visibility"]["visible"] == 2
    assert out["retrieval_sanity"]["extraction_progress"] == {
        "extracted": 2,
        "unextracted": 0,
        "total": 2,
    }
    assert out["retrieval_sanity"]["unprocessed_experiences_count"] == 0
    assert out["retrieval_sanity"]["unextracted_experiences_count"] == 0
    assert out["claim_strategy"]["temporal_implication_count"] == 0
    assert out["claim_strategy"]["testable_hypothesis_rate"] is None
    assert out["claim_strategy"]["avg_candidates_per_observation"] == 0.0
    assert out["claim_strategy"]["llm_escalation_rate"] == 0.0
    assert out["claim_strategy"]["gates"]["provenance_completeness_gte_0_98"] is False


@pytest.mark.asyncio
async def test_run_batch_extract_reflect_blocks_weak_local_extraction_model(monkeypatch) -> None:
    module = _load_script_module(
        "run_batch_extract_reflect_policy_module",
        "scripts/run_batch_extract_reflect.py",
    )
    monkeypatch.setattr(
        module,
        "_load_docs",
        lambda doc_glob, max_docs: [("doc1", "x" * 120)][:max_docs],
    )

    args = argparse.Namespace(
        docs_glob="unused",
        max_docs=1,
        llm_mode="local",
        llm_model="qwen3-4b",
        allow_weak_extraction_model=False,
    )

    with pytest.raises(RuntimeError, match="disabled for extraction"):
        await module._run(args)


@pytest.mark.asyncio
async def test_run_batch_extract_reflect_fails_early_when_local_model_not_ready(monkeypatch) -> None:
    module = _load_script_module(
        "run_batch_extract_reflect_preflight_module",
        "scripts/run_batch_extract_reflect.py",
    )

    monkeypatch.setattr(
        module,
        "ensure_local_chat_model_ready",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("preflight failed")),  # noqa: ARG005
    )
    monkeypatch.setattr(
        module,
        "_load_docs",
        lambda doc_glob, max_docs: (_ for _ in ()).throw(AssertionError("should not load docs")),  # noqa: ARG005
    )

    args = argparse.Namespace(
        docs_glob="unused",
        max_docs=1,
        llm_mode="local",
        llm_model="qwen3-30b",
        llm_url="http://127.0.0.1:1234/v1",
        llm_api_key="not-needed",
        allow_weak_extraction_model=False,
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        await module._run(args)


@pytest.mark.asyncio
async def test_benchmark_reflection_smoke_reports_procedure_and_working_counts(monkeypatch) -> None:
    module = _load_script_module(
        "benchmark_reflection_smoke_module",
        "scripts/benchmark_reflection.py",
    )

    monkeypatch.setattr(module, "SiliconMemory", _FakeMemory)
    monkeypatch.setattr(module, "ReflectionEngine", _FakeEngine)

    args = argparse.Namespace(
        db_path="silicon_memory.db",
        db_grpc_host="127.0.0.1",
        db_grpc_port=8643,
        db_retry_attempts=1,
        db_request_timeout_s=2.0,
        user_id="u",
        tenant_id="t",
        cycles=0,
        max_experiences=10,
        include_dream=True,
        dream_enable_procedure_detection=True,
        dream_enable_question_generation=True,
        dream_enable_entity_consolidation=False,
        dream_enable_predicate_consolidation=False,
        disable_hypothesis_generation=True,
        disable_hypothesis_validation=True,
        dream_max_hypothesis_validations=0,
        extraction_cache_dir="",
        llm_mode="mock",
        llm_url="http://localhost:8000/v1",
        llm_model="qwen3-4b",
        llm_api_key="not-needed",
        llm_cache_mode="off",
        llm_cache_dir="unused",
        output="unused.json",
    )

    out = await module._run(args)
    assert out["dream"]["questions_generated"] == 1
    assert out["dream"]["procedures_created"] == 1
    assert out["retrieval_sanity"]["procedures_count"] >= 1
    assert out["retrieval_sanity"]["open_questions_count"] >= 1


@pytest.mark.asyncio
async def test_benchmark_reflection_fails_early_when_local_model_not_ready(monkeypatch) -> None:
    module = _load_script_module(
        "benchmark_reflection_preflight_module",
        "scripts/benchmark_reflection.py",
    )

    monkeypatch.setattr(
        module,
        "ensure_local_chat_model_ready",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("preflight failed")),  # noqa: ARG005
    )

    args = argparse.Namespace(
        user_id="u",
        tenant_id="t",
        llm_mode="local",
        llm_model="qwen3-30b",
        llm_url="http://127.0.0.1:1234/v1",
        llm_api_key="not-needed",
        allow_weak_extraction_model=False,
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        await module._run(args)
