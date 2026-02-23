"""Tests for resilient chunk-split extraction fallback."""

from __future__ import annotations

from uuid import uuid4

import pytest

from silicon_memory.core.types import Experience
from silicon_memory.reflection.llm_extractor import (
    ExtractedArgument,
    ExtractedEvent,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionResult,
    LLMPatternExtractor,
)
from silicon_memory.reflection.types import Pattern, PatternType, ReflectionConfig


class _SplitRetryExtractor(LLMPatternExtractor):
    async def _extract_from_text(  # noqa: D401
        self,
        text: str,
        evidence_ids: list,  # noqa: ANN001
        source_lookup: dict,  # noqa: ANN001
        known_context: str = "",
    ) -> list[Pattern]:
        # Simulate large combined chunk parse failure.
        if text.count("[SRC:") > 1:
            return []
        return [
            Pattern(
                type=PatternType.FACT,
                description="A relates_to B",
                subject="A",
                predicate="relates_to",
                object="B",
                confidence=0.8,
                evidence=list(evidence_ids),
            ),
        ]


class _WeakModel:
    model = "qwen3-4b"


class _StrongModel:
    model = "qwen3-30b"


class _LimitHitModel:
    model = "qwen3-30b"

    async def generate_structured(  # noqa: D401
        self,
        prompt: str,  # noqa: ARG002
        schema: type,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> ExtractionResult:
        return ExtractionResult(
            facts=[
                ExtractedFact(subject="A", predicate="rel", object="B"),
                ExtractedFact(subject="C", predicate="rel", object="D"),
            ],
            relationships=[
                ExtractedRelationship(person1="X", relationship="knows", person2="Y"),
                ExtractedRelationship(person1="Y", relationship="knows", person2="Z"),
            ],
            arguments=[ExtractedArgument(claim="c1", evidence="e1", rhetoric="logos", actor="x")],
            events=[ExtractedEvent(date="2024-01-01", event="ev1", actors=["x"])],
        )


class _MissingDimensionRetryModel:
    model = "qwen3-30b"

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(  # noqa: D401
        self,
        prompt: str,  # noqa: ARG002
        schema: type,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> ExtractionResult:
        self.calls += 1
        if self.calls == 1:
            return ExtractionResult(
                facts=[ExtractedFact(subject="A", predicate="rel", object="B")],
                relationships=[],
                arguments=[],
                events=[],
            )
        return ExtractionResult(
            facts=[],
            relationships=[
                ExtractedRelationship(person1="Alice", relationship="knows", person2="Bob"),
            ],
            arguments=[],
            events=[ExtractedEvent(date="2024-01-01", event="met", actors=["Alice", "Bob"])],
        )


class _UnavailableModelExtractor(LLMPatternExtractor):
    async def _extract_from_text(  # noqa: D401
        self,
        text: str,  # noqa: ARG002
        evidence_ids: list,  # noqa: ANN001, ARG002
        source_lookup: dict,  # noqa: ANN001, ARG002
        known_context: str = "",
    ) -> list[Pattern]:
        _ = known_context
        raise RuntimeError(
            "Error code: 400 - {'error': {'message': \"No models loaded. "
            "Please load a model in the developer page or use the 'lms load' command.\"}}",
        )


@pytest.mark.asyncio
async def test_extract_chunk_splits_when_empty() -> None:
    extractor = _SplitRetryExtractor(
        memory=object(),  # not used in this test path
        llm=object(),  # not used in this test path
        config=ReflectionConfig(),
    )
    exps = [
        Experience(id=uuid4(), content="Doc one."),
        Experience(id=uuid4(), content="Doc two."),
    ]
    texts = [
        "[SRC:doc-1]\nDoc one.",
        "[SRC:doc-2]\nDoc two.",
    ]
    patterns = await extractor._extract_chunk(texts, exps, known_context="")  # noqa: SLF001
    assert len(patterns) == 2


@pytest.mark.asyncio
async def test_extract_patterns_flat_blocks_weak_model_policy() -> None:
    extractor = LLMPatternExtractor(
        memory=object(),
        llm=_WeakModel(),
        config=ReflectionConfig(),
    )
    exps = [
        Experience(id=uuid4(), content="A" * 120),
    ]
    with pytest.raises(RuntimeError, match="disabled for extraction"):
        await extractor.extract_patterns_flat(exps)


@pytest.mark.asyncio
async def test_extract_patterns_flat_fails_fast_when_model_unavailable() -> None:
    extractor = _UnavailableModelExtractor(
        memory=object(),
        llm=_StrongModel(),
        config=ReflectionConfig(),
    )
    exps = [
        Experience(id=uuid4(), content="B" * 120),
    ]
    with pytest.raises(RuntimeError, match="no loaded/available model"):
        await extractor.extract_patterns_flat(exps)


@pytest.mark.asyncio
async def test_extract_patterns_flat_reports_limit_hit_diagnostics() -> None:
    extractor = LLMPatternExtractor(
        memory=object(),
        llm=_LimitHitModel(),
        config=ReflectionConfig(extraction_max_items=2),
    )
    exps = [
        Experience(id=uuid4(), content="C" * 120),
    ]
    patterns = await extractor.extract_patterns_flat(exps)
    assert patterns
    diag = extractor.extraction_diagnostics()
    assert diag["llm_extraction_calls"] >= 1
    # facts/relationships are exactly at max_items => potential clipping signal.
    assert diag["limit_hit_chunks"].get("facts", 0) >= 1
    assert diag["limit_hit_chunks"].get("relationships", 0) >= 1
    assert diag["limit_hit_chunks_total"] >= 2


@pytest.mark.asyncio
async def test_extract_patterns_flat_retries_missing_dimensions() -> None:
    model = _MissingDimensionRetryModel()
    extractor = LLMPatternExtractor(
        memory=object(),
        llm=model,
        config=ReflectionConfig(),
    )
    exps = [
        Experience(id=uuid4(), content="D" * 120),
    ]
    patterns = await extractor.extract_patterns_flat(exps)
    assert patterns
    assert any(p.type == PatternType.RELATIONSHIP for p in patterns)
    assert any(p.type == PatternType.TIMELINE_EVENT for p in patterns)
    diag = extractor.extraction_diagnostics()
    assert diag["llm_extraction_calls"] == 2
