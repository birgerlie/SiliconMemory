"""Ensure relationship extraction emits bidirectional edges."""

from __future__ import annotations

import pytest

from silicon_memory.reflection.llm_extractor import (
    ExtractedRelationship,
    ExtractionResult,
    LLMPatternExtractor,
)
from silicon_memory.reflection.types import ReflectionConfig


class _StubLLM:
    async def generate_structured(self, prompt, schema, max_tokens=2048):  # noqa: ANN001, ANN201
        return ExtractionResult(
            relationships=[
                ExtractedRelationship(
                    person1="Alice",
                    relationship="works with",
                    person2="Bob",
                    context="Project Delta",
                    source="doc-1",
                    confidence=0.9,
                )
            ]
        )

    async def generate(self, prompt, max_tokens=2048, temperature=0.3):  # noqa: ANN001, ANN201
        return "{}"


@pytest.mark.asyncio
async def test_relationships_are_bi_directional():
    extractor = LLMPatternExtractor(
        memory=object(),  # not used in this path
        llm=_StubLLM(),
        config=ReflectionConfig(),
    )

    patterns = await extractor._extract_from_text(  # noqa: SLF001
        text="[SRC:doc-1]\nAlice worked with Bob on Project Delta.",
        evidence_ids=[],
        source_lookup={"doc-1": {"document_id": "doc-1"}},
        known_context="",
    )

    rels = [p for p in patterns if p.type.value == "relationship"]
    assert len(rels) == 2

    pairs = {(p.subject, p.predicate, p.object, p.context.get("relationship_direction")) for p in rels}
    assert ("Alice", "works with", "Bob", "forward") in pairs
    assert ("Bob", "works with", "Alice", "reverse") in pairs
