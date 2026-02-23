"""Tests for development LLM helpers (mock + replay cache)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from silicon_memory.llm.dev import CachedLLM, MockLLM


class _ExtractionSchema(BaseModel):
    facts: list[dict] = []
    relationships: list[dict] = []
    arguments: list[dict] = []
    events: list[dict] = []


class _EchoLLM:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.complete_calls = 0
        self.structured_calls = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        _ = (max_tokens, temperature, system)
        self.generate_calls += 1
        return f"gen:{prompt}"

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        _ = (system, temperature, max_tokens)
        self.complete_calls += 1
        return f"complete:{prompt}"

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        max_tokens: int | None = None,
    ) -> BaseModel:
        _ = (prompt, max_tokens)
        self.structured_calls += 1
        return schema.model_validate(
            {
                "facts": [{"subject": "A"}],
                "relationships": [],
                "arguments": [],
                "events": [],
            },
        )


class _GenerateNoSystemLLM:
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        _ = (max_tokens, temperature)
        return f"no-system:{prompt}"


@pytest.mark.asyncio
async def test_mock_llm_generate_structured_for_extraction() -> None:
    llm = MockLLM()
    out = await llm.generate_structured("extract", _ExtractionSchema)
    assert out.facts == []
    assert out.relationships == []
    assert out.arguments == []
    assert out.events == []


@pytest.mark.asyncio
async def test_cached_llm_reuses_generate_hits(tmp_path: Path) -> None:
    base = _EchoLLM()
    llm = CachedLLM(base, cache_dir=tmp_path, mode="record_replay")

    a = await llm.generate("hello")
    b = await llm.generate("hello")

    assert a == "gen:hello"
    assert b == "gen:hello"
    assert base.generate_calls == 1


@pytest.mark.asyncio
async def test_cached_llm_replay_only_raises_on_miss(tmp_path: Path) -> None:
    base = _EchoLLM()
    llm = CachedLLM(base, cache_dir=tmp_path, mode="replay_only")

    with pytest.raises(RuntimeError):
        await llm.generate("missing")


@pytest.mark.asyncio
async def test_cached_llm_generate_fallback_without_system_kwarg(tmp_path: Path) -> None:
    llm = CachedLLM(_GenerateNoSystemLLM(), cache_dir=tmp_path, mode="record_replay")
    out = await llm.generate("hello")
    assert out == "no-system:hello"
