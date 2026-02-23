"""Tests for local LLM preflight checks."""

from __future__ import annotations

import pytest

from silicon_memory.llm import preflight


class _FakeCreateOK:
    async def create(self, **kwargs):  # noqa: ANN003
        return {"ok": True, "kwargs": kwargs}


class _FakeCreateError:
    def __init__(self, message: str) -> None:
        self._message = message

    async def create(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        raise RuntimeError(self._message)


class _FakeClient:
    def __init__(self, create_impl) -> None:  # noqa: ANN001
        self.chat = type(
            "_Chat",
            (),
            {"completions": type("_Completions", (), {"create": create_impl})()},
        )()


@pytest.mark.asyncio
async def test_local_preflight_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "AsyncOpenAI",
        lambda **kwargs: _FakeClient(_FakeCreateOK().create),  # noqa: ARG005
    )
    await preflight.ensure_local_chat_model_ready(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen3-30b",
    )


@pytest.mark.asyncio
async def test_local_preflight_no_models_loaded_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "AsyncOpenAI",
        lambda **kwargs: _FakeClient(  # noqa: ARG005
            _FakeCreateError("Error code: 400 - No models loaded").create
        ),
    )
    with pytest.raises(RuntimeError, match="no model is loaded"):
        await preflight.ensure_local_chat_model_ready(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen3-30b",
        )


@pytest.mark.asyncio
async def test_local_preflight_connection_error_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "AsyncOpenAI",
        lambda **kwargs: _FakeClient(  # noqa: ARG005
            _FakeCreateError("Connection refused").create
        ),
    )
    with pytest.raises(RuntimeError, match="cannot reach local OpenAI endpoint"):
        await preflight.ensure_local_chat_model_ready(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen3-30b",
        )
