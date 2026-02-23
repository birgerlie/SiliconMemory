"""Tests for LLM model policy helpers."""

from __future__ import annotations

from silicon_memory.llm.model_policy import (
    detect_llm_model_name,
    ensure_extraction_model_allowed,
    is_weak_extraction_model_name,
)


class _BaseProvider:
    model = "qwen3-4b"


class _CachedWrapper:
    def __init__(self, llm) -> None:  # noqa: ANN001
        self._llm = llm


class _SchedulerWrapper:
    def __init__(self, provider) -> None:  # noqa: ANN001
        self._provider = provider


def test_is_weak_extraction_model_name() -> None:
    assert is_weak_extraction_model_name("qwen3-4b")
    assert is_weak_extraction_model_name("qwen/qwen3-4b-2507")
    assert not is_weak_extraction_model_name("qwen3-30b")


def test_detect_llm_model_name_from_wrappers() -> None:
    wrapped = _SchedulerWrapper(_CachedWrapper(_BaseProvider()))
    assert detect_llm_model_name(wrapped) == "qwen3-4b"


def test_ensure_extraction_model_allowed_enforces_policy() -> None:
    wrapped = _SchedulerWrapper(_CachedWrapper(_BaseProvider()))
    try:
        ensure_extraction_model_allowed(wrapped, allow_weak_extraction_model=False)
        assert False, "expected RuntimeError for weak extraction model"
    except RuntimeError as exc:
        assert "disabled for extraction" in str(exc)

    model = ensure_extraction_model_allowed(
        wrapped,
        allow_weak_extraction_model=True,
    )
    assert model == "qwen3-4b"
