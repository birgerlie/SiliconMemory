"""Model policy helpers for task-specific runtime guardrails."""

from __future__ import annotations

from typing import Any

_WEAK_EXTRACTION_MODEL_MARKERS: tuple[str, ...] = (
    "qwen3-4b",
    "qwen/qwen3-4b",
)


def is_weak_extraction_model_name(model_name: str) -> bool:
    """Return True for models that are considered too weak for extraction."""
    norm = str(model_name or "").strip().lower()
    if not norm:
        return False
    return any(marker in norm for marker in _WEAK_EXTRACTION_MODEL_MARKERS)


def detect_llm_model_name(llm: Any, max_depth: int = 5) -> str:
    """Best-effort model-name extraction from wrapped LLM objects."""
    if isinstance(llm, str):
        return llm
    if llm is None:
        return ""

    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(llm, 0)]
    wrapper_attrs = ("_llm", "llm", "_provider", "provider", "wrapped", "inner", "delegate")

    while queue:
        obj, depth = queue.pop(0)
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        model = _model_from_object(obj)
        if model:
            return model

        if depth >= max_depth:
            continue

        for attr in wrapper_attrs:
            child = getattr(obj, attr, None)
            if child is None:
                continue
            queue.append((child, depth + 1))

    return ""


def ensure_extraction_model_allowed(
    llm: Any,
    *,
    allow_weak_extraction_model: bool = False,
) -> str:
    """Raise when extraction is attempted with a blocked weak model."""
    model_name = detect_llm_model_name(llm)
    if is_weak_extraction_model_name(model_name) and not allow_weak_extraction_model:
        raise RuntimeError(
            "Extraction policy: qwen3-4b-class models are disabled for extraction. "
            "Use qwen3-30b or qwen3-80b, or enable explicit override for diagnostics.",
        )
    return model_name


def _model_from_object(obj: Any) -> str:
    for attr in ("model", "_model"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    cfg = getattr(obj, "_config", None)
    if cfg is not None:
        value = getattr(cfg, "model", None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""
