"""Local LLM preflight checks for script workflows."""

from __future__ import annotations

from openai import AsyncOpenAI


def _error_text(exc: Exception) -> str:
    try:
        return str(exc)
    except Exception:
        return exc.__class__.__name__


async def ensure_local_chat_model_ready(
    *,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
    timeout_s: float = 8.0,
) -> None:
    """Fail fast when a local OpenAI-compatible model is not usable.

    Sends a tiny chat completion to verify both endpoint reachability and
    requested model readiness before expensive ingest/extract work begins.
    """
    model_name = str(model or "").strip()
    if not model_name:
        raise RuntimeError("Local LLM preflight failed: model is empty.")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=float(timeout_s),
    )

    try:
        await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        return
    except Exception as exc:
        msg = _error_text(exc)
        low = msg.lower()

        if "no models loaded" in low:
            raise RuntimeError(
                "Local LLM preflight failed: no model is loaded on "
                f"{base_url}. Requested model '{model_name}'. "
                "Load it first (example: "
                f"`lms load {model_name} --identifier {model_name} -y`)."
            ) from exc

        connection_markers = (
            "connection refused",
            "failed to connect",
            "connection error",
            "timed out",
            "timeout",
            "name or service not known",
        )
        if any(marker in low for marker in connection_markers):
            raise RuntimeError(
                "Local LLM preflight failed: cannot reach local OpenAI endpoint at "
                f"{base_url}. Start your local LLM server and verify `--llm-url`."
            ) from exc

        model_missing_markers = (
            "model not found",
            "unknown model",
            "does not exist",
            "invalid model",
            "not available",
        )
        if any(marker in low for marker in model_missing_markers):
            raise RuntimeError(
                "Local LLM preflight failed: requested model "
                f"'{model_name}' is not available on {base_url}. "
                "Check model identifier and load it before running scripts."
            ) from exc

        raise RuntimeError(
            "Local LLM preflight failed for "
            f"model '{model_name}' at {base_url}: {msg}"
        ) from exc
