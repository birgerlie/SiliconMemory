"""Lightweight LLM provider for e2e tests using a local OpenAI-compatible server."""

from __future__ import annotations

import httpx


class LocalLLMProvider:
    """LLM provider that talks to a local OpenAI-compatible API.

    Implements the complete() interface used by:
    - DecisionBriefGenerator._synthesize_with_llm
    - MeetingTranscriptAdapter._segment_with_llm
    - MeetingTranscriptAdapter._extract_action_items_llm
    - NewsArticleAdapter._extract_claims_llm
    - SnapshotService._generate_llm_summary

    Usage:
        provider = LocalLLMProvider("http://localhost:8000")
        text = await provider.complete("Hello", system="You are helpful.")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model  # Auto-detected if None
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _detect_model(self) -> str:
        """Auto-detect a loaded model from the server's /v1/models endpoint."""
        if self._model:
            return self._model
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            # Prefer a loaded model
            for m in models:
                if m.get("status") == "loaded":
                    self._model = m["id"]
                    return self._model
            if models:
                self._model = models[0].get("id", "default")
                return self._model
        except Exception:
            pass
        self._model = "default"
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate a completion via OpenAI-compatible chat API.

        Args:
            prompt: The user prompt
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text
        """
        model = await self._detect_model()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"]

    async def close(self):
        await self._client.aclose()
