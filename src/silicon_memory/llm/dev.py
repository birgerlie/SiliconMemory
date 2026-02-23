"""Development LLM helpers: mock provider and replay cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class MockLLM:
    """Deterministic no-network LLM stub for development pipelines."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        _ = (prompt, max_tokens, temperature, system)
        return '{"mock": true}'

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        return await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        max_tokens: int | None = None,
    ) -> Any:
        _ = (prompt, max_tokens)
        payload = self._default_payload(schema)
        if hasattr(schema, "model_validate"):
            return schema.model_validate(payload)
        return payload

    def _default_payload(self, schema: type) -> dict[str, Any]:
        schema_name = getattr(schema, "__name__", "")
        if schema_name == "ExtractionResult":
            return {
                "facts": [],
                "relationships": [],
                "arguments": [],
                "events": [],
            }
        if schema_name == "HypothesisSet":
            return {"hypotheses": []}

        fields = getattr(schema, "model_fields", {})
        if isinstance(fields, dict) and fields:
            out: dict[str, Any] = {}
            for name in fields:
                if name.endswith("s") or name in {
                    "facts",
                    "relationships",
                    "arguments",
                    "events",
                    "hypotheses",
                    "aliases",
                    "clusters",
                    "groups",
                }:
                    out[name] = []
                elif name in {"confidence", "score", "probability"}:
                    out[name] = 0.5
                elif name.startswith("is_") or name.startswith("has_"):
                    out[name] = False
                else:
                    out[name] = ""
            return out
        return {}


class CachedLLM:
    """File-backed replay cache wrapper for any async LLM client."""

    def __init__(
        self,
        llm: Any,
        cache_dir: str | Path,
        mode: str = "record_replay",
    ) -> None:
        if mode not in {"record_replay", "replay_only"}:
            raise ValueError("mode must be 'record_replay' or 'replay_only'")
        self._llm = llm
        self._mode = mode
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        payload = {
            "fn": "generate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
        }
        path = self._entry_path(payload)
        cached = self._read(path)
        if cached is not None:
            return str(cached.get("text", ""))
        if self._mode == "replay_only":
            raise RuntimeError(f"LLM cache miss (replay_only): {path.name}")
        try:
            text = await self._llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
            )
        except TypeError:
            # Some providers (e.g. OpenAI-compatible wrappers) do not
            # accept a `system` kwarg on generate().
            text = await self._llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        self._write(path, {"text": text})
        return text

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "fn": "complete",
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        path = self._entry_path(payload)
        cached = self._read(path)
        if cached is not None:
            return str(cached.get("text", ""))
        if self._mode == "replay_only":
            raise RuntimeError(f"LLM cache miss (replay_only): {path.name}")
        text = await self._llm.complete(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._write(path, {"text": text})
        return text

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        max_tokens: int | None = None,
    ) -> Any:
        payload = {
            "fn": "generate_structured",
            "prompt": prompt,
            "schema": getattr(schema, "__name__", str(schema)),
            "max_tokens": max_tokens,
        }
        path = self._entry_path(payload)
        cached = self._read(path)
        if cached is not None:
            data = cached.get("payload", {})
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            return data
        if self._mode == "replay_only":
            raise RuntimeError(f"LLM cache miss (replay_only): {path.name}")

        result = await self._llm.generate_structured(
            prompt=prompt,
            schema=schema,
            max_tokens=max_tokens,
        )
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif hasattr(result, "dict"):
            data = result.dict()
        elif isinstance(result, dict):
            data = result
        else:
            data = {"value": str(result)}
        self._write(path, {"payload": data})
        return result

    def _entry_path(self, payload: dict[str, Any]) -> Path:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
